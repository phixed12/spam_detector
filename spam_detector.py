#!/usr/bin/env python3
"""
Spam Business Detector
Scores and flags potential spam/fake business listings from CSV or XLSX input.
Supports Birdeye reseller export column naming conventions.
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import NamedTuple

import asyncio

import aiohttp
import numpy as np
import pandas as pd
import pgeocode
import phonenumbers
import tldextract
import yaml
from rapidfuzz import fuzz  # kept for any future scalar use; cdist removed

# ---------------------------------------------------------------------------
# Column name aliases — map flexible input headers to canonical names.
# Aliases use the *normalized* form of the column name (see _normalize_col_name).
# Birdeye exports use "Field > SubField" naming; normalization strips the ">".
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "business_name": [
        "business_name", "name", "company", "company_name", "biz_name", "business",
        "location_name",
    ],
    "address": [
        "address", "street", "street_address", "addr", "address1",
        "address_line_1", "address_line1",
    ],
    "city":  ["city", "town", "municipality"],
    "state": ["state", "st", "province", "region"],
    "zip":   ["zip", "zip_code", "zipcode", "postal_code", "postal"],
    "phone": [
        "phone", "phone_number", "telephone", "tel", "mobile", "cell",
        "local_phone_phone_number",
        "main_phone_phone_number",
        "local_phone", "main_phone",
    ],
    "website": [
        "website", "url", "web", "site", "domain", "webpage",
        "website_url_url", "website_url",
    ],
    "industry": [
        "industry", "category", "type", "business_type", "vertical", "service_type",
    ],
    "email": [
        "email", "email_address", "e_mail", "contact_email", "emails",
    ],
    "owner_name": [
        "owner_name", "owner", "contact_name", "contact", "rep_name",
    ],
    "date_added": [
        "date_added", "created_at", "created_date", "added_date",
        "submission_date", "date",
    ],
    "reseller_id":   ["reseller_id", "reseller", "partner_id", "partner"],
    "reseller_name": ["reseller_name", "partner_name", "agent"],
    "photo_gallery": ["photo_gallery_url", "photo_gallery", "photos", "gallery_url"],
    "x_handle":        ["x_handle", "twitter_handle", "twitter", "x"],
    "facebook_url":    ["facebook_page_url", "facebook_url", "facebook", "fb"],
    "instagram_handle":["instagram_handle", "instagram", "ig"],
    "year_established": [
        "year_established", "established", "founded", "year_founded", "founding_year",
    ],
    "hours_monday":    ["hours_monday",    "monday_hours",    "mon_hours"],
    "hours_tuesday":   ["hours_tuesday",   "tuesday_hours",   "tue_hours"],
    "hours_wednesday": ["hours_wednesday", "wednesday_hours", "wed_hours"],
    "hours_thursday":  ["hours_thursday",  "thursday_hours",  "thu_hours"],
    "hours_friday":    ["hours_friday",    "friday_hours",    "fri_hours"],
    "hours_saturday":  ["hours_saturday",  "saturday_hours",  "sat_hours"],
    "hours_sunday":    ["hours_sunday",    "sunday_hours",    "sun_hours"],
    "description": [
        "description", "desc", "business_description", "about", "overview",
    ],
    "descrp": ["descrp"],
    "address_hidden":  ["address_hidden", "hide_address", "hidden_address"],
    "service_area": [
        "service_area_places_name", "service_area_places",
        "service_area", "service_areas", "coverage_area",
    ],
    "keywords":        ["keywords", "tags", "keyword", "seo_keywords"],
    "landing_page_url":["landing_page_url", "landing_page", "landing_url", "lp_url"],
}

HOURS_DAYS   = ["hours_monday","hours_tuesday","hours_wednesday","hours_thursday",
                "hours_friday","hours_saturday","hours_sunday"]
SOCIAL_KEYS  = ["x_handle", "facebook_url", "instagram_handle"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RuleResult(NamedTuple):
    flag:   str
    weight: int
    mask:   pd.Series   # bool — True where the rule fired
    detail: pd.Series   # str  — human-readable explanation (populated where mask=True)


# ---------------------------------------------------------------------------
# Scalar helpers (used for per-value operations and in pre-computation)
# ---------------------------------------------------------------------------

def _normalize_col_name(col: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", col.lower().strip())
    return s.strip("_")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_config(path: str) -> str:
    """Return path if it exists; fall back to spam_config.template.yaml."""
    if os.path.exists(path):
        return path
    template = str(Path(path).parent / "spam_config.template.yaml")
    if os.path.exists(template):
        print(f"[WARN] '{path}' not found — falling back to '{Path(template).name}'")
        return template
    raise FileNotFoundError(
        f"Config not found: '{path}'. "
        f"Copy spam_config.template.yaml to spam_config.yaml to get started."
    )


def normalize_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_legal_suffixes(name: str) -> str:
    return re.sub(
        r"\b(llc|inc|co|corp|ltd|company|companies|group|services|solutions|associates)\b",
        "", name,
    ).strip()


def clean_phone(phone) -> str:
    if not isinstance(phone, str):
        phone = str(phone) if pd.notna(phone) else ""
    return re.sub(r"\D", "", phone)


def extract_domain(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""


def extract_full_domain(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    ext = tldextract.extract(url)
    return ".".join(p for p in [ext.subdomain, ext.domain, ext.suffix] if p)


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def map_columns(df: pd.DataFrame) -> dict:
    actual = {_normalize_col_name(c): c for c in df.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in actual:
                found = actual[alias]
                break
        mapping[canonical] = found
    return mapping


# ---------------------------------------------------------------------------
# Vectorized column helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, col_map: dict, canonical: str,
         default: str = "") -> pd.Series:
    """Return the Series for a canonical name, or a constant-default Series if absent."""
    c = col_map.get(canonical)
    if c is None:
        return pd.Series(default, index=df.index, dtype=str)
    return df[c].fillna(default).astype(str).str.strip()


def _is_high_risk(df: pd.DataFrame, col_map: dict, cfg: dict) -> pd.Series:
    """Boolean Series: True where row is in a high-risk industry."""
    combined = _col(df, col_map, "industry").str.lower() + " " + \
               _col(df, col_map, "business_name").str.lower()
    pattern = "|".join(re.escape(kw.lower()) for kw in cfg["high_risk_industries"])
    if not pattern:
        return pd.Series(False, index=df.index)
    return combined.str.contains(pattern, regex=True, na=False)


def _compute_risk_tier(scores: pd.Series, thresholds: dict) -> pd.Series:
    return pd.Series(
        np.select(
            [scores >= thresholds["high_confidence_spam"],
             scores >= thresholds["likely_spam"],
             scores >= thresholds["review"]],
            ["High Confidence Spam", "Likely Spam", "Review"],
            default="Clean",
        ),
        index=scores.index,
    )


def _join_nonempty(frame: pd.DataFrame, sep: str) -> pd.Series:
    """Join non-empty string values per row across all columns of frame."""
    stacked = frame.stack()
    stacked = stacked[stacked != ""]
    if stacked.empty:
        return pd.Series("", index=frame.index)
    return stacked.groupby(level=0).agg(sep.join).reindex(frame.index, fill_value="")


def _count_col(df: pd.DataFrame, col_map: dict, canonical: str,
               transform=None) -> dict:
    """Value-count dict for a column, with an optional vectorized transform."""
    c = col_map.get(canonical)
    if c is None:
        return {}
    s = df[c].fillna("").astype(str)
    if transform is not None:
        s = transform(s)
    return s[s != ""].value_counts().to_dict()


# ---------------------------------------------------------------------------
# SmartyStreets RDI helpers — concurrent async fetching, module-level cache
# ---------------------------------------------------------------------------

_SMARTY_URL = "https://us-street.api.smartystreets.com/street-address"

# Persists across build_rdi_cache calls within the same process.
# Guarantees no address is ever looked up twice, even across different uploads.
_RDI_CACHE: dict = {}


async def _fetch_one_rdi(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    address: str, city: str, state: str, zipcode: str,
    auth_id: str, auth_token: str,
) -> str:
    """Single async SmartyStreets RDI lookup, rate-limited by semaphore."""
    async with semaphore:
        async with session.get(
            _SMARTY_URL,
            params={"auth-id": auth_id, "auth-token": auth_token,
                    "street": address, "city": city, "state": state,
                    "zipcode": zipcode, "candidates": 1},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
            if data and isinstance(data, list):
                return data[0].get("metadata", {}).get("rdi", "")
            return ""


async def _fetch_rdi_concurrent(
    lookups: list,
    auth_id: str,
    auth_token: str,
    max_concurrent: int,
) -> dict:
    """
    Fire all RDI lookups concurrently (capped at max_concurrent in-flight).
    Returns {address: rdi_value_or_exception}.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = {
            address: _fetch_one_rdi(
                session, semaphore, address, city, state, zipcode, auth_id, auth_token,
            )
            for address, city, state, zipcode in lookups
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return dict(zip(tasks.keys(), results))


def build_rdi_cache(df: pd.DataFrame, col_map: dict, cfg: dict) -> dict:
    """
    Populate and return the per-process RDI address cache.

    - Only addresses not already in _RDI_CACHE are fetched.
    - All new fetches run concurrently via aiohttp / asyncio.gather.
    - Returns a dict covering every high-risk address in this DataFrame.
    """
    global _RDI_CACHE

    auth_id    = (str(cfg.get("smartystreets_auth_id", "")).strip()
                  or os.environ.get("SMARTYSTREETS_AUTH_ID", "").strip())
    auth_token = (str(cfg.get("smartystreets_auth_token", "")).strip()
                  or os.environ.get("SMARTYSTREETS_AUTH_TOKEN", "").strip())
    if not auth_id or not auth_token:
        print("[WARN] SmartyStreets credentials not configured — residential address check skipped.")
        return {}

    addr_col = col_map.get("address")
    if not addr_col:
        return {}

    city_col  = col_map.get("city")
    state_col = col_map.get("state")
    zip_col   = col_map.get("zip")

    hr_mask      = _is_high_risk(df, col_map, cfg)
    hr_df        = df[hr_mask]
    addrs        = hr_df[addr_col].fillna("").astype(str).str.strip()
    unique_addrs = addrs[addrs != ""].unique()

    # Skip addresses already cached — never call the API twice for the same address
    to_fetch = [a for a in unique_addrs if a not in _RDI_CACHE]

    if to_fetch:
        lookups = []
        for address in to_fetch:
            first = hr_df[addrs == address].iloc[0]
            lookups.append((
                address,
                str(first[city_col]).strip()  if city_col  else "",
                str(first[state_col]).strip() if state_col else "",
                str(first[zip_col]).strip()   if zip_col   else "",
            ))

        max_concurrent = int(cfg.get("smartystreets_max_concurrent", 10))
        coro = _fetch_rdi_concurrent(lookups, auth_id, auth_token, max_concurrent)

        # asyncio.run() creates a fresh event loop; if one is already running
        # (e.g. inside Jupyter), fall back to a dedicated thread.
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            raw = asyncio.run(coro)

        for address, result in raw.items():
            if isinstance(result, Exception):
                print(f"[WARN] SmartyStreets API error for '{address}': {result}")
                _RDI_CACHE[address] = ""
            else:
                _RDI_CACHE[address] = result

    return {addr: _RDI_CACHE[addr] for addr in unique_addrs if addr in _RDI_CACHE}


# ---------------------------------------------------------------------------
# Vectorized rule functions
# Each takes the full DataFrame + col_map + cfg (+ optional pre-computed data)
# and returns list[RuleResult].
# ---------------------------------------------------------------------------

def rule_high_risk_industry(df, col_map, cfg) -> list:
    if "high_risk_industry" in cfg.get("disabled_rules", []):
        return []
    mask = _is_high_risk(df, col_map, cfg)
    combined = (_col(df, col_map, "industry").str.lower() + " " +
                _col(df, col_map, "business_name").str.lower())
    kw_pattern = "(" + "|".join(re.escape(kw.lower()) for kw in cfg["high_risk_industries"]) + ")"
    first_match = combined.str.extract(kw_pattern, expand=False).fillna("")
    detail = ("Matched industry keyword: '" + first_match + "'").where(mask, "")
    return [RuleResult("HIGH_RISK_INDUSTRY", cfg["weights"]["high_risk_industry"], mask, detail)]


def rule_generic_name(df, col_map, cfg) -> list:
    results = []
    disabled = cfg.get("disabled_rules", [])
    name_norm = _col(df, col_map, "business_name").str.lower()
    has_name  = name_norm != ""

    if "generic_name" not in disabled:
        kw_pat = "|".join(re.escape(kw.lower()) for kw in cfg["spam_name_keywords"])
        if kw_pat:
            mask = has_name & name_norm.str.contains(kw_pat, regex=True, na=False)
            # str.findall returns lists; convert list to comma-separated for detail
            matched = name_norm.str.findall(kw_pat).apply(
                lambda lst: str(lst) if isinstance(lst, list) else "")
            detail = ("Spam keywords in name: " + matched).where(mask, "")
            results.append(RuleResult("GENERIC_NAME_KEYWORD",
                                      cfg["weights"]["generic_name"], mask, detail))

    if "keyword_stuffed_name" not in disabled:
        ind_pat = "|".join(re.escape(kw.lower()) for kw in cfg["high_risk_industries"])
        if ind_pat:
            hit_lists = name_norm.str.findall(ind_pat)
            hit_count = hit_lists.str.len().fillna(0).astype(int)
            mask = has_name & (hit_count >= 2)
            detail = ("Multiple industry keywords in name: " +
                      hit_lists.apply(lambda l: str(l) if isinstance(l, list) else "")
                      ).where(mask, "")
            results.append(RuleResult("KEYWORD_STUFFED_NAME",
                                      cfg["weights"]["keyword_stuffed_name"], mask, detail))
    return results


def rule_po_box(df, col_map, cfg) -> list:
    if "po_box" in cfg.get("disabled_rules", []):
        return []
    addr = _col(df, col_map, "address").str.upper()
    mask = addr.str.contains(r"\bP\.?\s*O\.?\s*BOX\b", regex=True, na=False)
    return [RuleResult("PO_BOX", cfg["weights"]["po_box"], mask,
                        pd.Series("Address is a PO Box", index=df.index).where(mask, ""))]


def rule_virtual_mailbox(df, col_map, cfg) -> list:
    if "virtual_mailbox" in cfg.get("disabled_rules", []):
        return []
    addr_upper = _col(df, col_map, "address").str.upper()
    virt_pat = "|".join(re.escape(v.upper()) for v in cfg["virtual_mailbox_keywords"])
    if not virt_pat:
        return []
    mask = addr_upper.str.contains(virt_pat, regex=True, na=False)
    matched = addr_upper.str.extract(
        "(" + virt_pat + ")", expand=False).fillna("")
    detail = ("Address matches virtual mailbox provider: '" + matched + "'").where(mask, "")
    return [RuleResult("VIRTUAL_MAILBOX", cfg["weights"]["virtual_mailbox"], mask, detail)]


def rule_shared_address(df, col_map, cfg, shared_address_counts: dict) -> list:
    if "shared_address" in cfg.get("disabled_rules", []):
        return []
    addr_norm = _col(df, col_map, "address").str.lower().str.strip()
    counts    = addr_norm.map(shared_address_counts).fillna(0).astype(int)
    threshold = cfg["shared_address_threshold"]
    mask      = (addr_norm != "") & (counts >= threshold)
    detail    = ("Address shared by " + counts.astype(str) +
                 f" businesses (threshold: {threshold})").where(mask, "")
    return [RuleResult("SHARED_ADDRESS", cfg["weights"]["shared_address"], mask, detail)]


def rule_state_zip_mismatch(df, col_map, cfg) -> list:
    if "state_zip_mismatch" in cfg.get("disabled_rules", []):
        return []
    state    = _col(df, col_map, "state").str.upper().str.strip()
    zip_code = _col(df, col_map, "zip").str.split("-").str[0].str.strip()
    has_both = (state != "") & (zip_code != "")
    if not has_both.any():
        return []
    try:
        geo      = pgeocode.Nominatim("us").query_postal_code(zip_code)
        expected = geo["state_code"].fillna("").str.upper()
        mask     = has_both & (expected != "") & (expected != state)
        detail   = ("ZIP " + zip_code + " belongs to " +
                    expected + ", not " + state).where(mask, "")
        return [RuleResult("STATE_ZIP_MISMATCH", cfg["weights"]["state_zip_mismatch"],
                           mask, detail)]
    except Exception:
        return []


def rule_invalid_phone(df, col_map, cfg) -> list:
    if "invalid_phone" in cfg.get("disabled_rules", []):
        return []
    raw    = _col(df, col_map, "phone")
    digits = raw.str.replace(r"\D", "", regex=True)
    has_phone = raw != ""
    mask   = has_phone & ~digits.str.len().isin([10, 11])
    detail = ("Phone '" + raw + "' has unexpected digit count (" +
              digits.str.len().astype(str) + ")").where(mask, "")
    return [RuleResult("INVALID_PHONE", cfg["weights"]["invalid_phone"], mask, detail)]


def rule_tollfree_phone(df, col_map, cfg) -> list:
    if "voip_tollfree_phone" in cfg.get("disabled_rules", []):
        return []
    raw     = _col(df, col_map, "phone")
    digits  = raw.str.replace(r"\D", "", regex=True)
    valid   = digits.str.len().isin([10, 11])
    area    = digits.str[-10:-7]
    toll_pat= "|".join(re.escape(p) for p in cfg["toll_free_prefixes"])
    mask    = valid & area.str.fullmatch(toll_pat, na=False)
    detail  = ("Toll-free area code: " + area).where(mask, "")
    return [RuleResult("TOLLFREE_PHONE", cfg["weights"]["voip_tollfree_phone"], mask, detail)]


def rule_shared_phone(df, col_map, cfg, shared_phone_counts: dict) -> list:
    if "shared_phone" in cfg.get("disabled_rules", []):
        return []
    digits    = _col(df, col_map, "phone").str.replace(r"\D", "", regex=True)
    counts    = digits.map(shared_phone_counts).fillna(0).astype(int)
    threshold = cfg["shared_phone_threshold"]
    mask      = (digits != "") & (counts >= threshold)
    detail    = ("Phone shared by " + counts.astype(str) +
                 f" businesses (threshold: {threshold})").where(mask, "")
    return [RuleResult("SHARED_PHONE", cfg["weights"]["shared_phone"], mask, detail)]


def rule_area_code_mismatch(df, col_map, cfg) -> list:
    if "area_code_mismatch" in cfg.get("disabled_rules", []):
        return []
    from phonenumbers import geocoder as ph_geocoder
    raw    = _col(df, col_map, "phone")
    digits = raw.str.replace(r"\D", "", regex=True)
    valid  = digits.str.len().isin([10, 11])
    area   = digits.str[-10:-7]
    state  = _col(df, col_map, "state").str.upper()
    city   = _col(df, col_map, "city").str.upper()

    # Look up region only for unique area codes (small set)
    unique_acs = area[valid & (area != "")].unique()
    ac_region  = {}
    for ac in unique_acs:
        try:
            parsed = phonenumbers.parse(f"+1{ac}0000000", "US")
            ac_region[ac] = ph_geocoder.description_for_number(parsed, "en").upper()
        except Exception:
            ac_region[ac] = ""

    region = area.map(ac_region).fillna("")

    # Element-wise substring check (region contains state or city)
    state_in_region = pd.Series(
        [bool(r and s and s in r) for r, s in zip(region, state)], index=df.index)
    city_in_region  = pd.Series(
        [bool(r and c and c in r) for r, c in zip(region, city)],  index=df.index)

    mask   = valid & (region != "") & ~state_in_region & ~city_in_region
    detail = ("Area code " + area + " region '" +
              region + "' may not match state " + state).where(mask, "")
    return [RuleResult("AREA_CODE_MISMATCH", cfg["weights"]["area_code_mismatch"], mask, detail)]


def rule_no_website(df, col_map, cfg) -> list:
    if "no_website" in cfg.get("disabled_rules", []):
        return []
    website  = _col(df, col_map, "website")
    mask     = (website == "") & _is_high_risk(df, col_map, cfg)
    detail   = pd.Series("No website provided for a high-risk industry business",
                         index=df.index).where(mask, "")
    return [RuleResult("NO_WEBSITE", cfg["weights"]["no_website"], mask, detail)]


def rule_website_quality(df, col_map, cfg, shared_domain_counts: dict) -> list:
    """Generic builder, lead-gen domain, shared domain, domain-name mismatch."""
    results  = []
    disabled = cfg.get("disabled_rules", [])
    website  = _col(df, col_map, "website")
    has_site = website != ""

    # Pre-compute domain Series once
    domain      = website.apply(extract_domain)
    full_domain = website.apply(extract_full_domain)

    if "generic_builder_site" not in disabled:
        builder_pat = "|".join(
            re.escape(b) for b in cfg["generic_website_builders"])
        if builder_pat:
            mask   = has_site & full_domain.str.contains(builder_pat, regex=True, na=False)
            detail = ("Website is on a generic builder: " + full_domain).where(mask, "")
            results.append(RuleResult("GENERIC_BUILDER_SITE",
                                      cfg["weights"]["generic_builder_site"], mask, detail))

    lead_pat = "|".join(re.escape(d) for d in cfg["lead_gen_domains"])
    if lead_pat:
        mask   = has_site & full_domain.str.contains(lead_pat, regex=True, na=False)
        detail = ("Website is a lead-gen directory: " + full_domain).where(mask, "")
        results.append(RuleResult("LEAD_GEN_DOMAIN",
                                  cfg["weights"]["shared_domain"], mask, detail))

    if "shared_domain" not in disabled:
        counts    = domain.map(shared_domain_counts).fillna(0).astype(int)
        mask      = has_site & (domain != "") & (counts >= 2)
        detail    = ("Website domain '" + domain + "' shared by " +
                     counts.astype(str) + " businesses").where(mask, "")
        results.append(RuleResult("SHARED_DOMAIN",
                                  cfg["weights"]["shared_domain"], mask, detail))

    # Domain-name mismatch: domain shares no token (>3 chars) with business name
    name_norm = _col(df, col_map, "business_name").str.lower()

    def _mismatch(row_data):
        name, dom = row_data
        if not dom or not name:
            return False
        tokens = {t for t in normalize_text(name).split() if len(t) > 3}
        return bool(tokens) and not any(t in dom for t in tokens)

    mismatch_mask = pd.Series(
        list(map(_mismatch, zip(name_norm, domain))), index=df.index)
    mask   = has_site & (domain != "") & mismatch_mask
    detail = ("Domain '" + domain + "' shares no words with business name").where(mask, "")
    results.append(RuleResult("DOMAIN_NAME_MISMATCH",
                               cfg["weights"]["new_domain"], mask, detail))

    return results


def rule_free_email(df, col_map, cfg) -> list:
    if "free_email" in cfg.get("disabled_rules", []):
        return []
    email      = _col(df, col_map, "email").str.lower()
    email_dom  = email.str.split("@").str[-1]
    free_domains = set(cfg["free_email_domains"])
    mask  = (email != "") & email_dom.isin(free_domains)
    detail = ("Email uses free provider: " + email_dom).where(mask, "")
    return [RuleResult("FREE_EMAIL_PROVIDER", cfg["weights"]["free_email"], mask, detail)]


def rule_autogenerated_email(df, col_map, cfg) -> list:
    if "autogenerated_email" in cfg.get("disabled_rules", []):
        return []
    email = _col(df, col_map, "email").str.lower()
    local = email.str.split("@").str[0]
    digit_ratio = local.str.count(r"\d") / local.str.len().clip(lower=1)
    has_long_run = local.str.contains(r"\d{4,}", regex=True, na=False)
    mask   = (email != "") & ((digit_ratio > 0.5) | ((local.str.len() > 8) & has_long_run))
    detail = ("Email local part appears auto-generated: " + local).where(mask, "")
    return [RuleResult("AUTOGENERATED_EMAIL", cfg["weights"]["autogenerated_email"],
                       mask, detail)]


def rule_shared_email(df, col_map, cfg, shared_email_counts: dict) -> list:
    if "shared_email" in cfg.get("disabled_rules", []):
        return []
    email  = _col(df, col_map, "email").str.lower()
    counts = email.map(shared_email_counts).fillna(0).astype(int)
    mask   = (email != "") & (counts >= 2)
    detail = ("Email shared by " + counts.astype(str) + " businesses").where(mask, "")
    return [RuleResult("SHARED_EMAIL", cfg["weights"]["shared_email"], mask, detail)]


def rule_email_domain_mismatch(df, col_map, cfg) -> list:
    if "email_domain_mismatch" in cfg.get("disabled_rules", []):
        return []
    email      = _col(df, col_map, "email").str.lower()
    website    = _col(df, col_map, "website")
    email_dom  = email.str.split("@").str[-1]
    web_domain = website.apply(extract_domain)
    free_set   = set(cfg["free_email_domains"])
    mask = (
        (email != "") & (website != "") &
        (web_domain != "") & (email_dom != "") &
        ~email_dom.isin(free_set) &
        (email_dom != web_domain)
    )
    detail = ("Email domain '" + email_dom + "' differs from website '" +
              web_domain + "'").where(mask, "")
    return [RuleResult("EMAIL_DOMAIN_MISMATCH", cfg["weights"]["email_domain_mismatch"],
                       mask, detail)]


def rule_exact_duplicate(df, col_map, cfg) -> list:
    if "exact_duplicate_row" in cfg.get("disabled_rules", []):
        return []
    # pd.util.hash_pandas_object is a fast vectorised row hash (MurmurHash3)
    row_hash  = pd.util.hash_pandas_object(df, index=False)
    counts    = row_hash.groupby(row_hash).transform("count")
    first_idx = row_hash.groupby(row_hash).transform("idxmin")
    own_idx   = pd.Series(df.index, index=df.index)

    mask      = counts > 1
    is_first  = first_idx == own_idx

    detail = pd.Series("", index=df.index)
    not_first = mask & ~is_first
    detail[not_first] = "Exact duplicate of row " + first_idx[not_first].astype(str)
    is_first_dup = mask & is_first
    detail[is_first_dup] = "This row has " + (counts[is_first_dup] - 1).astype(str) + " exact duplicate(s)"

    return [RuleResult("EXACT_DUPLICATE", cfg["weights"]["exact_duplicate_row"], mask, detail)]


def _name_blocking_fingerprints(names: pd.Series) -> list:
    """
    Three blocking fingerprints for groupby-based near-duplicate detection.
    Names sharing any fingerprint are near-duplicate candidates.

    FP1 sorted-token set  — "Quick Dallas Tow" == "Dallas Tow Quick"
    FP2 sorted char-trigrams (first 8) — "Locksmith" ≈ "Locksmiths"
    FP3 length-bucket + 5-char prefix  — "Smith Plumbing" ≈ "Smith Plumbers"
    """
    fp_tokens = (
        names.str.split()
             .apply(lambda t: " ".join(sorted(set(t)))
                    if isinstance(t, list) and t else "")
    )

    def _tg(s):
        if not s or len(s) < 3:
            return s
        return "|".join(sorted(set(s[i:i+3] for i in range(len(s) - 2)))[:8])
    fp_trigrams = names.apply(_tg)

    fp_prefix = (names.str.len() // 3).astype(str) + ":" + names.str[:5]

    return [fp_tokens, fp_trigrams, fp_prefix]


def rule_near_duplicate_name(df, col_map, cfg, norm_names: list) -> list:
    """
    Near-duplicate name detection via multi-key fingerprint groupby.
    O(n log n) — no pairwise comparison.
    """
    if "near_duplicate_name" in cfg.get("disabled_rules", []):
        return []

    names    = pd.Series(norm_names, index=df.index)
    has_name = names != ""
    if has_name.sum() < 2:
        return []

    fps      = _name_blocking_fingerprints(names)
    mask     = pd.Series(False, index=df.index)
    detail   = pd.Series("", index=df.index)
    own_idx  = pd.Series(df.index, index=df.index)
    SENTINEL = "\x00"

    for fp in fps:
        # Mask rows with a valid fingerprint that occurs more than once
        keyed  = fp.where(has_name & (fp != ""), other=SENTINEL)
        counts = keyed.groupby(keyed).transform("count")
        is_dup = (keyed != SENTINEL) & (counts > 1)

        new = is_dup & ~mask          # rows not yet described by a previous fp
        if not new.any():
            continue

        first_in_grp = (keyed.where(is_dup)
                             .groupby(keyed.where(is_dup))
                             .transform("idxmin"))

        # Non-first members: reference the first row in their group
        not_first = new & (first_in_grp != own_idx)
        if not_first.any():
            first_ref  = first_in_grp[not_first]               # src_idx → other_idx
            other_names = names.loc[first_ref.values].values    # lookup names at other_idx
            detail[not_first] = (
                "Near-duplicate name with row " + first_ref.astype(str)
                + ": '" + pd.Series(other_names, index=first_ref.index) + "'"
            )

        # First members: note how many others share their fingerprint
        is_first = new & (first_in_grp == own_idx)
        if is_first.any():
            n_others = (counts[is_first] - 1).astype(str)
            detail[is_first] = (
                "Near-duplicate name; shares fingerprint with "
                + n_others + " other row(s) in this batch"
            )

        mask |= is_dup

    return [RuleResult("NEAR_DUPLICATE_NAME", cfg["weights"]["near_duplicate_name"],
                       mask, detail)]


def rule_same_owner(df, col_map, cfg) -> list:
    if "same_owner_multi_biz" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("owner_name"):
        return []
    owner    = _col(df, col_map, "owner_name").apply(normalize_text)
    industry = _col(df, col_map, "industry")
    counts   = owner.map(owner[owner != ""].value_counts()).fillna(0).astype(int)
    mask     = (owner != "") & (industry != "") & (counts >= 2)
    detail   = ("Owner '" + owner + "' appears across " +
                counts.astype(str) + " businesses").where(mask, "")
    return [RuleResult("SAME_OWNER_MULTI_BIZ", cfg["weights"]["same_owner_multi_biz"],
                       mask, detail)]


def rule_batch_submission(df, col_map, cfg) -> list:
    if "batch_submission" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("date_added"):
        return []
    dates    = _col(df, col_map, "date_added").str[:10]
    reseller = _col(df, col_map, "reseller_id")
    if (reseller == "").all():
        reseller = _col(df, col_map, "reseller_name")
    reseller = reseller.replace("", "UNKNOWN")
    keys     = reseller + "::" + dates
    valid    = (dates != "") & (dates != "nan")
    counts   = keys.map(keys[valid].value_counts()).fillna(0).astype(int)
    threshold = cfg["batch_submission_threshold"]
    mask     = valid & (counts >= threshold)
    detail   = ("Reseller '" + reseller + "' submitted " + counts.astype(str) +
                " listings on " + dates + f" (threshold: {threshold})").where(mask, "")
    return [RuleResult("BATCH_SUBMISSION", cfg["weights"]["batch_submission"], mask, detail)]


def rule_no_photos(df, col_map, cfg) -> list:
    if "no_photos" in cfg.get("disabled_rules", []) or not col_map.get("photo_gallery"):
        return []
    mask   = _col(df, col_map, "photo_gallery") == ""
    detail = pd.Series("Photo Gallery URL is blank", index=df.index).where(mask, "")
    return [RuleResult("NO_PHOTOS", cfg["weights"]["no_photos"], mask, detail)]


def rule_no_social(df, col_map, cfg) -> list:
    if "no_social" in cfg.get("disabled_rules", []):
        return []
    if not any(col_map.get(k) for k in SOCIAL_KEYS):
        return []
    has_any = pd.Series(False, index=df.index)
    for k in SOCIAL_KEYS:
        has_any = has_any | (_col(df, col_map, k) != "")
    mask   = ~has_any
    detail = pd.Series("All social media fields (X, Facebook, Instagram) are blank",
                       index=df.index).where(mask, "")
    return [RuleResult("NO_SOCIAL_PRESENCE", cfg["weights"]["no_social"], mask, detail)]


def rule_year_established(df, col_map, cfg) -> list:
    if "year_established" in cfg.get("disabled_rules", []) or not col_map.get("year_established"):
        return []
    results = []
    raw     = _col(df, col_map, "year_established")
    col_present = raw != ""

    blank_mask = ~col_present
    if blank_mask.any():
        results.append(RuleResult(
            "YEAR_ESTABLISHED_MISSING", cfg["weights"]["year_established"],
            blank_mask,
            pd.Series("Year Established is blank", index=df.index).where(blank_mask, ""),
        ))

    has_val = col_present
    year_str = raw.str.extract(r"(\d{4})", expand=False)
    parsed   = pd.to_numeric(year_str, errors="coerce")
    invalid_mask = has_val & parsed.isna()
    if invalid_mask.any():
        results.append(RuleResult(
            "YEAR_ESTABLISHED_INVALID", cfg["weights"]["year_established"],
            invalid_mask,
            ("Year Established could not be parsed: '" + raw + "'").where(invalid_mask, ""),
        ))

    today = date.today()
    months_since = (today.year - parsed.fillna(9999)) * 12 + today.month
    recent_mask  = has_val & ~parsed.isna() & (months_since <= 12)
    if recent_mask.any():
        results.append(RuleResult(
            "RECENTLY_ESTABLISHED", cfg["weights"]["year_established"],
            recent_mask,
            ("Business established within last 12 months (year: " +
             parsed.fillna(0).astype(int).astype(str) + ")").where(recent_mask, ""),
        ))
    return results


def rule_no_hours(df, col_map, cfg) -> list:
    if "no_hours" in cfg.get("disabled_rules", []):
        return []
    if not any(col_map.get(d) for d in HOURS_DAYS):
        return []
    has_hours = pd.Series(False, index=df.index)
    for d in HOURS_DAYS:
        has_hours = has_hours | (_col(df, col_map, d) != "")
    mask   = ~has_hours
    detail = pd.Series("All business hours fields are blank",
                       index=df.index).where(mask, "")
    return [RuleResult("NO_HOURS_PROVIDED", cfg["weights"]["no_hours"], mask, detail)]


def rule_description_quality(df, col_map, cfg) -> list:
    if "description_quality" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("description") and not col_map.get("descrp"):
        return []

    desc = _col(df, col_map, "description")
    alt  = _col(df, col_map, "descrp")
    desc = desc.where(desc != "", other=alt)

    results = []
    blank_mask = desc == ""
    results.append(RuleResult(
        "DESCRIPTION_MISSING", cfg["weights"]["description_missing"], blank_mask,
        pd.Series("Description is blank", index=df.index).where(blank_mask, ""),
    ))

    short_mask = ~blank_mask & (desc.str.strip().str.len() < 20)
    char_counts = desc.str.strip().str.len().astype(str)
    results.append(RuleResult(
        "DESCRIPTION_TOO_SHORT", cfg["weights"]["description_missing"], short_mask,
        ("Description is under 20 characters (" + char_counts + " chars)").where(short_mask, ""),
    ))

    # Keyword-stuffed check: per-row city pattern — apply() unavoidable here
    # because city is a different pattern for every row.
    candidates = ~blank_mask & ~short_mask
    if candidates.any():
        norm_desc = desc.str.lower().str.strip()
        city      = _col(df, col_map, "city").str.lower().str.strip()
        ind_kws   = [kw for kw in cfg["high_risk_industries"] if len(kw) > 3]

        def _stuffed(row_data):
            nd, ct = row_data
            hits = []
            if ct and nd.count(ct) >= 3:
                hits.append(f"city '{ct}' ×{nd.count(ct)}")
            hits += [kw for kw in ind_kws if nd.count(kw.lower()) >= 2]
            return (len(hits) >= 3, str(hits[:5]) if len(hits) >= 3 else "")

        stuffed_results = list(map(_stuffed, zip(norm_desc[candidates], city[candidates])))
        stuffed_mask_vals = pd.Series(False, index=df.index)
        stuffed_detail    = pd.Series("", index=df.index)
        for idx, (fired, det) in zip(df.index[candidates], stuffed_results):
            stuffed_mask_vals[idx] = fired
            stuffed_detail[idx]    = ("Description contains repeated city/service keywords: "
                                      + det) if fired else ""
        results.append(RuleResult(
            "DESCRIPTION_KEYWORD_STUFFED", cfg["weights"]["description_stuffed"],
            stuffed_mask_vals, stuffed_detail,
        ))
    return results


def rule_address_hidden(df, col_map, cfg) -> list:
    if "address_hidden" in cfg.get("disabled_rules", []) or not col_map.get("address_hidden"):
        return []
    hidden = _col(df, col_map, "address_hidden").str.lower()
    is_hidden = hidden.isin(["true", "yes", "1", "y"])
    mask   = is_hidden & _is_high_risk(df, col_map, cfg)
    detail = pd.Series("Address is marked hidden for a high-risk industry business",
                       index=df.index).where(mask, "")
    return [RuleResult("ADDRESS_HIDDEN_HIGH_RISK", cfg["weights"]["address_hidden"],
                       mask, detail)]


def rule_service_area_no_address(df, col_map, cfg) -> list:
    if "service_area_no_address" in cfg.get("disabled_rules", []) or not col_map.get("service_area"):
        return []
    service_area = _col(df, col_map, "service_area")
    address      = _col(df, col_map, "address")
    mask   = (service_area != "") & (address == "")
    detail = ("Service area '" + service_area.str[:60] +
              "' set but no street address provided").where(mask, "")
    return [RuleResult("SERVICE_AREA_NO_ADDRESS", cfg["weights"]["service_area_no_address"],
                       mask, detail)]


def rule_keyword_field_stuffed(df, col_map, cfg) -> list:
    if "keyword_field_stuffed" in cfg.get("disabled_rules", []) or not col_map.get("keywords"):
        return []
    raw     = _col(df, col_map, "keywords")
    has_kws = raw != ""
    ind_kws = cfg["high_risk_industries"]

    def _check(kw_str):
        kw_list = [k.strip() for k in re.split(r"[,;|]", kw_str) if k.strip()]
        if len(kw_list) < 4:
            return False, ""
        norm_kws    = [normalize_text(k) for k in kw_list]
        service_hits = [k for k in norm_kws if any(ind in k for ind in ind_kws)]
        all_words   = " ".join(norm_kws).split()
        freq        = Counter(w for w in all_words if len(w) > 3)
        repeated    = {w: c for w, c in freq.items() if c >= 3}
        if len(service_hits) >= 4 or repeated:
            parts = []
            if len(service_hits) >= 4:
                parts.append(f"{len(service_hits)} service-type keywords")
            if repeated:
                top = sorted(repeated.items(), key=lambda x: -x[1])[:3]
                parts.append(f"repeated terms: {dict(top)}")
            return True, f"Keywords field has {len(kw_list)} entries — {'; '.join(parts)}"
        return False, ""

    # apply only on rows that have keywords (avoids processing empty strings)
    fired  = pd.Series(False, index=df.index)
    detail = pd.Series("", index=df.index)
    if has_kws.any():
        results_arr = raw[has_kws].apply(_check)
        fired[has_kws]  = results_arr.apply(lambda t: t[0])
        detail[has_kws] = results_arr.apply(lambda t: t[1])

    return [RuleResult("KEYWORD_FIELD_STUFFED", cfg["weights"]["keyword_field_stuffed"],
                       fired, detail)]


def rule_landing_page_domain_mismatch(df, col_map, cfg) -> list:
    if ("landing_page_domain_mismatch" in cfg.get("disabled_rules", []) or
            not col_map.get("landing_page_url")):
        return []
    landing         = _col(df, col_map, "landing_page_url")
    website         = _col(df, col_map, "website")
    landing_domain  = landing.apply(extract_domain)
    website_domain  = website.apply(extract_domain)
    mask = (
        (landing != "") & (website != "") &
        (landing_domain != "") & (website_domain != "") &
        (landing_domain != website_domain)
    )
    detail = ("Landing page domain '" + landing_domain + "' differs from website domain '" +
              website_domain + "'").where(mask, "")
    return [RuleResult("LANDING_PAGE_DOMAIN_MISMATCH",
                       cfg["weights"]["landing_page_domain_mismatch"], mask, detail)]


def rule_residential_address(df, col_map, cfg, rdi_cache: dict) -> list:
    if "residential_address" in cfg.get("disabled_rules", []) or not rdi_cache:
        return []
    address = _col(df, col_map, "address")
    rdi     = address.map(rdi_cache).fillna("")
    mask    = rdi == "Residential"
    detail  = pd.Series("Address classified as Residential by SmartyStreets (RDI check)",
                        index=df.index).where(mask, "")
    return [RuleResult("RESIDENTIAL_ADDRESS", cfg["weights"]["residential_address"],
                       mask, detail)]


# ---------------------------------------------------------------------------
# Main processing — fully vectorised rule dispatch and aggregation
# ---------------------------------------------------------------------------

def process(df: pd.DataFrame, cfg: dict, verbose: bool = False) -> pd.DataFrame:
    col_map = map_columns(df)
    if verbose:
        print(f"[INFO] Mapped columns: { {k: v for k, v in col_map.items() if v} }")

    # --- Shared-count dicts (vectorised transforms) ---
    shared_address_counts = _count_col(
        df, col_map, "address",
        lambda s: s.str.lower().str.strip())
    shared_phone_counts = _count_col(
        df, col_map, "phone",
        lambda s: s.str.replace(r"\D", "", regex=True))
    shared_domain_counts = _count_col(
        df, col_map, "website",
        lambda s: s.apply(extract_domain))       # tldextract: apply unavoidable
    shared_email_counts = _count_col(
        df, col_map, "email",
        lambda s: s.str.lower().str.strip())

    # --- RDI cache (HTTP calls; iterates unique addresses only) ---
    rdi_cache = build_rdi_cache(df, col_map, cfg)

    # --- Normalised names for near-dup (single apply on the column) ---
    name_col   = col_map.get("business_name")
    raw_names  = (df[name_col].fillna("").astype(str) if name_col
                  else pd.Series("", index=df.index))
    norm_names = raw_names.apply(
        lambda x: strip_legal_suffixes(normalize_text(x))).tolist()

    # --- Run all rules ---
    all_results: list[RuleResult] = []
    all_results += rule_high_risk_industry(df, col_map, cfg)
    all_results += rule_generic_name(df, col_map, cfg)
    all_results += rule_po_box(df, col_map, cfg)
    all_results += rule_virtual_mailbox(df, col_map, cfg)
    all_results += rule_shared_address(df, col_map, cfg, shared_address_counts)
    all_results += rule_state_zip_mismatch(df, col_map, cfg)
    all_results += rule_invalid_phone(df, col_map, cfg)
    all_results += rule_tollfree_phone(df, col_map, cfg)
    all_results += rule_shared_phone(df, col_map, cfg, shared_phone_counts)
    all_results += rule_area_code_mismatch(df, col_map, cfg)
    all_results += rule_no_website(df, col_map, cfg)
    all_results += rule_website_quality(df, col_map, cfg, shared_domain_counts)
    all_results += rule_free_email(df, col_map, cfg)
    all_results += rule_autogenerated_email(df, col_map, cfg)
    all_results += rule_shared_email(df, col_map, cfg, shared_email_counts)
    all_results += rule_email_domain_mismatch(df, col_map, cfg)
    all_results += rule_exact_duplicate(df, col_map, cfg)
    all_results += rule_near_duplicate_name(df, col_map, cfg, norm_names)
    all_results += rule_same_owner(df, col_map, cfg)
    all_results += rule_batch_submission(df, col_map, cfg)
    all_results += rule_no_photos(df, col_map, cfg)
    all_results += rule_no_social(df, col_map, cfg)
    all_results += rule_year_established(df, col_map, cfg)
    all_results += rule_no_hours(df, col_map, cfg)
    all_results += rule_description_quality(df, col_map, cfg)
    all_results += rule_address_hidden(df, col_map, cfg)
    all_results += rule_service_area_no_address(df, col_map, cfg)
    all_results += rule_keyword_field_stuffed(df, col_map, cfg)
    all_results += rule_landing_page_domain_mismatch(df, col_map, cfg)
    all_results += rule_residential_address(df, col_map, cfg, rdi_cache)

    # --- Aggregate scores (vectorised sum) ---
    spam_score = pd.Series(0, index=df.index, dtype=int)
    for r in all_results:
        spam_score = spam_score + r.mask.astype(int) * r.weight

    # --- Build flags_triggered and flag_details via stack/groupby ---
    # Use enumerated keys to avoid collisions if two rules share a flag name.
    flag_wide   = pd.DataFrame(
        {f"{i}:{r.flag}": r.mask.map({True: r.flag, False: ""})
         for i, r in enumerate(all_results)},
        index=df.index,
    )
    detail_wide = pd.DataFrame(
        {f"{i}:{r.flag}": (r.flag + ": " + r.detail.astype(str)).where(r.mask, "")
         for i, r in enumerate(all_results)},
        index=df.index,
    )
    flags_triggered = _join_nonempty(flag_wide,   ", ")
    flag_details    = _join_nonempty(detail_wide, " | ")

    # --- is_duplicate_of_row (vectorised hash) ---
    row_hash  = pd.util.hash_pandas_object(df, index=False)
    first_idx = row_hash.groupby(row_hash).transform("idxmin")
    own_idx_s = pd.Series(df.index, index=df.index)
    is_dup_of_row = pd.Series("", index=df.index)
    dup_mask      = first_idx != own_idx_s
    is_dup_of_row[dup_mask] = first_idx[dup_mask].astype(str)

    risk_tier = _compute_risk_tier(spam_score, cfg["score_thresholds"])

    result_df = pd.DataFrame({
        "spam_score":         spam_score,
        "risk_tier":          risk_tier,
        "flags_triggered":    flags_triggered,
        "flag_details":       flag_details,
        "is_duplicate_of_row": is_dup_of_row,
    }, index=df.index)

    return df.join(result_df)


# ---------------------------------------------------------------------------
# Reporting, I/O, CLI
# ---------------------------------------------------------------------------

def print_summary(out_df: pd.DataFrame):
    total = len(out_df)
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"Total records processed: {total}")
    print()
    tier_counts = out_df["risk_tier"].value_counts()
    for tier in ["High Confidence Spam", "Likely Spam", "Review", "Clean"]:
        count = tier_counts.get(tier, 0)
        pct   = 100 * count / total if total else 0
        print(f"  {tier:25s}: {count:5d}  ({pct:.1f}%)")
    print()
    print("Top 5 most common flags:")
    all_flags = []
    for flags_str in out_df["flags_triggered"].dropna():
        all_flags += [f.strip() for f in flags_str.split(",") if f.strip()]
    for flag, count in Counter(all_flags).most_common(5):
        print(f"  {flag:40s}: {count}")
    print()
    print("Shared field clusters:")
    for label, key in [("Shared Phone", "SHARED_PHONE"),
                       ("Shared Address", "SHARED_ADDRESS"),
                       ("Shared Email", "SHARED_EMAIL")]:
        flagged = out_df[out_df["flags_triggered"].str.contains(key, na=False)]
        if not flagged.empty:
            print(f"  {label}: {len(flagged)} records flagged")
    reseller_col = next(
        (c for c in out_df.columns
         if c.lower() in ("reseller_id", "reseller_name", "reseller", "partner_id")),
        None,
    )
    if reseller_col:
        print()
        print("Reseller breakdown (avg spam score):")
        bd = (out_df.groupby(reseller_col)["spam_score"]
              .agg(["count", "mean"])
              .sort_values("mean", ascending=False)
              .head(10))
        for reseller, row in bd.iterrows():
            print(f"  {str(reseller):30s}: {int(row['count'])} records, avg {row['mean']:.1f}")
    print("=" * 60)


def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def save_output(out_df: pd.DataFrame, input_path: str):
    p       = Path(input_path)
    out_path = p.parent / (p.stem + "_flagged.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Flagged Results", index=False)
        total       = len(out_df)
        tier_counts = out_df["risk_tier"].value_counts()
        summary_rows = [
            {"Metric": f"Risk Tier: {t}",
             "Value":   tier_counts.get(t, 0),
             "Percent": f"{100 * tier_counts.get(t, 0) / total:.1f}%" if total else "0%"}
            for t in ["High Confidence Spam", "Likely Spam", "Review", "Clean"]
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
    print(f"\n[OUTPUT] Written to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Spam Business Detector — score and flag fake business listings")
    parser.add_argument("input", help="Path to input CSV or XLSX file")
    parser.add_argument("--config", default="spam_config.yaml",
                        help="Path to YAML config file (default: spam_config.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(find_config(args.config))
    df  = load_input(args.input)
    print(f"[INFO] {len(df)} records loaded, {len(df.columns)} columns")

    out_df = process(df, cfg, verbose=args.verbose)
    print_summary(out_df)
    save_output(out_df, args.input)


if __name__ == "__main__":
    main()
