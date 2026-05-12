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

import pandas as pd
import pgeocode
import phonenumbers
import requests
import tldextract
import yaml
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Column name aliases — map flexible input headers to canonical names.
# Aliases use the *normalized* form of the column name (see _normalize_col_name).
# Birdeye exports use "Field > SubField" naming; normalization strips the ">".
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    # Core fields
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

    # Phone — Birdeye: "Local Phone > Phone Number" / "Main Phone > Phone Number"
    "phone": [
        "phone", "phone_number", "telephone", "tel", "mobile", "cell",
        "local_phone_phone_number",   # "Local Phone > Phone Number"
        "main_phone_phone_number",    # "Main Phone > Phone Number"
        "local_phone",
        "main_phone",
    ],

    # Website — Birdeye: "Website URL > URL"
    "website": [
        "website", "url", "web", "site", "domain", "webpage",
        "website_url_url",   # "Website URL > URL"
        "website_url",
    ],

    # Industry / category
    "industry": [
        "industry", "category", "type", "business_type", "vertical", "service_type",
    ],

    # Email — Birdeye uses "Emails" (plural)
    "email": [
        "email", "email_address", "e_mail", "contact_email",
        "emails",
    ],

    # Owner / contact
    "owner_name": [
        "owner_name", "owner", "contact_name", "contact", "rep_name",
    ],

    # Dates / reseller
    "date_added": [
        "date_added", "created_at", "created_date", "added_date",
        "submission_date", "date",
    ],
    "reseller_id":   ["reseller_id", "reseller", "partner_id", "partner"],
    "reseller_name": ["reseller_name", "partner_name", "agent"],

    # --- Birdeye-specific new fields ---

    # "Photo Gallery > URL"
    "photo_gallery": ["photo_gallery_url", "photo_gallery", "photos", "gallery_url"],

    # Social handles
    "x_handle":       ["x_handle", "twitter_handle", "twitter", "x"],
    "facebook_url":   ["facebook_page_url", "facebook_url", "facebook", "fb"],
    "instagram_handle": ["instagram_handle", "instagram", "ig"],

    # "Year Established"
    "year_established": [
        "year_established", "established", "founded", "year_founded",
        "founding_year",
    ],

    # Hours — "Hours > Monday" … "Hours > Sunday"
    "hours_monday":    ["hours_monday",    "monday_hours",    "mon_hours"],
    "hours_tuesday":   ["hours_tuesday",   "tuesday_hours",   "tue_hours"],
    "hours_wednesday": ["hours_wednesday", "wednesday_hours", "wed_hours"],
    "hours_thursday":  ["hours_thursday",  "thursday_hours",  "thu_hours"],
    "hours_friday":    ["hours_friday",    "friday_hours",    "fri_hours"],
    "hours_saturday":  ["hours_saturday",  "saturday_hours",  "sat_hours"],
    "hours_sunday":    ["hours_sunday",    "sunday_hours",    "sun_hours"],

    # Description — Birdeye also exports a "descrp" column
    "description": [
        "description", "desc", "business_description", "about", "overview",
    ],
    "descrp": ["descrp"],

    # "Address Hidden"
    "address_hidden": ["address_hidden", "hide_address", "hidden_address"],

    # "Service Area Places > Name"
    "service_area": [
        "service_area_places_name",  # "Service Area Places > Name"
        "service_area_places",
        "service_area",
        "service_areas",
        "coverage_area",
    ],

    # "Keywords"
    "keywords": ["keywords", "tags", "keyword", "seo_keywords"],

    # "Landing Page URL"
    "landing_page_url": [
        "landing_page_url", "landing_page", "landing_url", "lp_url",
    ],
}

HOURS_DAYS = [
    "hours_monday", "hours_tuesday", "hours_wednesday", "hours_thursday",
    "hours_friday", "hours_saturday", "hours_sunday",
]

SOCIAL_KEYS = ["x_handle", "facebook_url", "instagram_handle"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_col_name(col: str) -> str:
    """Lowercase, collapse any run of non-alphanumeric chars to a single underscore."""
    s = col.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_legal_suffixes(name: str) -> str:
    suffixes = r"\b(llc|inc|co|corp|ltd|company|companies|group|services|solutions|associates)\b"
    return re.sub(suffixes, "", name).strip()


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


def map_columns(df: pd.DataFrame) -> dict:
    """Return mapping canonical_name -> actual_df_column (or None if absent)."""
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


def get(row, col_map, canonical, default=""):
    col = col_map.get(canonical)
    if col is None:
        return default
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return str(val).strip()


def _is_high_risk(row, col_map, cfg) -> bool:
    industry = normalize_text(get(row, col_map, "industry"))
    name = normalize_text(get(row, col_map, "business_name"))
    combined = f"{industry} {name}"
    return any(kw.lower() in combined for kw in cfg["high_risk_industries"])


# ---------------------------------------------------------------------------
# Rule functions — each returns list of (flag_name, detail, weight) tuples
# ---------------------------------------------------------------------------

def rule_high_risk_industry(row, col_map, cfg) -> list:
    if "high_risk_industry" in cfg.get("disabled_rules", []):
        return []
    industry = normalize_text(get(row, col_map, "industry"))
    name = normalize_text(get(row, col_map, "business_name"))
    combined = f"{industry} {name}"
    for kw in cfg["high_risk_industries"]:
        if kw.lower() in combined:
            return [("HIGH_RISK_INDUSTRY", f"Matched industry keyword: '{kw}'",
                     cfg["weights"]["high_risk_industry"])]
    return []


def rule_generic_name(row, col_map, cfg) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])
    name = get(row, col_map, "business_name")
    if not name:
        return []
    norm = normalize_text(name)

    if "generic_name" not in disabled:
        kw_hits = [kw for kw in cfg["spam_name_keywords"] if kw.lower() in norm]
        if kw_hits:
            flags.append(("GENERIC_NAME_KEYWORD",
                          f"Spam keywords in name: {kw_hits}",
                          cfg["weights"]["generic_name"]))

    if "keyword_stuffed_name" not in disabled:
        industry_hits = [kw for kw in cfg["high_risk_industries"] if kw.lower() in norm]
        if len(industry_hits) >= 2:
            flags.append(("KEYWORD_STUFFED_NAME",
                          f"Multiple industry keywords in name: {industry_hits}",
                          cfg["weights"]["keyword_stuffed_name"]))

    return flags


def rule_address_anomalies(row, col_map, cfg, shared_address_counts: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])
    address = get(row, col_map, "address")
    state = get(row, col_map, "state").upper()
    zip_code = get(row, col_map, "zip")

    if address:
        norm_addr = address.upper()

        if "po_box" not in disabled and re.search(r"\bP\.?\s*O\.?\s*BOX\b", norm_addr):
            flags.append(("PO_BOX", "Address is a PO Box", cfg["weights"]["po_box"]))

        if "virtual_mailbox" not in disabled:
            for virt in cfg["virtual_mailbox_keywords"]:
                if virt.upper() in norm_addr:
                    flags.append(("VIRTUAL_MAILBOX",
                                  f"Address matches virtual mailbox provider: '{virt}'",
                                  cfg["weights"]["virtual_mailbox"]))
                    break

        if "shared_address" not in disabled:
            count = shared_address_counts.get(normalize_text(address), 1)
            threshold = cfg["shared_address_threshold"]
            if count >= threshold:
                flags.append(("SHARED_ADDRESS",
                              f"Address shared by {count} businesses (threshold: {threshold})",
                              cfg["weights"]["shared_address"]))

    if "state_zip_mismatch" not in disabled and state and zip_code:
        try:
            result = pgeocode.Nominatim("us").query_postal_code(zip_code.split("-")[0])
            if result is not None and not pd.isna(result.state_code):
                if result.state_code.upper() != state:
                    flags.append(("STATE_ZIP_MISMATCH",
                                  f"ZIP {zip_code} belongs to {result.state_code.upper()}, not {state}",
                                  cfg["weights"]["state_zip_mismatch"]))
        except Exception:
            pass

    return flags


def rule_phone_anomalies(row, col_map, cfg, shared_phone_counts: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])
    raw_phone = get(row, col_map, "phone")
    digits = clean_phone(raw_phone)
    state = get(row, col_map, "state").upper()
    city = get(row, col_map, "city")

    if not raw_phone:
        return []

    if "invalid_phone" not in disabled:
        if len(digits) not in (10, 11):
            flags.append(("INVALID_PHONE",
                          f"Phone '{raw_phone}' has unexpected digit count ({len(digits)})",
                          cfg["weights"]["invalid_phone"]))
            return flags

    area_code = digits[-10:-7] if len(digits) >= 10 else ""

    if "voip_tollfree_phone" not in disabled and area_code in cfg["toll_free_prefixes"]:
        flags.append(("TOLLFREE_PHONE", f"Toll-free area code: {area_code}",
                      cfg["weights"]["voip_tollfree_phone"]))

    if "shared_phone" not in disabled:
        count = shared_phone_counts.get(digits, 1)
        threshold = cfg["shared_phone_threshold"]
        if count >= threshold:
            flags.append(("SHARED_PHONE",
                          f"Phone shared by {count} businesses (threshold: {threshold})",
                          cfg["weights"]["shared_phone"]))

    if "area_code_mismatch" not in disabled and area_code and state:
        try:
            parsed = phonenumbers.parse(f"+1{digits[-10:]}", "US")
            from phonenumbers import geocoder
            region = geocoder.description_for_number(parsed, "en")
            if region and state.upper() not in region.upper() and city.upper() not in region.upper():
                flags.append(("AREA_CODE_MISMATCH",
                              f"Area code {area_code} region '{region}' may not match state {state}",
                              cfg["weights"]["area_code_mismatch"]))
        except Exception:
            pass

    return flags


def rule_website_anomalies(row, col_map, cfg, shared_domain_counts: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])
    website = get(row, col_map, "website")
    name = normalize_text(get(row, col_map, "business_name"))

    # No website for high-risk industry (weight now 4 per Birdeye spec)
    if "no_website" not in disabled and not website and _is_high_risk(row, col_map, cfg):
        flags.append(("NO_WEBSITE",
                      "No website provided for a high-risk industry business",
                      cfg["weights"]["no_website"]))

    if not website:
        return flags

    domain = extract_domain(website)
    full_domain = extract_full_domain(website)

    if "generic_builder_site" not in disabled:
        for builder in cfg["generic_website_builders"]:
            if builder in full_domain:
                flags.append(("GENERIC_BUILDER_SITE",
                              f"Website is on a generic builder: {full_domain}",
                              cfg["weights"]["generic_builder_site"]))
                break

    for lead_domain in cfg["lead_gen_domains"]:
        if lead_domain in full_domain:
            flags.append(("LEAD_GEN_DOMAIN",
                          f"Website is a lead-gen directory: {full_domain}",
                          cfg["weights"]["shared_domain"]))
            break

    if "shared_domain" not in disabled and domain:
        count = shared_domain_counts.get(domain, 1)
        if count >= 2:
            flags.append(("SHARED_DOMAIN",
                          f"Website domain '{domain}' shared by {count} businesses",
                          cfg["weights"]["shared_domain"]))

    if domain and name:
        name_tokens = {t for t in name.split() if len(t) > 3}
        if name_tokens and not any(t in domain for t in name_tokens):
            flags.append(("DOMAIN_NAME_MISMATCH",
                          f"Domain '{domain}' shares no words with business name",
                          cfg["weights"]["new_domain"]))

    return flags


def rule_email_anomalies(row, col_map, cfg, shared_email_counts: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])
    email = get(row, col_map, "email").lower()
    website = get(row, col_map, "website")

    if not email:
        return []

    if "free_email" not in disabled:
        email_domain = email.split("@")[-1] if "@" in email else ""
        if email_domain in cfg["free_email_domains"]:
            flags.append(("FREE_EMAIL_PROVIDER",
                          f"Email uses free provider: {email_domain}",
                          cfg["weights"]["free_email"]))

        if "autogenerated_email" not in disabled:
            local = email.split("@")[0] if "@" in email else email
            digit_ratio = sum(c.isdigit() for c in local) / max(len(local), 1)
            if digit_ratio > 0.5 or (len(local) > 8 and re.search(r"\d{4,}", local)):
                flags.append(("AUTOGENERATED_EMAIL",
                              f"Email local part appears auto-generated: {local}",
                              cfg["weights"]["autogenerated_email"]))

    if "shared_email" not in disabled:
        count = shared_email_counts.get(email, 1)
        if count >= 2:
            flags.append(("SHARED_EMAIL",
                          f"Email shared by {count} businesses",
                          cfg["weights"]["shared_email"]))

    if "email_domain_mismatch" not in disabled and website:
        web_domain = extract_domain(website)
        email_domain = email.split("@")[-1] if "@" in email else ""
        if web_domain and email_domain and email_domain not in cfg["free_email_domains"]:
            if email_domain != web_domain:
                flags.append(("EMAIL_DOMAIN_MISMATCH",
                              f"Email domain '{email_domain}' differs from website '{web_domain}'",
                              cfg["weights"]["email_domain_mismatch"]))

    return flags


def rule_duplicate_detection(row, col_map, cfg, idx, norm_names: list,
                             exact_hash_map: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])

    if "exact_duplicate_row" not in disabled:
        row_key = tuple(str(v) for v in row.values)
        other = [i for i in exact_hash_map.get(row_key, []) if i != idx]
        if other:
            flags.append(("EXACT_DUPLICATE",
                          f"Exact duplicate of row(s): {other}",
                          cfg["weights"]["exact_duplicate_row"]))

    if "near_duplicate_name" not in disabled:
        my_name = norm_names[idx]
        if my_name:
            for other_idx, other_name in enumerate(norm_names):
                if other_idx == idx or not other_name:
                    continue
                score = fuzz.ratio(my_name, other_name)
                if score >= 90:
                    flags.append(("NEAR_DUPLICATE_NAME",
                                  f"Name is ~{score}% similar to row {other_idx}: '{other_name}'",
                                  cfg["weights"]["near_duplicate_name"]))
                    break

    return flags


def rule_same_owner(row, col_map, cfg, owner_industry_map: dict) -> list:
    if "same_owner_multi_biz" in cfg.get("disabled_rules", []):
        return []
    owner = normalize_text(get(row, col_map, "owner_name"))
    industry = normalize_text(get(row, col_map, "industry"))
    if not owner:
        return []
    entries = owner_industry_map.get(owner, [])
    if len(entries) >= 2 and industry:
        return [("SAME_OWNER_MULTI_BIZ",
                 f"Owner '{owner}' appears across {len(entries)} businesses",
                 cfg["weights"]["same_owner_multi_biz"])]
    return []


def rule_batch_submission(row, col_map, cfg, batch_counts: dict) -> list:
    if "batch_submission" in cfg.get("disabled_rules", []):
        return []
    date_raw = get(row, col_map, "date_added")
    reseller = get(row, col_map, "reseller_id") or get(row, col_map, "reseller_name") or "UNKNOWN"
    if not date_raw:
        return []
    count = batch_counts.get(f"{reseller}::{date_raw[:10]}", 0)
    threshold = cfg["batch_submission_threshold"]
    if count >= threshold:
        return [("BATCH_SUBMISSION",
                 f"Reseller '{reseller}' submitted {count} listings on {date_raw[:10]} (threshold: {threshold})",
                 cfg["weights"]["batch_submission"])]
    return []


# ---------------------------------------------------------------------------
# New Birdeye-specific rules
# ---------------------------------------------------------------------------

def rule_no_photos(row, col_map, cfg) -> list:
    if "no_photos" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("photo_gallery"):
        return []
    if not get(row, col_map, "photo_gallery"):
        return [("NO_PHOTOS", "Photo Gallery URL is blank",
                 cfg["weights"]["no_photos"])]
    return []


def rule_no_social(row, col_map, cfg) -> list:
    if "no_social" in cfg.get("disabled_rules", []):
        return []
    if not any(col_map.get(k) for k in SOCIAL_KEYS):
        return []
    if not any(get(row, col_map, k) for k in SOCIAL_KEYS):
        return [("NO_SOCIAL_PRESENCE", "All social media fields (X, Facebook, Instagram) are blank",
                 cfg["weights"]["no_social"])]
    return []


def rule_year_established(row, col_map, cfg) -> list:
    if "year_established" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("year_established"):
        return []

    year_raw = get(row, col_map, "year_established")
    if not year_raw:
        return [("YEAR_ESTABLISHED_MISSING", "Year Established is blank",
                 cfg["weights"]["year_established"])]

    m = re.search(r"\d{4}", year_raw)
    if not m:
        return [("YEAR_ESTABLISHED_INVALID",
                 f"Year Established could not be parsed: '{year_raw}'",
                 cfg["weights"]["year_established"])]

    year = int(m.group())
    today = date.today()
    # Flag if established within approximately the last 12 months
    # (year-only field: flag current year or previous year)
    months_since = (today.year - year) * 12 + today.month
    if months_since <= 12:
        return [("RECENTLY_ESTABLISHED",
                 f"Business established within last 12 months (year: {year})",
                 cfg["weights"]["year_established"])]
    return []


def rule_no_hours(row, col_map, cfg) -> list:
    if "no_hours" in cfg.get("disabled_rules", []):
        return []
    if not any(col_map.get(d) for d in HOURS_DAYS):
        return []
    if not any(get(row, col_map, d) for d in HOURS_DAYS):
        return [("NO_HOURS_PROVIDED", "All business hours fields are blank",
                 cfg["weights"]["no_hours"])]
    return []


def rule_description_quality(row, col_map, cfg) -> list:
    if "description_quality" in cfg.get("disabled_rules", []):
        return []

    has_desc_col = col_map.get("description") or col_map.get("descrp")
    if not has_desc_col:
        return []

    desc = get(row, col_map, "description") or get(row, col_map, "descrp")

    if not desc:
        return [("DESCRIPTION_MISSING", "Description is blank",
                 cfg["weights"]["description_missing"])]

    if len(desc.strip()) < 20:
        return [("DESCRIPTION_TOO_SHORT",
                 f"Description is under 20 characters ({len(desc.strip())} chars)",
                 cfg["weights"]["description_missing"])]

    # Keyword-stuffed: 3+ repeated city or service terms
    norm_desc = normalize_text(desc)
    city = normalize_text(get(row, col_map, "city"))

    hits = []
    if city and norm_desc.count(city) >= 3:
        hits.append(f"city '{city}' ×{norm_desc.count(city)}")

    service_repeats = [
        kw for kw in cfg["high_risk_industries"]
        if len(kw) > 3 and norm_desc.count(kw.lower()) >= 2
    ]
    hits += service_repeats

    if len(hits) >= 3:
        return [("DESCRIPTION_KEYWORD_STUFFED",
                 f"Description contains repeated city/service keywords: {hits[:5]}",
                 cfg["weights"]["description_stuffed"])]

    return []


def rule_address_hidden(row, col_map, cfg) -> list:
    if "address_hidden" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("address_hidden"):
        return []

    hidden = get(row, col_map, "address_hidden").lower().strip()
    if hidden not in ("true", "yes", "1", "y"):
        return []

    if _is_high_risk(row, col_map, cfg):
        return [("ADDRESS_HIDDEN_HIGH_RISK",
                 "Address is marked hidden for a high-risk industry business",
                 cfg["weights"]["address_hidden"])]
    return []


def rule_service_area_no_address(row, col_map, cfg) -> list:
    if "service_area_no_address" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("service_area"):
        return []

    service_area = get(row, col_map, "service_area")
    address = get(row, col_map, "address")

    if service_area and not address:
        return [("SERVICE_AREA_NO_ADDRESS",
                 f"Service area '{service_area[:60]}' set but no street address provided",
                 cfg["weights"]["service_area_no_address"])]
    return []


def rule_keyword_field_stuffed(row, col_map, cfg) -> list:
    if "keyword_field_stuffed" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("keywords"):
        return []

    raw = get(row, col_map, "keywords")
    if not raw:
        return []

    kw_list = [k.strip() for k in re.split(r"[,;|]", raw) if k.strip()]
    if len(kw_list) < 4:
        return []

    norm_kws = [normalize_text(k) for k in kw_list]

    # Count keywords that match high-risk service terms
    service_hits = [k for k in norm_kws
                    if any(ind in k for ind in cfg["high_risk_industries"])]

    # Detect repeated geo/location words (≥ 3 occurrences of same word across keywords)
    all_words = " ".join(norm_kws).split()
    word_freq = Counter(w for w in all_words if len(w) > 3)
    repeated = {w: c for w, c in word_freq.items() if c >= 3}

    if len(service_hits) >= 4 or repeated:
        details = []
        if len(service_hits) >= 4:
            details.append(f"{len(service_hits)} service-type keywords")
        if repeated:
            top = sorted(repeated.items(), key=lambda x: -x[1])[:3]
            details.append(f"repeated terms: {dict(top)}")
        return [("KEYWORD_FIELD_STUFFED",
                 f"Keywords field has {len(kw_list)} entries — {'; '.join(details)}",
                 cfg["weights"]["keyword_field_stuffed"])]

    return []


def rule_residential_address(row, col_map, cfg, rdi_cache: dict) -> list:
    if "residential_address" in cfg.get("disabled_rules", []):
        return []
    if not rdi_cache:
        return []
    address = get(row, col_map, "address")
    if not address:
        return []
    rdi = rdi_cache.get(address)
    if rdi == "Residential":
        return [("RESIDENTIAL_ADDRESS",
                 "Address classified as Residential by SmartyStreets (RDI check)",
                 cfg["weights"]["residential_address"])]
    return []


def rule_landing_page_domain_mismatch(row, col_map, cfg) -> list:
    if "landing_page_domain_mismatch" in cfg.get("disabled_rules", []):
        return []
    if not col_map.get("landing_page_url"):
        return []

    landing = get(row, col_map, "landing_page_url")
    website = get(row, col_map, "website")

    if not landing or not website:
        return []

    landing_domain = extract_domain(landing)
    website_domain = extract_domain(website)

    if landing_domain and website_domain and landing_domain != website_domain:
        return [("LANDING_PAGE_DOMAIN_MISMATCH",
                 f"Landing page domain '{landing_domain}' differs from website domain '{website_domain}'",
                 cfg["weights"]["landing_page_domain_mismatch"])]
    return []


def _fetch_rdi(street: str, city: str, state: str, zipcode: str,
               auth_id: str, auth_token: str) -> str:
    """Call SmartyStreets US Street Address API; return rdi value or '' on no match/error."""
    params = {
        "auth-id": auth_id,
        "auth-token": auth_token,
        "street": street,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "candidates": 1,
    }
    resp = requests.get(
        "https://us-street.api.smartystreets.com/street-address",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data and isinstance(data, list):
        return data[0].get("metadata", {}).get("rdi", "")
    return ""


def build_rdi_cache(df: pd.DataFrame, col_map: dict, cfg: dict) -> dict:
    """
    Pre-fetch RDI for every unique address belonging to a high-risk-industry row.
    Returns an empty dict (silently skipping the rule) if credentials are absent.
    """
    auth_id = str(cfg.get("smartystreets_auth_id", "")).strip()
    auth_token = str(cfg.get("smartystreets_auth_token", "")).strip()

    if not auth_id or not auth_token:
        print("[WARN] SmartyStreets credentials not configured — residential address check skipped.")
        return {}

    cache = {}
    for _, row in df.iterrows():
        if not _is_high_risk(row, col_map, cfg):
            continue
        address = get(row, col_map, "address")
        if not address or address in cache:
            continue
        try:
            rdi = _fetch_rdi(
                address,
                get(row, col_map, "city"),
                get(row, col_map, "state"),
                get(row, col_map, "zip"),
                auth_id,
                auth_token,
            )
            cache[address] = rdi
        except Exception as exc:
            print(f"[WARN] SmartyStreets API error for '{address}': {exc}")
            cache[address] = ""

    return cache


def compute_risk_tier(score: int, thresholds: dict) -> str:
    if score >= thresholds["high_confidence_spam"]:
        return "High Confidence Spam"
    if score >= thresholds["likely_spam"]:
        return "Likely Spam"
    if score >= thresholds["review"]:
        return "Review"
    return "Clean"


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def build_shared_counts(df: pd.DataFrame, col_map: dict, canonical: str,
                        transform=None) -> dict:
    col = col_map.get(canonical)
    if col is None:
        return {}
    series = df[col].fillna("").astype(str)
    if transform:
        series = series.apply(transform)
    return series[series != ""].value_counts().to_dict()


def build_exact_hash_map(df: pd.DataFrame) -> dict:
    hmap = defaultdict(list)
    for idx, row in df.iterrows():
        hmap[tuple(str(v) for v in row.values)].append(idx)
    return hmap


def build_batch_counts(df: pd.DataFrame, col_map: dict) -> dict:
    date_col = col_map.get("date_added")
    reseller_col = col_map.get("reseller_id") or col_map.get("reseller_name")
    counts = defaultdict(int)
    if not date_col:
        return counts
    for _, row in df.iterrows():
        date_raw = str(row.get(date_col, ""))[:10]
        reseller = str(row.get(reseller_col, "UNKNOWN")) if reseller_col else "UNKNOWN"
        if date_raw and date_raw != "nan":
            counts[f"{reseller}::{date_raw}"] += 1
    return counts


def build_owner_industry_map(df: pd.DataFrame, col_map: dict) -> dict:
    owner_col = col_map.get("owner_name")
    industry_col = col_map.get("industry")
    result = defaultdict(list)
    if not owner_col:
        return result
    for _, row in df.iterrows():
        owner = normalize_text(str(row.get(owner_col, "")))
        industry = normalize_text(str(row.get(industry_col, ""))) if industry_col else ""
        if owner:
            result[owner].append(industry)
    return result


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(df: pd.DataFrame, cfg: dict, verbose: bool = False) -> pd.DataFrame:
    col_map = map_columns(df)

    if verbose:
        mapped = {k: v for k, v in col_map.items() if v}
        print(f"[INFO] Mapped columns: {mapped}")

    shared_address_counts = build_shared_counts(df, col_map, "address", normalize_text)
    shared_phone_counts   = build_shared_counts(df, col_map, "phone", clean_phone)
    shared_domain_counts  = build_shared_counts(df, col_map, "website", extract_domain)
    shared_email_counts   = build_shared_counts(
        df, col_map, "email", lambda x: x.lower().strip())
    exact_hash_map     = build_exact_hash_map(df)
    batch_counts       = build_batch_counts(df, col_map)
    owner_industry_map = build_owner_industry_map(df, col_map)
    rdi_cache          = build_rdi_cache(df, col_map, cfg)

    name_col = col_map.get("business_name")
    norm_names = []
    for _, row in df.iterrows():
        raw = str(row.get(name_col, "")) if name_col else ""
        norm_names.append(strip_legal_suffixes(normalize_text(raw)))

    results = []
    thresholds = cfg["score_thresholds"]

    for idx, row in df.iterrows():
        if verbose:
            print(f"[INFO] Processing row {idx}: {get(row, col_map, 'business_name')}")

        all_flags = []
        # --- existing rules ---
        all_flags += rule_high_risk_industry(row, col_map, cfg)
        all_flags += rule_generic_name(row, col_map, cfg)
        all_flags += rule_address_anomalies(row, col_map, cfg, shared_address_counts)
        all_flags += rule_phone_anomalies(row, col_map, cfg, shared_phone_counts)
        all_flags += rule_website_anomalies(row, col_map, cfg, shared_domain_counts)
        all_flags += rule_email_anomalies(row, col_map, cfg, shared_email_counts)
        all_flags += rule_duplicate_detection(row, col_map, cfg, idx, norm_names, exact_hash_map)
        all_flags += rule_same_owner(row, col_map, cfg, owner_industry_map)
        all_flags += rule_batch_submission(row, col_map, cfg, batch_counts)
        # --- new Birdeye-specific rules ---
        all_flags += rule_no_photos(row, col_map, cfg)
        all_flags += rule_no_social(row, col_map, cfg)
        all_flags += rule_year_established(row, col_map, cfg)
        all_flags += rule_no_hours(row, col_map, cfg)
        all_flags += rule_description_quality(row, col_map, cfg)
        all_flags += rule_address_hidden(row, col_map, cfg)
        all_flags += rule_service_area_no_address(row, col_map, cfg)
        all_flags += rule_residential_address(row, col_map, cfg, rdi_cache)
        all_flags += rule_keyword_field_stuffed(row, col_map, cfg)
        all_flags += rule_landing_page_domain_mismatch(row, col_map, cfg)

        spam_score = sum(w for _, _, w in all_flags)
        risk_tier = compute_risk_tier(spam_score, thresholds)

        row_key = tuple(str(v) for v in row.values)
        dup_rows = [i for i in exact_hash_map.get(row_key, []) if i != idx]

        results.append({
            "_orig_idx": idx,
            "spam_score": spam_score,
            "risk_tier": risk_tier,
            "flags_triggered": ", ".join(f for f, _, _ in all_flags),
            "flag_details": " | ".join(f"{f}: {d}" for f, d, _ in all_flags),
            "is_duplicate_of_row": str(dup_rows[0]) if dup_rows else "",
        })

    result_df = pd.DataFrame(results).set_index("_orig_idx")
    return df.join(result_df)


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
        pct = 100 * count / total if total else 0
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
    for label, key in [("Shared Phone", "SHARED_PHONE"), ("Shared Address", "SHARED_ADDRESS"),
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
        breakdown = out_df.groupby(reseller_col)["spam_score"].agg(["count", "mean"])
        for reseller, row in breakdown.sort_values("mean", ascending=False).head(10).iterrows():
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
    p = Path(input_path)
    out_path = p.parent / (p.stem + "_flagged.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Flagged Results", index=False)
        total = len(out_df)
        tier_counts = out_df["risk_tier"].value_counts()
        summary_rows = [
            {
                "Metric": f"Risk Tier: {t}",
                "Value": tier_counts.get(t, 0),
                "Percent": f"{100 * tier_counts.get(t, 0) / total:.1f}%" if total else "0%",
            }
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

    for p, label in [(args.input, "Input"), (args.config, "Config")]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(args.config)
    df = load_input(args.input)
    print(f"[INFO] {len(df)} records loaded, {len(df.columns)} columns")

    out_df = process(df, cfg, verbose=args.verbose)
    print_summary(out_df)
    save_output(out_df, args.input)


if __name__ == "__main__":
    main()
