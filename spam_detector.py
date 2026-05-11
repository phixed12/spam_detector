#!/usr/bin/env python3
"""
Spam Business Detector
Scores and flags potential spam/fake business listings from CSV or XLSX input.
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pgeocode
import phonenumbers
import tldextract
import yaml
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Column name aliases — map flexible input headers to canonical names
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "business_name": ["business_name", "name", "company", "company_name", "biz_name", "business"],
    "address": ["address", "street", "street_address", "addr", "address1"],
    "city": ["city", "town", "municipality"],
    "state": ["state", "st", "province", "region"],
    "zip": ["zip", "zip_code", "zipcode", "postal_code", "postal"],
    "phone": ["phone", "phone_number", "telephone", "tel", "mobile", "cell"],
    "website": ["website", "url", "web", "site", "domain", "webpage"],
    "industry": ["industry", "category", "type", "business_type", "vertical", "service_type"],
    "email": ["email", "email_address", "e_mail", "contact_email"],
    "owner_name": ["owner_name", "owner", "contact_name", "contact", "rep_name"],
    "date_added": ["date_added", "created_at", "created_date", "added_date", "submission_date", "date"],
    "reseller_id": ["reseller_id", "reseller", "partner_id", "partner"],
    "reseller_name": ["reseller_name", "partner_name", "agent"],
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def extract_full_domain(url) -> str:
    """Return full subdomain+domain+suffix for generic-builder detection."""
    if not isinstance(url, str) or not url.strip():
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    ext = tldextract.extract(url)
    parts = [p for p in [ext.subdomain, ext.domain, ext.suffix] if p]
    return ".".join(parts)


def map_columns(df: pd.DataFrame) -> dict:
    """Return mapping canonical_name -> actual_df_column (or None if missing)."""
    actual = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
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
        industry_words = cfg["high_risk_industries"]
        industry_hits = [kw for kw in industry_words if kw.lower() in norm]
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

        # PO Box
        if "po_box" not in disabled and re.search(r"\bP\.?\s*O\.?\s*BOX\b", norm_addr):
            flags.append(("PO_BOX", "Address is a PO Box",
                          cfg["weights"]["po_box"]))

        # Virtual mailbox providers
        if "virtual_mailbox" not in disabled:
            for virt in cfg["virtual_mailbox_keywords"]:
                if virt.upper() in norm_addr:
                    flags.append(("VIRTUAL_MAILBOX",
                                  f"Address matches virtual mailbox provider: '{virt}'",
                                  cfg["weights"]["virtual_mailbox"]))
                    break

        # Shared address
        if "shared_address" not in disabled:
            norm_key = normalize_text(address)
            count = shared_address_counts.get(norm_key, 1)
            threshold = cfg["shared_address_threshold"]
            if count >= threshold:
                flags.append(("SHARED_ADDRESS",
                              f"Address shared by {count} businesses (threshold: {threshold})",
                              cfg["weights"]["shared_address"]))

    # State/zip mismatch
    if "state_zip_mismatch" not in disabled and state and zip_code:
        try:
            nomi = pgeocode.Nominatim("us")
            result = nomi.query_postal_code(zip_code.split("-")[0])
            if result is not None and not pd.isna(result.state_code):
                expected_state = result.state_code.upper()
                if expected_state != state.upper():
                    flags.append(("STATE_ZIP_MISMATCH",
                                  f"ZIP {zip_code} belongs to {expected_state}, not {state}",
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

    # Invalid phone
    if "invalid_phone" not in disabled:
        if len(digits) not in (10, 11):
            flags.append(("INVALID_PHONE",
                          f"Phone '{raw_phone}' has unexpected digit count ({len(digits)})",
                          cfg["weights"]["invalid_phone"]))
            return flags  # Further checks unreliable

    area_code = digits[-10:-7] if len(digits) >= 10 else ""

    # Toll-free
    if "voip_tollfree_phone" not in disabled:
        if area_code in cfg["toll_free_prefixes"]:
            flags.append(("TOLLFREE_PHONE",
                          f"Toll-free area code: {area_code}",
                          cfg["weights"]["voip_tollfree_phone"]))

    # Shared phone
    if "shared_phone" not in disabled:
        count = shared_phone_counts.get(digits, 1)
        threshold = cfg["shared_phone_threshold"]
        if count >= threshold:
            flags.append(("SHARED_PHONE",
                          f"Phone shared by {count} businesses (threshold: {threshold})",
                          cfg["weights"]["shared_phone"]))

    # Area code vs state mismatch
    if "area_code_mismatch" not in disabled and area_code and state:
        try:
            parsed = phonenumbers.parse(f"+1{digits[-10:]}", "US")
            from phonenumbers import geocoder
            region = geocoder.description_for_number(parsed, "en")
            if region and state and state.upper() not in region.upper() and city.upper() not in region.upper():
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
    industry = get(row, col_map, "industry")
    name = normalize_text(get(row, col_map, "business_name"))

    # No website for service business
    if "no_website" not in disabled:
        if not website and industry:
            norm_ind = normalize_text(industry)
            if any(kw in norm_ind for kw in cfg["high_risk_industries"]):
                flags.append(("NO_WEBSITE",
                              "No website provided for a service-type business",
                              cfg["weights"]["no_website"]))

    if not website:
        return flags

    domain = extract_domain(website)
    full_domain = extract_full_domain(website)

    # Generic website builder
    if "generic_builder_site" not in disabled:
        for builder in cfg["generic_website_builders"]:
            if builder in full_domain:
                flags.append(("GENERIC_BUILDER_SITE",
                              f"Website is on a generic builder: {full_domain}",
                              cfg["weights"]["generic_builder_site"]))
                break

    # Lead-gen aggregator
    for lead_domain in cfg["lead_gen_domains"]:
        if lead_domain in full_domain:
            flags.append(("LEAD_GEN_DOMAIN",
                          f"Website is a lead-gen directory: {full_domain}",
                          cfg["weights"]["shared_domain"]))
            break

    # Shared domain
    if "shared_domain" not in disabled and domain:
        count = shared_domain_counts.get(domain, 1)
        if count >= 2:
            flags.append(("SHARED_DOMAIN",
                          f"Website domain '{domain}' shared by {count} businesses",
                          cfg["weights"]["shared_domain"]))

    # Domain doesn't reference business name at all
    if domain and name:
        domain_words = set(re.split(r"[\.\-_]", domain))
        name_tokens = set(name.split())
        # Remove very short tokens
        name_tokens = {t for t in name_tokens if len(t) > 3}
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

    # Free provider
    if "free_email" not in disabled:
        email_domain = email.split("@")[-1] if "@" in email else ""
        if email_domain in cfg["free_email_domains"]:
            flags.append(("FREE_EMAIL_PROVIDER",
                          f"Email uses free provider: {email_domain}",
                          cfg["weights"]["free_email"]))

        # Auto-generated pattern: lots of digits or random chars
        if "autogenerated_email" not in disabled:
            local = email.split("@")[0] if "@" in email else email
            digit_ratio = sum(c.isdigit() for c in local) / max(len(local), 1)
            if digit_ratio > 0.5 or (len(local) > 8 and re.search(r"\d{4,}", local)):
                flags.append(("AUTOGENERATED_EMAIL",
                              f"Email local part appears auto-generated: {local}",
                              cfg["weights"]["autogenerated_email"]))

    # Shared email
    if "shared_email" not in disabled:
        count = shared_email_counts.get(email, 1)
        if count >= 2:
            flags.append(("SHARED_EMAIL",
                          f"Email shared by {count} businesses",
                          cfg["weights"]["shared_email"]))

    # Email domain vs website domain mismatch
    if "email_domain_mismatch" not in disabled and website:
        web_domain = extract_domain(website)
        email_domain = email.split("@")[-1] if "@" in email else ""
        if web_domain and email_domain and email_domain not in cfg["free_email_domains"]:
            if email_domain != web_domain:
                flags.append(("EMAIL_DOMAIN_MISMATCH",
                              f"Email domain '{email_domain}' differs from website '{web_domain}'",
                              cfg["weights"]["email_domain_mismatch"]))

    return flags


def rule_duplicate_detection(row, col_map, cfg, df, idx,
                             norm_names: list, exact_hash_map: dict) -> list:
    flags = []
    disabled = cfg.get("disabled_rules", [])

    # Exact duplicate row
    if "exact_duplicate_row" not in disabled:
        row_key = tuple(str(v) for v in row.values)
        if exact_hash_map.get(row_key, [None])[0] != idx and idx in sum(
            [v for k, v in exact_hash_map.items() if idx in v], []
        ):
            other = [i for i in exact_hash_map[row_key] if i != idx]
            if other:
                flags.append(("EXACT_DUPLICATE",
                              f"Exact duplicate of row(s): {other}",
                              cfg["weights"]["exact_duplicate_row"]))

    # Near-duplicate name
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
                    break  # one hit enough

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
        industries = set(e for e in entries if e)
        if len(industries) >= 1:
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
    key = f"{reseller}::{date_raw[:10]}"  # group by day
    count = batch_counts.get(key, 0)
    threshold = cfg["batch_submission_threshold"]
    if count >= threshold:
        return [("BATCH_SUBMISSION",
                 f"Reseller '{reseller}' submitted {count} listings on {date_raw[:10]} (threshold: {threshold})",
                 cfg["weights"]["batch_submission"])]
    return []


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
    """Map row-tuple -> list of indices with that exact tuple."""
    hmap = defaultdict(list)
    for idx, row in df.iterrows():
        key = tuple(str(v) for v in row.values)
        hmap[key].append(idx)
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
        print(f"[INFO] Mapped columns: { {k: v for k, v in col_map.items() if v} }")

    # Pre-compute shared counts
    shared_address_counts = build_shared_counts(
        df, col_map, "address", transform=normalize_text)
    shared_phone_counts = build_shared_counts(
        df, col_map, "phone", transform=clean_phone)
    shared_domain_counts = build_shared_counts(
        df, col_map, "website", transform=extract_domain)
    shared_email_counts = build_shared_counts(
        df, col_map, "email", transform=lambda x: x.lower().strip())
    exact_hash_map = build_exact_hash_map(df)
    batch_counts = build_batch_counts(df, col_map)
    owner_industry_map = build_owner_industry_map(df, col_map)

    # Normalized names for near-duplicate detection
    name_col = col_map.get("business_name")
    norm_names = []
    if name_col:
        for _, row in df.iterrows():
            raw = str(row.get(name_col, ""))
            n = strip_legal_suffixes(normalize_text(raw))
            norm_names.append(n)
    else:
        norm_names = [""] * len(df)

    results = []
    thresholds = cfg["score_thresholds"]

    for idx, row in df.iterrows():
        if verbose:
            bname = get(row, col_map, "business_name")
            print(f"[INFO] Processing row {idx}: {bname}")

        all_flags = []
        all_flags += rule_high_risk_industry(row, col_map, cfg)
        all_flags += rule_generic_name(row, col_map, cfg)
        all_flags += rule_address_anomalies(row, col_map, cfg, shared_address_counts)
        all_flags += rule_phone_anomalies(row, col_map, cfg, shared_phone_counts)
        all_flags += rule_website_anomalies(row, col_map, cfg, shared_domain_counts)
        all_flags += rule_email_anomalies(row, col_map, cfg, shared_email_counts)
        all_flags += rule_duplicate_detection(row, col_map, cfg, df, idx,
                                              norm_names, exact_hash_map)
        all_flags += rule_same_owner(row, col_map, cfg, owner_industry_map)
        all_flags += rule_batch_submission(row, col_map, cfg, batch_counts)

        spam_score = sum(w for _, _, w in all_flags)
        risk_tier = compute_risk_tier(spam_score, thresholds)
        flags_triggered = ", ".join(f for f, _, _ in all_flags)
        flag_details = " | ".join(f"{f}: {d}" for f, d, _ in all_flags)

        # Find duplicate row (first exact dup other than self)
        row_key = tuple(str(v) for v in row.values)
        dup_rows = [i for i in exact_hash_map.get(row_key, []) if i != idx]
        is_dup_of = str(dup_rows[0]) if dup_rows else ""

        results.append({
            "_orig_idx": idx,
            "spam_score": spam_score,
            "risk_tier": risk_tier,
            "flags_triggered": flags_triggered,
            "flag_details": flag_details,
            "is_duplicate_of_row": is_dup_of,
        })

    result_df = pd.DataFrame(results).set_index("_orig_idx")
    out_df = df.join(result_df)
    return out_df


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
    from collections import Counter
    flag_counts = Counter(all_flags)
    for flag, count in flag_counts.most_common(5):
        print(f"  {flag:35s}: {count}")

    # Cluster detection summary
    print()
    print("Shared field clusters:")
    for col_label, col in [("Shared Phone", "shared_phone"), ("Shared Address", "shared_address"),
                            ("Shared Email", "shared_email")]:
        flagged = out_df[out_df["flags_triggered"].str.contains(col.upper(), na=False)]
        if not flagged.empty:
            print(f"  {col_label}: {len(flagged)} records flagged")

    # Reseller breakdown
    reseller_col = None
    for c in out_df.columns:
        if c.lower() in ("reseller_id", "reseller_name", "reseller", "partner_id"):
            reseller_col = c
            break
    if reseller_col:
        print()
        print("Reseller breakdown (spam score avg):")
        breakdown = out_df.groupby(reseller_col)["spam_score"].agg(["count", "mean"])
        for reseller, row in breakdown.sort_values("mean", ascending=False).head(10).iterrows():
            print(f"  {str(reseller):30s}: {int(row['count'])} records, avg score {row['mean']:.1f}")

    print("=" * 60)


def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def save_output(out_df: pd.DataFrame, input_path: str):
    p = Path(input_path)
    out_path = p.parent / (p.stem + "_flagged.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Flagged Results", index=False)

        # Summary sheet
        summary_rows = []
        total = len(out_df)
        tier_counts = out_df["risk_tier"].value_counts()
        for tier in ["High Confidence Spam", "Likely Spam", "Review", "Clean"]:
            count = tier_counts.get(tier, 0)
            summary_rows.append({
                "Metric": f"Risk Tier: {tier}",
                "Value": count,
                "Percent": f"{100*count/total:.1f}%" if total else "0%",
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n[OUTPUT] Written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spam Business Detector — score and flag fake business listings")
    parser.add_argument("input", help="Path to input CSV or XLSX file")
    parser.add_argument("--config", default="spam_config.yaml",
                        help="Path to YAML config file (default: spam_config.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-row processing info")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading config: {args.config}")
    cfg = load_config(args.config)

    print(f"[INFO] Loading input: {args.input}")
    df = load_input(args.input)
    print(f"[INFO] {len(df)} records loaded, {len(df.columns)} columns")

    print("[INFO] Processing...")
    out_df = process(df, cfg, verbose=args.verbose)

    print_summary(out_df)
    save_output(out_df, args.input)


if __name__ == "__main__":
    main()
