"""
Spam Business Detector — Streamlit UI
"""

import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# Make spam_detector importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from spam_detector import find_config, load_config, process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(find_config(str(Path(__file__).parent / "spam_config.yaml")))

TIER_ORDER = ["High Confidence Spam", "Likely Spam", "Review", "Clean"]

TIER_BG = {
    "High Confidence Spam": "#ffcccc",
    "Likely Spam":          "#ffe0b3",
    "Review":               "#fffacc",
    "Clean":                "#d4f4dd",
}

TIER_BADGE = {
    "High Confidence Spam": "🔴 High Confidence Spam",
    "Likely Spam":          "🟠 Likely Spam",
    "Review":               "🟡 Review",
    "Clean":                "🟢 Clean",
}

TIER_METRIC_COLOR = {
    "High Confidence Spam": "inverse",
    "Likely Spam":          "off",
    "Review":               "off",
    "Clean":                "normal",
}

OUTPUT_COLS = ["spam_score", "risk_tier", "flags_triggered", "flag_details", "is_duplicate_of_row"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cfg(path: str) -> dict:
    return load_config(path)


def read_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported file type: {uploaded.name}")


def _detection_cache_key(file_bytes: bytes, cfg: dict) -> str:
    """SHA-256 of file bytes + stable JSON-serialised config. Used as session_state key."""
    cfg_bytes = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(file_bytes + cfg_bytes).hexdigest()


def _run_detection(file_bytes: bytes, file_name: str, cfg: dict,
                   on_progress=None) -> pd.DataFrame:
    """
    Parse the uploaded file and run the full detection pipeline.
    on_progress(fraction, message) is called at each processing step so
    the caller can drive a progress bar.
    """
    buf  = io.BytesIO(file_bytes)
    name = file_name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(buf, dtype=str, keep_default_na=False)
    elif name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(buf, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file type: {file_name}")
    return process(raw, cfg, verbose=False, on_progress=on_progress)


def style_by_tier(df: pd.DataFrame):
    def row_style(row):
        color = TIER_BG.get(row.get("risk_tier", ""), "")
        return [f"background-color: {color}; color: #111" if color else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_excel(out_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Flagged Results", index=False)

        # Summary sheet
        total = len(out_df)
        tier_counts = out_df["risk_tier"].value_counts()
        summary_rows = []
        for tier in TIER_ORDER:
            count = int(tier_counts.get(tier, 0))
            summary_rows.append({
                "Risk Tier": tier,
                "Count": count,
                "Percent": f"{100 * count / total:.1f}%" if total else "0%",
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Top flags sheet
        all_flags = []
        for f_str in out_df["flags_triggered"].dropna():
            all_flags += [f.strip() for f in f_str.split(",") if f.strip()]
        flag_df = (
            pd.Series(Counter(all_flags), name="count")
            .reset_index()
            .rename(columns={"index": "flag"})
            .sort_values("count", ascending=False)
        )
        flag_df.to_excel(writer, sheet_name="Top Flags", index=False)

    return buf.getvalue()


def flag_counter(out_df: pd.DataFrame) -> pd.Series:
    all_flags = []
    for f_str in out_df["flags_triggered"].dropna():
        all_flags += [f.strip() for f in f_str.split(",") if f.strip()]
    return pd.Series(Counter(all_flags)).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Spam Business Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — config overview + live threshold editing
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    cfg = load_cfg(str(CONFIG_PATH))

    with st.expander("Score thresholds", expanded=True):
        review_thresh = st.number_input(
            "Review threshold", value=cfg["score_thresholds"]["review"], min_value=1, step=1)
        likely_thresh = st.number_input(
            "Likely Spam threshold", value=cfg["score_thresholds"]["likely_spam"], min_value=1, step=1)
        hcs_thresh = st.number_input(
            "High Confidence Spam threshold", value=cfg["score_thresholds"]["high_confidence_spam"], min_value=1, step=1)

    with st.expander("Shared-field thresholds"):
        shared_addr = st.number_input(
            "Shared address (flag if ≥ N)", value=cfg["shared_address_threshold"], min_value=2, step=1)
        shared_phone = st.number_input(
            "Shared phone (flag if ≥ N)", value=cfg["shared_phone_threshold"], min_value=2, step=1)
        batch_thresh = st.number_input(
            "Batch submission (flag if ≥ N/day)", value=cfg["batch_submission_threshold"], min_value=2, step=1)

    with st.expander("High-risk industries"):
        industries_text = st.text_area(
            "One per line",
            value="\n".join(cfg["high_risk_industries"]),
            height=180,
        )

    with st.expander("Rule weights"):
        st.caption("Each weight adds to a row's spam score when that rule fires (0 = disabled).")
        w = cfg["weights"]

        def _w(key, label):
            return st.number_input(label, value=int(w.get(key, 0)),
                                   min_value=0, max_value=10, step=1, key=f"w_{key}")

        st.markdown("**Business name**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_high_risk_industry   = _w("high_risk_industry",   "High-risk industry")
            w_generic_name         = _w("generic_name",         "Generic keywords")
        with wc2:
            w_keyword_stuffed_name = _w("keyword_stuffed_name", "Keyword-stuffed name")
            w_same_owner_multi_biz = _w("same_owner_multi_biz", "Same owner, multi-biz")

        st.markdown("**Address**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_virtual_mailbox        = _w("virtual_mailbox",        "Virtual mailbox")
            w_po_box                 = _w("po_box",                 "PO Box")
            w_shared_address         = _w("shared_address",         "Shared address")
        with wc2:
            w_state_zip_mismatch     = _w("state_zip_mismatch",     "State/ZIP mismatch")
            w_address_hidden         = _w("address_hidden",         "Address hidden")
            w_service_area_no_address = _w("service_area_no_address", "Service area, no address")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_residential_address    = _w("residential_address",    "Residential (RDI)")

        st.markdown("**Phone**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_shared_phone      = _w("shared_phone",      "Shared phone")
            w_area_code_mismatch = _w("area_code_mismatch", "Area code mismatch")
        with wc2:
            w_voip_tollfree_phone = _w("voip_tollfree_phone", "VoIP / toll-free")
            w_invalid_phone       = _w("invalid_phone",       "Invalid phone")

        st.markdown("**Website**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_no_website      = _w("no_website",      "No website")
            w_shared_domain   = _w("shared_domain",   "Shared domain")
            w_new_domain      = _w("new_domain",      "Domain/name mismatch")
        with wc2:
            w_generic_builder_site        = _w("generic_builder_site",        "Generic builder site")
            w_landing_page_domain_mismatch = _w("landing_page_domain_mismatch", "Landing page mismatch")

        st.markdown("**Email**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_shared_email        = _w("shared_email",        "Shared email")
            w_free_email          = _w("free_email",          "Free provider")
        with wc2:
            w_email_domain_mismatch = _w("email_domain_mismatch", "Domain mismatch")
            w_autogenerated_email   = _w("autogenerated_email",   "Auto-generated")

        st.markdown("**Duplicates & batches**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_exact_duplicate_row = _w("exact_duplicate_row", "Exact duplicate")
            w_near_duplicate_name = _w("near_duplicate_name", "Near-duplicate name")
        with wc2:
            w_batch_submission    = _w("batch_submission",    "Batch submission")

        st.markdown("**Birdeye fields**")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_no_photos              = _w("no_photos",              "No photos")
            w_no_social              = _w("no_social",              "No social presence")
            w_year_established       = _w("year_established",       "Year established")
        with wc2:
            w_no_hours               = _w("no_hours",               "No hours")
            w_description_missing    = _w("description_missing",    "Description missing/short")
            w_description_stuffed    = _w("description_stuffed",    "Description keyword-stuffed")
        wc1, wc2 = st.columns(2)
        with wc1:
            w_keyword_field_stuffed  = _w("keyword_field_stuffed",  "Keywords field stuffed")

    with st.expander("SmartyStreets API"):
        st.caption("Required for residential address (RDI) checks. Leave blank to skip.")
        _cfg_id    = str(cfg.get("smartystreets_auth_id", "") or "").strip()
        _cfg_token = str(cfg.get("smartystreets_auth_token", "") or "").strip()
        _env_id    = os.environ.get("SMARTYSTREETS_AUTH_ID", "").strip()
        _env_token = os.environ.get("SMARTYSTREETS_AUTH_TOKEN", "").strip()
        _default_id    = _cfg_id    or _env_id
        _default_token = _cfg_token or _env_token
        smarty_auth_id = st.text_input(
            "Auth ID",
            value=_default_id,
            type="password",
        )
        smarty_auth_token = st.text_input(
            "Auth Token",
            value=_default_token,
            type="password",
        )
        if smarty_auth_id and smarty_auth_token:
            _src = "config" if (_cfg_id and _cfg_token) else "env vars"
            st.success(f"RDI check active (credentials from {_src})", icon="✅")
        else:
            if _env_id or _env_token:
                st.warning("Partial credentials in env vars — both ID and token required", icon="⚠️")
            else:
                st.warning("Credentials not set — RDI check will be skipped", icon="⚠️")

    st.divider()
    _cfg_label = CONFIG_PATH.name
    if CONFIG_PATH.name == "spam_config.template.yaml":
        _cfg_label += " ⚠️ (fallback)"
    st.caption(f"Config file: `{_cfg_label}`")

# Build live config from sidebar inputs
live_cfg = dict(cfg)
live_cfg["score_thresholds"] = {
    "review": int(review_thresh),
    "likely_spam": int(likely_thresh),
    "high_confidence_spam": int(hcs_thresh),
}
live_cfg["shared_address_threshold"] = int(shared_addr)
live_cfg["shared_phone_threshold"] = int(shared_phone)
live_cfg["batch_submission_threshold"] = int(batch_thresh)
live_cfg["high_risk_industries"] = [
    l.strip() for l in industries_text.splitlines() if l.strip()
]
live_cfg["disabled_rules"] = cfg.get("disabled_rules", [])
live_cfg["smartystreets_auth_id"] = smarty_auth_id.strip()
live_cfg["smartystreets_auth_token"] = smarty_auth_token.strip()
live_cfg["weights"] = {
    "high_risk_industry":          w_high_risk_industry,
    "generic_name":                w_generic_name,
    "keyword_stuffed_name":        w_keyword_stuffed_name,
    "same_owner_multi_biz":        w_same_owner_multi_biz,
    "virtual_mailbox":             w_virtual_mailbox,
    "po_box":                      w_po_box,
    "shared_address":              w_shared_address,
    "state_zip_mismatch":          w_state_zip_mismatch,
    "address_hidden":              w_address_hidden,
    "service_area_no_address":     w_service_area_no_address,
    "residential_address":         w_residential_address,
    "shared_phone":                w_shared_phone,
    "area_code_mismatch":          w_area_code_mismatch,
    "voip_tollfree_phone":         w_voip_tollfree_phone,
    "invalid_phone":               w_invalid_phone,
    "no_website":                  w_no_website,
    "shared_domain":               w_shared_domain,
    "new_domain":                  w_new_domain,
    "generic_builder_site":        w_generic_builder_site,
    "landing_page_domain_mismatch": w_landing_page_domain_mismatch,
    "shared_email":                w_shared_email,
    "free_email":                  w_free_email,
    "email_domain_mismatch":       w_email_domain_mismatch,
    "autogenerated_email":         w_autogenerated_email,
    "exact_duplicate_row":         w_exact_duplicate_row,
    "near_duplicate_name":         w_near_duplicate_name,
    "duplicate_name":              w.get("duplicate_name", 5),  # not exposed; keep config value
    "cluster_detected":            w.get("cluster_detected", 3),
    "batch_submission":            w_batch_submission,
    "no_photos":                   w_no_photos,
    "no_social":                   w_no_social,
    "year_established":            w_year_established,
    "no_hours":                    w_no_hours,
    "description_missing":         w_description_missing,
    "description_stuffed":         w_description_stuffed,
    "keyword_field_stuffed":       w_keyword_field_stuffed,
}

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🔍 Spam Business Detector")
st.caption("Upload a CSV or XLSX of business listings to score and flag potential spam.")

uploaded = st.file_uploader(
    "Drop your file here",
    type=["csv", "xlsx", "xls"],
    help="Expected columns: business_name, address, city, state, zip, phone, website, industry, email, owner_name, date_added",
)

if uploaded is None:
    st.info("Upload a file to get started. Adjust thresholds in the sidebar before running.")
    st.stop()

# Parse file
try:
    raw_df = read_upload(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.success(f"Loaded **{len(raw_df):,} records** · {len(raw_df.columns)} columns")

with st.expander("Preview raw input (first 5 rows)"):
    st.dataframe(raw_df.head(), use_container_width=True)

run = st.button("🚀 Run Spam Detector", type="primary", use_container_width=True)
if run:
    st.session_state["show_results"] = True
    st.session_state.pop("_det_key", None)   # force re-run on explicit click

if not st.session_state.get("show_results"):
    st.stop()

_key = _detection_cache_key(uploaded.getvalue(), live_cfg)

if st.session_state.get("_det_key") != _key:
    _prog = st.progress(0, text="Starting…")
    try:
        out_df = _run_detection(
            uploaded.getvalue(), uploaded.name, live_cfg,
            on_progress=lambda f, m: _prog.progress(min(f, 1.0), text=m),
        )
    except Exception as e:
        _prog.empty()
        st.error(f"Detection failed: {e}")
        st.exception(e)
        st.stop()
    st.session_state["_det_key"]    = _key
    st.session_state["_det_result"] = out_df
    _prog.empty()
else:
    out_df = st.session_state["_det_result"]

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Summary")

total = len(out_df)
tier_counts = out_df["risk_tier"].value_counts()

col_t, col_hcs, col_ls, col_rv, col_cl = st.columns(5)
col_t.metric("Total Records", f"{total:,}")
col_hcs.metric("🔴 High Conf Spam", int(tier_counts.get("High Confidence Spam", 0)),
               delta=f"{100 * tier_counts.get('High Confidence Spam', 0) / total:.0f}%",
               delta_color="inverse")
col_ls.metric("🟠 Likely Spam", int(tier_counts.get("Likely Spam", 0)),
              delta=f"{100 * tier_counts.get('Likely Spam', 0) / total:.0f}%",
              delta_color="inverse")
col_rv.metric("🟡 Review", int(tier_counts.get("Review", 0)),
              delta=f"{100 * tier_counts.get('Review', 0) / total:.0f}%",
              delta_color="off")
col_cl.metric("🟢 Clean", int(tier_counts.get("Clean", 0)),
              delta=f"{100 * tier_counts.get('Clean', 0) / total:.0f}%",
              delta_color="normal")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

chart_col, flag_col = st.columns(2)

with chart_col:
    st.markdown("**Risk tier distribution**")
    tier_df = (
        pd.DataFrame({"Tier": TIER_ORDER,
                      "Count": [int(tier_counts.get(t, 0)) for t in TIER_ORDER]})
        .set_index("Tier")
    )
    st.bar_chart(tier_df, color="#e05c5c")

with flag_col:
    st.markdown("**Top 10 flags triggered**")
    top_flags = flag_counter(out_df).head(10)
    if not top_flags.empty:
        st.bar_chart(top_flags.rename("Count"), color="#e08c3c")
    else:
        st.caption("No flags triggered.")

# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

with st.expander("Cluster details"):
    cluster_info = []
    for flag_key, label in [
        ("SHARED_PHONE",        "Shared phone"),
        ("SHARED_ADDRESS",      "Shared address"),
        ("SHARED_EMAIL",        "Shared email"),
        ("SHARED_DOMAIN",       "Shared domain"),
        ("RESIDENTIAL_ADDRESS", "Residential address (RDI)"),
    ]:
        flagged = out_df[out_df["flags_triggered"].str.contains(flag_key, na=False)]
        cluster_info.append({"Signal": label, "Records flagged": len(flagged)})
    st.dataframe(pd.DataFrame(cluster_info), use_container_width=True, hide_index=True)

    # Reseller breakdown if present
    reseller_col = next(
        (c for c in out_df.columns if c.lower() in
         ("reseller_id", "reseller_name", "reseller", "partner_id", "partner_name")),
        None,
    )
    if reseller_col:
        st.markdown("**Reseller breakdown**")
        breakdown = (
            out_df.groupby(reseller_col)["spam_score"]
            .agg(count="count", avg_score="mean")
            .sort_values("avg_score", ascending=False)
            .reset_index()
        )
        breakdown["avg_score"] = breakdown["avg_score"].round(1)
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

# Residential address hits — only rendered when any rows were flagged
_rdi_flagged = out_df[out_df["flags_triggered"].str.contains("RESIDENTIAL_ADDRESS", na=False)]
if not _rdi_flagged.empty:
    with st.expander(f"🏠 Residential address hits ({len(_rdi_flagged)} records)", expanded=True):
        if not (smarty_auth_id and smarty_auth_token):
            st.info(
                "These results are from a previous run with credentials set. "
                "Add SmartyStreets credentials in the sidebar to run fresh RDI checks.",
                icon="ℹ️",
            )
        # Show the most useful columns for reviewing residential hits
        addr_col = next((c for c in out_df.columns if c.lower() in
                         ("address", "street", "street_address")), None)
        name_col = next((c for c in out_df.columns if c.lower() in
                         ("business_name", "name", "company")), None)
        city_col  = next((c for c in out_df.columns if c.lower() == "city"), None)
        state_col = next((c for c in out_df.columns if c.lower() == "state"), None)

        show_cols = [c for c in [name_col, addr_col, city_col, state_col,
                                  "spam_score", "risk_tier", "flags_triggered"] if c]
        rdi_display = _rdi_flagged[show_cols].copy()
        rdi_display["spam_score"] = pd.to_numeric(rdi_display["spam_score"], errors="coerce")
        st.dataframe(
            style_by_tier(rdi_display),
            use_container_width=True,
            column_config={
                "spam_score":      st.column_config.NumberColumn("Score", format="%d"),
                "risk_tier":       st.column_config.TextColumn("Risk Tier"),
                "flags_triggered": st.column_config.TextColumn("Flags", width="large"),
            },
            hide_index=True,
        )
elif smarty_auth_id and smarty_auth_token:
    st.success("No residential addresses detected by SmartyStreets RDI check.", icon="🏠")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Flagged Results")

# Filter controls
filter_cols = st.columns([2, 2, 4])
with filter_cols[0]:
    tier_filter = st.multiselect(
        "Filter by risk tier",
        options=TIER_ORDER,
        default=TIER_ORDER,
    )
with filter_cols[1]:
    min_score = st.number_input("Min spam score", min_value=0, value=0, step=1)
with filter_cols[2]:
    flag_search = st.text_input("Filter by flag name (e.g. SHARED_PHONE)", placeholder="")

display_df = out_df.copy()
if tier_filter:
    display_df = display_df[display_df["risk_tier"].isin(tier_filter)]
if min_score > 0:
    display_df = display_df[display_df["spam_score"].astype(float) >= min_score]
if flag_search.strip():
    display_df = display_df[
        display_df["flags_triggered"].str.contains(flag_search.strip(), case=False, na=False)
    ]

st.caption(f"Showing **{len(display_df):,}** of **{total:,}** records")

# Reorder: output cols first, then original input cols
orig_cols = [c for c in display_df.columns if c not in OUTPUT_COLS]
ordered_cols = OUTPUT_COLS + orig_cols
display_df = display_df[ordered_cols]

# Make spam_score numeric for sorting
display_df["spam_score"] = pd.to_numeric(display_df["spam_score"], errors="coerce")

styled = style_by_tier(display_df)
st.dataframe(
    styled,
    use_container_width=True,
    height=520,
    column_config={
        "spam_score":         st.column_config.NumberColumn("Score", format="%d"),
        "risk_tier":          st.column_config.TextColumn("Risk Tier", width="medium"),
        "flags_triggered":    st.column_config.TextColumn("Flags", width="large"),
        "flag_details":       st.column_config.TextColumn("Details", width="large"),
        "is_duplicate_of_row": st.column_config.TextColumn("Dup of Row", width="small"),
    },
)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

st.divider()

excel_bytes = build_excel(out_df)
st.download_button(
    label="⬇️ Download flagged results (.xlsx)",
    data=excel_bytes,
    file_name=f"{Path(uploaded.name).stem}_flagged.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
    use_container_width=True,
)
st.caption("Output includes three sheets: Flagged Results, Summary, and Top Flags.")
