"""
Spam Business Detector — Streamlit UI
"""

import io
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# Make spam_detector importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from spam_detector import load_config, process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "spam_config.yaml"

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


def style_by_tier(df: pd.DataFrame) -> pd.io.formats.style.Styler:
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

    with st.expander("Disabled rules"):
        disabled_text = st.text_area(
            "Rule keys to skip (one per line)",
            value="\n".join(cfg.get("disabled_rules", [])),
            height=80,
        )

    with st.expander("SmartyStreets API"):
        st.caption("Required for residential address (RDI) checks. Leave blank to skip.")
        smarty_auth_id = st.text_input(
            "Auth ID",
            value=str(cfg.get("smartystreets_auth_id", "") or ""),
            type="password",
        )
        smarty_auth_token = st.text_input(
            "Auth Token",
            value=str(cfg.get("smartystreets_auth_token", "") or ""),
            type="password",
        )
        if smarty_auth_id and smarty_auth_token:
            st.success("RDI check active", icon="✅")
        else:
            st.warning("Credentials not set — RDI check will be skipped", icon="⚠️")

    st.divider()
    st.caption(f"Config file: `{CONFIG_PATH.name}`")

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
live_cfg["disabled_rules"] = [
    l.strip() for l in disabled_text.splitlines() if l.strip()
]
live_cfg["smartystreets_auth_id"] = smarty_auth_id.strip()
live_cfg["smartystreets_auth_token"] = smarty_auth_token.strip()

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

if not run and "result_df" not in st.session_state:
    st.stop()

# Run detection (or reuse cached result for this upload)
file_key = (uploaded.name, uploaded.size, hcs_thresh, likely_thresh, review_thresh,
            shared_addr, shared_phone, batch_thresh,
            bool(smarty_auth_id), bool(smarty_auth_token))

if run or st.session_state.get("file_key") != file_key:
    rdi_active = bool(smarty_auth_id and smarty_auth_token)
    spinner_msg = (
        "Running detection rules + SmartyStreets RDI checks… (external API calls in progress)"
        if rdi_active else
        "Running detection rules… this may take a moment for large files."
    )
    with st.spinner(spinner_msg):
        try:
            result_df = process(raw_df.copy(), live_cfg, verbose=False)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.exception(e)
            st.stop()
    st.session_state["result_df"] = result_df
    st.session_state["file_key"] = file_key

out_df = st.session_state["result_df"]

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
