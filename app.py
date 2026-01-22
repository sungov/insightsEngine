# app.py
import os
import re
import json
import math
import hashlib
from datetime import datetime
import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from langchain_openai import ChatOpenAI

# ============================================================
# Page + Theme
# ============================================================
st.set_page_config(page_title="tsworks | People Insights (Exec)", layout="wide")

APP_CSS = """
<style>
/* --- Global --- */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
small { color: #6b7280; }

/* --- Card grid for compact KPIs --- */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(180px, 1fr));
  gap: 14px;
  margin: 10px 0 2px 0;
}
.kpi-card{
  background: white;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 14px;
  padding: 14px 14px;
  box-shadow: 0 1px 0 rgba(15,23,42,.03);
}
.kpi-title{ font-size: 12px; color: rgba(15,23,42,.62); margin-bottom: 6px; }
.kpi-value{ font-size: 26px; font-weight: 700; color: rgba(15,23,42,.95); line-height: 1.05; }
.kpi-sub{ margin-top: 8px; font-size: 12px; color: rgba(15,23,42,.55); }

/* --- Section headers --- */
.section {
  background: rgba(248,250,252,.8);
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: 16px;
  padding: 14px 14px;
  margin: 10px 0 14px 0;
}

/* --- Chips --- */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top: 8px; }
.chip {
  display:inline-flex; align-items:center; gap:6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(15,23,42,.10);
  background: white;
  font-size: 12px;
  color: rgba(15,23,42,.75);
}

/* --- Tables --- */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(15,23,42,.08); }

/* --- Copilot container --- */
.copilot-box {
  border: 1px solid rgba(15,23,42,.08);
  border-radius: 16px;
  padding: 12px 12px;
  background: rgba(255,255,255,.9);
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ============================================================
# Constants + Helpers
# ============================================================
SAT_MAP = {
    "Extremely satisfied": 10, "Satisfied": 8, "Somewhat satisfied": 7,
    "Neutral": 5, "Somewhat dissatisfied": 3, "Dissatisfied": 2, "Extremely dissatisfied": 0
}
MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

MONTH_ORDER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
MONTHS_CANON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

REQUIRED_COLS = ["Name", "Department", "Reporting Manager", "Year", "Month"]

def safe_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def extract_first_json(text: str):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def wants_chart(user_q: str) -> bool:
    """Heuristic: user explicitly wants a chart/plot."""
    q = (user_q or "").lower()
    triggers = ["plot", "chart", "graph", "visual", "trend", "line chart", "bar chart", "pie chart", "histogram", "scatter"]
    return any(t in q for t in triggers)

def wants_names_list(user_q: str) -> bool:
    """Heuristic: user wants a list/table of people/managers/etc."""
    q = (user_q or "").lower()
    triggers = ["which employees", "who are", "list employees", "names of", "show employees", "top employees", "employee list"]
    return any(t in q for t in triggers)
  
def add_period_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Month"] = d["Month"].astype(str).str.strip().str[:3].str.title()
    d["_MonthNum"] = d["Month"].map(MONTH_ORDER)
    d["_PeriodKey"] = d["Year"] * 100 + d["_MonthNum"]
    d["PeriodDate"] = pd.to_datetime(
        d["Year"].astype("Int64").astype(str) + "-" + d["_MonthNum"].astype("Int64").astype(str) + "-01",
        errors="coerce"
    )
    d["_PeriodLabel"] = d["Month"] + " " + d["Year"].astype("Int64").astype(str)
    return d

def apply_time_window(df: pd.DataFrame, tw: str, default_year: int, default_month: str) -> pd.DataFrame:
    """Time window filter on full dataset. 'this_month' uses latest period in dataset unless user specifies."""
    d = df.copy()
    d = d.dropna(subset=["PeriodDate"]) if "PeriodDate" in d.columns else d

    tw = (tw or "all").strip().lower()

    if "PeriodDate" not in d.columns:
        return d

    if tw == "this_month":
        # interpret as latest available in file (default passed in already)
        # If df already has Year/Month columns normalized, use them
        if "Year" in d.columns and "Month" in d.columns:
            return d[(d["Year"].astype("Int64") == int(default_year)) & (d["Month"].astype(str) == str(default_month))].copy()
        # fallback to max PeriodDate month
        end = d["PeriodDate"].max()
        start = end.replace(day=1)
        return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()

    def window_months(n):
        end = d["PeriodDate"].max()
        start = (end - pd.DateOffset(months=n)).replace(day=1)
        return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()

    if tw == "last_3_months":
        return window_months(3)
    if tw == "last_6_months":
        return window_months(6)
    if tw == "last_12_months":
        return window_months(12)

    return d

def _normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
  
def safe_number(x, default=None):
    try:
        if pd.isna(x): return default
        if isinstance(x, (int,float)): return float(x)
        s = str(x).strip()
        # accept "85%" or "85"
        s = s.replace("%","")
        return float(s)
    except Exception:
        return default

def compute_goal_score_from_text(val) -> float:
    """
    Heuristic: if Goal Progress is numeric/percent -> map to 0..10.
    Otherwise keyword-based scoring.
    """
    num = safe_number(val, None)
    if num is not None:
        # treat 0..100 or 0..10
        if num > 10:
            return max(0, min(10, num / 10.0))
        return max(0, min(10, num))
    s = str(val).lower()
    if s in ["n/a", "na", "none", ""] or s == "n/a":
        return 5.0
    pos = ["on track", "good", "great", "completed", "achieved", "progress", "done", "ahead"]
    neg = ["blocked", "stuck", "behind", "delayed", "risk", "struggle", "not", "issue"]
    score = 5.0
    if any(w in s for w in pos): score += 2.0
    if any(w in s for w in neg): score -= 2.0
    return max(0, min(10, score))

def compute_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Health_Index (0-100) composite of:
      - Sat_Score (0-10) -> 0..100
      - Mood_Score (1-5) -> 0..100
      - Goal_Score (0-10) -> 0..100
    Weighted for exec readability.
    """
    d = df.copy()
    if "Sat_Score" not in d.columns:
        d["Sat_Score"] = 5
    if "Mood_Score" not in d.columns:
        d["Mood_Score"] = 3

    if "Goal Progress" in d.columns:
        d["Goal_Score"] = d["Goal Progress"].apply(compute_goal_score_from_text)
    else:
        d["Goal_Score"] = 5.0

    sat_100 = (d["Sat_Score"] / 10.0) * 100.0
    mood_100 = ((d["Mood_Score"] - 1) / 4.0) * 100.0  # 1..5 -> 0..100
    goals_100 = (d["Goal_Score"] / 10.0) * 100.0

    d["Health_Index"] = (0.45 * sat_100 + 0.25 * mood_100 + 0.30 * goals_100).round(1)
    return d

def classify_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Risk_Level based on:
      - current Health_Index
      - Burned Out / Challenged mood
      - negative trend last 3 months
    """
    d = df.copy()
    d = add_period_cols(d)
    d = d.dropna(subset=["PeriodDate"])

    # compute month-on-month trend per employee
    d = d.sort_values(["Name","PeriodDate"])
    d["Health_Delta_1"] = d.groupby("Name")["Health_Index"].diff(1)
    d["Health_Delta_3"] = d.groupby("Name")["Health_Index"].diff(3)

    def risk_row(r):
        hi = safe_number(r.get("Health_Index"), 50) or 50
        mood = str(r.get("How are you feeling overall this month?", "")).strip().lower()
        d1 = safe_number(r.get("Health_Delta_1"), 0) or 0
        d3 = safe_number(r.get("Health_Delta_3"), 0) or 0

        # base
        if hi >= 75:
            base = "Green"
        elif hi >= 60:
            base = "Amber"
        else:
            base = "Red"

        # mood override
        if "burned" in mood or "burnt" in mood:
            return "Red"
        if "challenged" in mood and base == "Green":
            base = "Amber"

        # trend override
        if d3 <= -12 or d1 <= -8:
            if base == "Green":
                base = "Amber"
            elif base == "Amber":
                base = "Red"
        return base

    d["Risk_Level"] = d.apply(risk_row, axis=1)
    return d


def sanitize_chart_spec(spec: dict, df: pd.DataFrame) -> dict:
    """
    Safety net:
    - Map synonyms to actual columns
    - Ensure x/y/group_by exist
    - If invalid, downgrade to text/table mode by setting mode='answer'
    """
    if not isinstance(spec, dict):
        return {"mode": "answer", "answer": "I couldn’t interpret that request."}

    # allowed chart types
    allowed_chart_types = {"line", "bar", "pie", "hist", "scatter"}
    chart_type = (spec.get("chart_type") or "line").strip().lower()
    if chart_type not in allowed_chart_types:
        chart_type = "line"

    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}

    # synonym mapping
    # (extend this list as you introduce new KPIs)
    synonyms = {
        "period": "_PeriodLabel",
        "month": "_PeriodLabel",
        "timelabel": "_PeriodLabel",
        "date": "PeriodDate",
        "dept": "Department",
        "department": "Department",
        "manager": "Reporting Manager",
        "reportingmanager": "Reporting Manager",
        "employee": "Name",
        "name": "Name",
        "risk": "Risk_Level",
        "risklevel": "Risk_Level",
        "healthindex": "Health_Index",
        "satisfaction": "Sat_Score",
        "mood": "Mood_Score",
    }

    def resolve_col(v):
        if v is None:
            return None
        v = str(v).strip()
        if v in cols:
            return v
        key = _normalize_colname(v)
        if key in synonyms and synonyms[key] in cols:
            return synonyms[key]
        if key in norm_map:
            return norm_map[key]
        return None

    x = resolve_col(spec.get("x"))
    y_raw = spec.get("y")
    group_by = resolve_col(spec.get("group_by")) if spec.get("group_by") else None

    # resolve y: allow "Sat_Score_mean", "count", "Health_Index_mean"
    y = None
    y_agg = None
    if isinstance(y_raw, str):
        y_raw = y_raw.strip()
        if y_raw.lower() in ["count", "responses", "total", "total_responses"]:
            y = "count"
            y_agg = "count"
        else:
            # mean suffix
            if y_raw.endswith("_mean"):
                base = resolve_col(y_raw.replace("_mean", ""))
                if base:
                    y = base
                    y_agg = "mean"
            else:
                base = resolve_col(y_raw)
                if base:
                    y = base
                    y_agg = None

    # if x missing, choose safe default
    if not x:
        if "_PeriodLabel" in cols:
            x = "_PeriodLabel"
        elif "Department" in cols:
            x = "Department"
        else:
            # cannot chart without x
            spec["mode"] = "answer"
            spec["answer"] = "I can answer this in text, but I can’t find a suitable chart axis in the dataset."
            return spec

    # if y missing for chart types that need it
    if chart_type in {"line", "bar", "scatter"} and not y:
        # pick a default metric if exists
        for candidate in ["Health_Index", "Sat_Score", "Mood_Score"]:
            if candidate in cols:
                y = candidate
                y_agg = "mean"
                break
        if not y:
            spec["mode"] = "answer"
            spec["answer"] = "I can summarize this in text, but I can’t find a numeric metric to plot."
            return spec

    # for pie, y optional; default to count
    if chart_type == "pie" and not y:
        y = "count"
        y_agg = "count"

    spec["chart_type"] = chart_type
    spec["x"] = x
    spec["y"] = y
    spec["_y_agg"] = y_agg
    spec["group_by"] = group_by
    return spec

def build_chart_safe(df: pd.DataFrame, spec: dict):
    """
    Build plotly chart with aggregation + chronological month ordering if _PeriodLabel exists.
    """
    chart_type = spec["chart_type"]
    x = spec["x"]
    y = spec["y"]
    group_by = spec.get("group_by")
    y_agg = spec.get("_y_agg")

    d = df.copy()

    # chronological ordering for _PeriodLabel
    # expects you created PeriodDate + _PeriodLabel earlier in pipeline
    if x == "_PeriodLabel" and "PeriodDate" in d.columns and "_PeriodLabel" in d.columns:
        sort_cols = ["PeriodDate"]
    else:
        sort_cols = None

    # aggregation
    if y == "count" and y_agg == "count":
        group_cols = [x] + ([group_by] if group_by else [])
        plot_df = d.groupby(group_cols, as_index=False).size().rename(columns={"size": "Value"})
        if sort_cols and x == "_PeriodLabel":
            # preserve order by PeriodDate (merge back)
            tmp = d[["_PeriodLabel", "PeriodDate"]].drop_duplicates()
            plot_df = plot_df.merge(tmp, on="_PeriodLabel", how="left").sort_values("PeriodDate")
        if chart_type == "bar":
            return px.bar(plot_df, x=x, y="Value", color=group_by, title="Count")
        if chart_type == "pie":
            return px.pie(plot_df, names=x, values="Value", hole=0.4, title="Count")
        return px.line(plot_df, x=x, y="Value", color=group_by, markers=True, title="Count")

    # numeric aggregation
    if y_agg in ["mean", "sum"]:
        group_cols = [x] + ([group_by] if group_by else [])
        plot_df = d.groupby(group_cols, as_index=False).agg({y: y_agg})
        if sort_cols and x == "_PeriodLabel":
            tmp = d[["_PeriodLabel", "PeriodDate"]].drop_duplicates()
            plot_df = plot_df.merge(tmp, on="_PeriodLabel", how="left").sort_values("PeriodDate")
        if chart_type == "line":
            return px.line(plot_df, x=x, y=y, color=group_by, markers=True, title=f"{y_agg.title()} {y}")
        if chart_type == "bar":
            return px.bar(plot_df, x=x, y=y, color=group_by, title=f"{y_agg.title()} {y}")
        if chart_type == "scatter":
            return px.scatter(plot_df, x=x, y=y, color=group_by, title=f"{y_agg.title()} {y}")
        if chart_type == "pie":
            return px.pie(plot_df, names=x, values=y, hole=0.4, title=f"{y_agg.title()} {y}")
        if chart_type == "hist":
            return px.histogram(d, x=y, color=group_by, title=f"Distribution of {y}")
        return px.line(plot_df, x=x, y=y, color=group_by, markers=True)

    # raw plot (no agg) fallback
    if chart_type == "hist":
        return px.histogram(d, x=y, color=group_by, title=f"Distribution of {y}")
    if chart_type == "scatter":
        return px.scatter(d, x=x, y=y, color=group_by, title=f"{y} vs {x}")
    if chart_type == "bar":
        return px.bar(d, x=x, y=y, color=group_by, title=f"{y} by {x}")
    if chart_type == "pie":
        return px.pie(d, names=x, hole=0.4, title=f"{x} breakdown")
    return px.line(d, x=x, y=y, color=group_by, markers=True, title=f"{y} trend")

def kpi_row(items):
    """
    items = [{"title":..., "value":..., "sub":...}, ...]
    """
    cols = st.columns(len(items), gap="small")
    for i, it in enumerate(items):
        with cols[i]:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-title">{it['title']}</div>
                  <div class="kpi-value">{it['value']}</div>
                  <div class="kpi-sub">{it.get('sub','')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def require_cols(df: pd.DataFrame, cols: list) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in uploaded file: {missing}")
        return False
    return True

# ============================================================
# Header
# ============================================================
st.title("👥 tsworks People Insights — Executive")
st.caption("Senior management view: org + department + manager + employee trends with AI Copilot.")

# ============================================================
# Sidebar (clear + less confusing)
# ============================================================
with st.sidebar:
    st.header("Data")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Set OPENAI_API_KEY in your environment.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Employee Pulse (Excel/CSV)", type=["xlsx","csv"])

    st.divider()
    st.header("Dashboard Scope")
    st.caption("These filters affect dashboard & tables (not the AI Copilot).")
    # (year/month defaults are set after load)
    # dept/manager applied after the time scope

# ============================================================
# Load & prep data
# ============================================================
if not uploaded_file:
    st.info("Upload an Excel/CSV to start.")
    st.stop()

# reset state when file changes
if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    # dashboard widgets
    for k in ["dash_year","dash_month","dash_dept","dash_mgr"]:
        st.session_state.pop(k, None)
    # copilot memory persists across filter changes; reset only on new file
    st.session_state["copilot_messages"] = []
    st.session_state["saved_insights"] = []
    st.session_state["last_copilot"] = None

df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.fillna("N/A")

if not require_cols(df_raw, REQUIRED_COLS):
    st.stop()

# normalize time cols
df_raw["Year"] = pd.to_numeric(df_raw["Year"], errors="coerce")
df_raw["Month"] = df_raw["Month"].astype(str).str.strip().str[:3].str.title()

# derived scores
if "How satisfied are you working at tsworks?" in df_raw.columns:
    df_raw["Sat_Score"] = df_raw["How satisfied are you working at tsworks?"].map(SAT_MAP).fillna(5)
else:
    df_raw["Sat_Score"] = 5

if "How are you feeling overall this month?" in df_raw.columns:
    df_raw["Mood_Score"] = df_raw["How are you feeling overall this month?"].map(MOOD_MAP).fillna(3)
else:
    df_raw["Mood_Score"] = 3

df_full = add_period_cols(df_raw)
df_full = compute_health_index(df_full)
df_full = classify_risk(df_full)

# latest period
latest_key = df_full["_PeriodKey"].dropna().max()
latest_row = df_full.loc[df_full["_PeriodKey"] == latest_key].iloc[0]
LATEST_YEAR = int(latest_row["Year"])
LATEST_MONTH = str(latest_row["Month"])

years = sorted(df_full["Year"].dropna().unique().astype(int).tolist())
if not years:
    st.error("No valid Year values found.")
    st.stop()

# ============================================================
# Sidebar: Dashboard filters (defaults to latest)
# ============================================================
with st.sidebar:
    # default year
    if st.session_state.get("dash_year") not in years:
        st.session_state["dash_year"] = LATEST_YEAR

    dash_year = st.selectbox("Year", years, index=years.index(st.session_state["dash_year"]), key="dash_year")

    # months for that year
    df_year = df_full[df_full["Year"] == dash_year]
    months = [m for m in MONTHS_CANON if m in set(df_year["Month"].dropna())]
    if not months:
        months = [LATEST_MONTH]

    if st.session_state.get("dash_month") not in months:
        st.session_state["dash_month"] = max(months, key=lambda m: MONTH_ORDER.get(m, 0))

    dash_month = st.selectbox("Month", months, index=months.index(st.session_state["dash_month"]), key="dash_month")

    # dept/mgr options from time slice
    base_slice = df_full[(df_full["Year"] == dash_year) & (df_full["Month"] == dash_month)].copy()
    depts = sorted(base_slice["Department"].dropna().unique().tolist())
    mgrs = sorted(base_slice["Reporting Manager"].dropna().unique().tolist())

    if st.session_state.get("dash_dept") not in (["All"] + depts):
        st.session_state["dash_dept"] = "All"
    dash_dept = st.selectbox("Department", ["All"] + depts, key="dash_dept")

    # manager list depends on dept filter for clarity
    mgr_slice = base_slice.copy()
    if dash_dept != "All":
        mgr_slice = mgr_slice[mgr_slice["Department"] == dash_dept]
    mgrs2 = sorted(mgr_slice["Reporting Manager"].dropna().unique().tolist())

    if st.session_state.get("dash_mgr") not in (["All"] + mgrs2):
        st.session_state["dash_mgr"] = "All"
    dash_mgr = st.selectbox("Manager", ["All"] + mgrs2, key="dash_mgr")

    st.divider()
    if st.button("Reset dashboard filters to latest", use_container_width=True):
        st.session_state["dash_year"] = LATEST_YEAR
        st.session_state["dash_month"] = LATEST_MONTH
        st.session_state["dash_dept"] = "All"
        st.session_state["dash_mgr"] = "All"
        st.rerun()

# ============================================================
# Apply dashboard filters
# ============================================================
df_dash = df_full[(df_full["Year"] == dash_year) & (df_full["Month"] == dash_month)].copy()
if dash_dept != "All":
    df_dash = df_dash[df_dash["Department"] == dash_dept]
if dash_mgr != "All":
    df_dash = df_dash[df_dash["Reporting Manager"] == dash_mgr]

# ============================================================
# Tabs
# ============================================================
tab_exec, tab_org, tab_emp, tab_watch, tab_copilot, tab_saved = st.tabs([
    "Executive Snapshot",
    "Org & Dept Insights",
    "Employee Explorer",
    "Watchlist",
    "People AI Copilot",
    "Saved Insights",
])

# ============================================================
# Executive Snapshot (compact cards + clear scope)
# ============================================================
with tab_exec:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Executive Snapshot")
    st.caption(f"Dashboard scope: **{dash_month} {dash_year}** • Department: **{dash_dept}** • Manager: **{dash_mgr}**")
    st.markdown("</div>", unsafe_allow_html=True)

    total = len(df_dash)
    if total == 0:
        st.warning("No responses in this scope.")
    else:
        promoters = len(df_dash[df_dash["Sat_Score"] >= 9])
        detractors = len(df_dash[df_dash["Sat_Score"] <= 6])
        nps = round(((promoters - detractors) / total) * 100)

        avg_sat = round(df_dash["Sat_Score"].mean(), 2)
        avg_mood = round(df_dash["Mood_Score"].mean(), 2)
        avg_hi = round(df_dash["Health_Index"].mean(), 1)

        red = int((df_dash["Risk_Level"] == "Red").sum())
        amber = int((df_dash["Risk_Level"] == "Amber").sum())
        green = int((df_dash["Risk_Level"] == "Green").sum())

        kpi_row([
            {"title":"Avg Health Index", "value": f"{avg_hi}", "sub":"0–100 composite (Satisfaction, Mood, Goals)"},
            {"title":"Employee NPS", "value": f"{nps}", "sub":"Promoters ≥9 vs Detractors ≤6"},
            {"title":"Avg Satisfaction", "value": f"{avg_sat}/10", "sub":"Current scope average"},
            {"title":"Avg Mood", "value": f"{avg_mood}/5", "sub":"Current scope average"},
        ])

        st.markdown('<div class="chips">', unsafe_allow_html=True)
        st.markdown(f'<span class="chip">🧾 Responses: <b>{total}</b></span>', unsafe_allow_html=True)
        st.markdown(f'<span class="chip">🟢 Green: <b>{green}</b></span>', unsafe_allow_html=True)
        st.markdown(f'<span class="chip">🟠 Amber: <b>{amber}</b></span>', unsafe_allow_html=True)
        st.markdown(f'<span class="chip">🔴 Red: <b>{red}</b></span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        c1, c2 = st.columns([1.2, 1])
        with c1:
            # dept/manager bar for current scope
            group_by = "Reporting Manager" if dash_dept != "All" else "Department"
            g = df_dash.groupby(group_by, as_index=False)["Health_Index"].mean().sort_values("Health_Index")
            fig = px.bar(g, x=group_by, y="Health_Index", title=f"Health Index by {group_by}")
            st.plotly_chart(fig, use_container_width=True, key=f"exec_hi_{dash_year}_{dash_month}_{dash_dept}_{dash_mgr}")

        with c2:
            mood_col = "How are you feeling overall this month?" if "How are you feeling overall this month?" in df_dash.columns else None
            if mood_col:
                fig2 = px.pie(df_dash, names=mood_col, title="Mood Distribution", hole=0.4)
                st.plotly_chart(fig2, use_container_width=True, key=f"exec_mood_{dash_year}_{dash_month}_{dash_dept}_{dash_mgr}")
            else:
                st.info("Mood question column not found in file.")

# ============================================================
# Org & Dept Insights (trends + comparisons)
# ============================================================
with tab_org:
    st.subheader("Org & Department Insights")
    st.caption("Trends always use the full dataset (across months). You can optionally scope by department/manager in the controls below.")

    cA, cB, cC = st.columns([1,1,1])
    with cA:
        scope_dept = st.selectbox("Scope Department", ["All"] + sorted(df_full["Department"].dropna().unique().tolist()), key="org_scope_dept")
    with cB:
        scope_mgr = st.selectbox("Scope Manager", ["All"] + sorted(df_full["Reporting Manager"].dropna().unique().tolist()), key="org_scope_mgr")
    with cC:
        win = st.selectbox("Time window", ["last_3_months","last_6_months","last_12_months","all"], index=1, key="org_scope_win")

    org_df = apply_time_window(df_full, win, LATEST_YEAR, LATEST_MONTH)
    if scope_dept != "All":
        org_df = org_df[org_df["Department"] == scope_dept]
    if scope_mgr != "All":
        org_df = org_df[org_df["Reporting Manager"] == scope_mgr]

    # Trend: Health Index
    tr = org_df.dropna(subset=["PeriodDate"]).groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Responses=("Name","count")
    ).sort_values("PeriodDate")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig = px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
        st.plotly_chart(fig, use_container_width=True, key=f"org_tr_hi_{safe_hash(str((scope_dept,scope_mgr,win)))}")
    with col2:
        fig = px.line(tr, x="_PeriodLabel", y="Responses", markers=True, title="Responses Trend")
        st.plotly_chart(fig, use_container_width=True, key=f"org_tr_resp_{safe_hash(str((scope_dept,scope_mgr,win)))}")

    st.divider()

    # Dept comparison (latest period)
    latest_df = df_full[df_full["_PeriodKey"] == latest_key].copy()
    dep = latest_df.groupby("Department", as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Responses=("Name","count"),
        Red=("Risk_Level", lambda s: int((s=="Red").sum())),
        Amber=("Risk_Level", lambda s: int((s=="Amber").sum()))
    ).sort_values("Health")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(dep, x="Department", y="Health", title=f"Department Health (Latest: {LATEST_MONTH} {LATEST_YEAR})")
        st.plotly_chart(fig, use_container_width=True, key="dep_hi_latest")
    with c2:
        fig = px.bar(dep, x="Department", y="Red", title="Red Risk Count by Department (Latest)")
        st.plotly_chart(fig, use_container_width=True, key="dep_red_latest")

    with st.expander("View department table (latest period)"):
        st.dataframe(dep, use_container_width=True, hide_index=True)

# ============================================================
# Employee Explorer (search + pagination + drilldown + history)
# ============================================================
with tab_emp:
    st.subheader("Employee Explorer")
    st.caption("Browse employees in the selected dashboard scope, then drill down to view their history across months.")

    # base table = df_dash scope
    base = df_dash.copy()
    if base.empty:
        st.warning("No employees in this scope.")
    else:
        c1, c2, c3 = st.columns([1.2,1,1])
        with c1:
            q = st.text_input("Search employee", placeholder="Type a name…", key="emp_search").strip()
        with c2:
            page_size = st.selectbox("Rows per page", [10,25,50,100], index=1, key="emp_page_size")
        with c3:
            sort_opt = st.selectbox("Sort by", ["Risk_Level","Health_Index","Sat_Score","Mood_Score","Name"], index=0, key="emp_sort")

        view = base.copy()
        if q:
            view = view[view["Name"].astype(str).str.contains(q, case=False, na=False)]

        # sort
        if sort_opt == "Risk_Level":
            order = {"Red":0,"Amber":1,"Green":2}
            view["_rk"] = view["Risk_Level"].map(order).fillna(9)
            view = view.sort_values(["_rk","Health_Index"], ascending=[True, True]).drop(columns=["_rk"])
        else:
            view = view.sort_values(sort_opt, ascending=(sort_opt!="Name"))

        total = len(view)
        pages = max(1, math.ceil(total / page_size))
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="emp_page")

        start = (page-1)*page_size
        end = start + page_size
        page_df = view.iloc[start:end].copy()

        cols = [c for c in [
            "Name","Department","Reporting Manager",
            "Risk_Level","Health_Index","Sat_Score","Mood_Score",
            "How satisfied are you working at tsworks?",
            "How are you feeling overall this month?",
            "Goal Progress",
            "Key Accomplishments this Month",
            "What’s not going well or causing disappointment?"
        ] if c in page_df.columns]

        st.dataframe(page_df[cols], use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Employee Deep Dive")

        emp_list = sorted(view["Name"].dropna().unique().tolist())
        chosen = st.selectbox("Choose an employee", ["-- Select --"] + emp_list, key="emp_choose")
        if chosen != "-- Select --":
            hist = df_full[df_full["Name"] == chosen].copy()
            hist = hist.sort_values("PeriodDate", ascending=False)

            top = hist.iloc[0]
            a,b,c = st.columns(3)
            a.metric("Employee", str(top.get("Name","")))
            b.metric("Department", str(top.get("Department","")))
            c.metric("Manager", str(top.get("Reporting Manager","")))

            # Trend chart
            tr = hist.dropna(subset=["PeriodDate"]).groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
                Health=("Health_Index","mean"),
                Sat=("Sat_Score","mean"),
                Mood=("Mood_Score","mean")
            ).sort_values("PeriodDate")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
                st.plotly_chart(fig, use_container_width=True, key=f"emp_hi_{safe_hash(chosen)}")
            with c2:
                fig = px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
                st.plotly_chart(fig, use_container_width=True, key=f"emp_sat_{safe_hash(chosen)}")

            # Responses by month
            show_cols = [c for c in [
                "How satisfied are you working at tsworks?",
                "How are you feeling overall this month?",
                "Goal Progress",
                "Key Accomplishments this Month",
                "What’s not going well or causing disappointment?",
                "Any concerns, blockers, or risks?",
                "Do you need support from bench resources or other teams?",
                "How is your work-life balance?",
                "How is your current workload?",
                "Are you currently supporting or mentoring junior team members?",
                "Suggestions for process or workflow improvements",
                "Planned PTO this month and coverage plan"
            ] if c in hist.columns]

            for _, r in hist.iterrows():
                period = f"{r.get('Month','')} {r.get('Year','')}"
                risk = r.get("Risk_Level","")
                hi = r.get("Health_Index","")
                with st.expander(f"{period} • Risk: {risk} • Health: {hi}", expanded=False):
                    for col in show_cols:
                        st.markdown(f"**{col}**")
                        st.write(r.get(col, "N/A"))

# ============================================================
# Watchlist (risk + trend + reasons) + Export
# ============================================================
with tab_watch:
    st.subheader("Watchlist")
    st.caption("Automated list of employees to review: low health, high risk, or deteriorating trends.")

    # scope controls
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        watch_scope = st.selectbox("Scope", ["Org-wide","Dashboard scope"], key="watch_scope")
    with c2:
        watch_top = st.selectbox("Show", [25,50,100,200], index=1, key="watch_top")
    with c3:
        watch_win = st.selectbox("Trend window", ["last_3_months","last_6_months"], index=0, key="watch_win")

    watch_base = df_full.copy()
    if watch_scope == "Dashboard scope":
        # match dashboard scope
        watch_base = df_full[(df_full["Year"] == dash_year) & (df_full["Month"] == dash_month)].copy()
        if dash_dept != "All":
            watch_base = watch_base[watch_base["Department"] == dash_dept]
        if dash_mgr != "All":
            watch_base = watch_base[watch_base["Reporting Manager"] == dash_mgr]

    # latest row per employee for scoring
    latest_per_emp = df_full.sort_values(["Name","PeriodDate"]).groupby("Name", as_index=False).tail(1).copy()

    # trend deltas (last 3/6 months)
    tw_df = apply_time_window(df_full, watch_win, LATEST_YEAR, LATEST_MONTH).copy()
    tw_df = tw_df.dropna(subset=["PeriodDate"]).sort_values(["Name","PeriodDate"])

    # compute delta from first to last within window
    first = tw_df.groupby("Name", as_index=False).first()[["Name","Health_Index"]].rename(columns={"Health_Index":"Health_Start"})
    last = tw_df.groupby("Name", as_index=False).last()[["Name","Health_Index","Risk_Level","Department","Reporting Manager"]].rename(columns={"Health_Index":"Health_End"})

    w = last.merge(first, on="Name", how="left")
    w["Health_Change"] = (w["Health_End"] - w["Health_Start"]).round(1)

    # add mood and sat for latest
    last_cols = ["Name","Sat_Score","Mood_Score","How are you feeling overall this month?","Goal_Score","Health_Index","Risk_Level"]
    last_snapshot = latest_per_emp[[c for c in last_cols if c in latest_per_emp.columns]].copy()
    w = w.merge(last_snapshot, on="Name", how="left", suffixes=("","_Latest"))

    # reasons
    def reasons(r):
        out = []
        hi = safe_number(r.get("Health_End"), 50) or 50
        ch = safe_number(r.get("Health_Change"), 0) or 0
        risk = str(r.get("Risk_Level",""))
        mood = str(r.get("How are you feeling overall this month?","")).lower()
        sat = safe_number(r.get("Sat_Score"), 5) or 5

        if risk == "Red": out.append("Red risk")
        if hi < 55: out.append("Low health")
        if ch <= -10: out.append(f"Decline {ch}")
        if "burned" in mood or "burnt" in mood: out.append("Burnout mood")
        if sat <= 3: out.append("Low satisfaction")
        return ", ".join(out) if out else "Review"

    w["Reasons"] = w.apply(reasons, axis=1)

    # rank for watchlist
    risk_rank = {"Red":0,"Amber":1,"Green":2}
    w["_rk"] = w["Risk_Level"].map(risk_rank).fillna(9)
    w = w.sort_values(["_rk","Health_End","Health_Change"], ascending=[True, True, True]).drop(columns=["_rk"])

    cols = [c for c in ["Name","Department","Reporting Manager","Risk_Level","Health_End","Health_Change","Sat_Score","Mood_Score","Reasons"] if c in w.columns]
    show = w[cols].head(int(watch_top)).copy()

    st.dataframe(show, use_container_width=True, hide_index=True)

    # export
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download Watchlist CSV", data=csv, file_name="watchlist.csv", mime="text/csv")

# ============================================================
# People AI Copilot (full dataset) + Pin to Saved Insights
# ============================================================
with tab_copilot:
    st.subheader("People AI Copilot")
    st.caption("Copilot behaves like ChatGPT: it answers in text by default and only draws charts when useful (or when you ask).")

    # memory
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []
    if "saved_insights" not in st.session_state:
        st.session_state.saved_insights = []
    if "last_copilot" not in st.session_state:
        st.session_state.last_copilot = None

    # quick prompts
    c1, c2, c3, c4 = st.columns(4)
    quick = None
    if c1.button("Org risks this month", use_container_width=True, key="qp1"):
        quick = "What are the biggest organizational risks this month? Identify hotspots and drivers."
    if c2.button("Department comparison", use_container_width=True, key="qp2"):
        quick = "Compare departments on Health Index and Satisfaction for the last 6 months. Highlight outliers."
    if c3.button("Manager hotspots", use_container_width=True, key="qp3"):
        quick = "Which managers show the highest risk signals over the last 3 months? Explain why."
    if c4.button("Employee deep-dive", use_container_width=True, key="qp4"):
        quick = "Identify employees with declining Satisfaction or Mood in the last 3 months and summarize patterns."

    # defaults (used when user says "this month")
    default_year = LATEST_YEAR
    default_month = LATEST_MONTH

    # Chat UI
    chat_area = st.container(height=680, border=True)
    with chat_area:
        for m in st.session_state.copilot_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    user_q = st.chat_input("Ask People AI… (uses full dataset)", key="copilot_input")
    if quick and not user_q:
        user_q = quick

    if user_q:
        st.session_state.copilot_messages.append({"role": "user", "content": user_q})
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_q)

        with chat_area:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)

                    # small, high-signal dataset context
                    min_p = df_full["PeriodDate"].min() if "PeriodDate" in df_full.columns else None
                    max_p = df_full["PeriodDate"].max() if "PeriodDate" in df_full.columns else None

                    # Keep lists short to avoid token bloat
                    dept_list = sorted(df_full["Department"].dropna().unique().tolist())[:25] if "Department" in df_full.columns else []
                    mgr_list = sorted(df_full["Reporting Manager"].dropna().unique().tolist())[:25] if "Reporting Manager" in df_full.columns else []

                    # dynamic allowed columns for safer charting
                    safe_cols = [c for c in df_full.columns if c in ["_PeriodLabel","PeriodDate","Department","Reporting Manager","Name","Risk_Level","Health_Index","Sat_Score","Mood_Score"]]

                    # ======================================================
                    # IMPORTANT: New Copilot contract (ChatGPT-like)
                    # - default: answer mode (Markdown)
                    # - chart only if asked OR clearly better
                    # - table for "who/which employees" style queries
                    # ======================================================
                    SYSTEM = f"""
You are the People AI Copilot for executive leadership.

Behave like ChatGPT/Gemini:
- Answer in clear executive Markdown by default.
- Only propose a chart when it materially helps (trends, comparisons, distributions) OR the user explicitly asks to plot/chart/graph/show trend.
- When the user asks "who/which employees/managers", prefer a TABLE (names + key fields) over charts.

If the request is ambiguous, ask ONE clarifying question and stop.

You must return EXACTLY ONE of the following JSON objects (and nothing else):

1) Clarify:
{{
  "mode": "clarify",
  "question": "<one short clarifying question>"
}}

2) Answer (Markdown):
{{
  "mode": "answer",
  "answer": "<executive Markdown>"
}}

3) Table:
{{
  "mode": "table",
  "title": "<short title>",
  "time_window": "this_month"|"last_3_months"|"last_6_months"|"last_12_months"|"all",
  "filter": {{"Department": "<optional>", "Reporting Manager": "<optional>", "Name": "<optional>"}},
  "columns": ["Name","Department","Reporting Manager","Risk_Level","Health_Index","Sat_Score","Mood_Score","Goal Progress"],
  "rows": [{{"Name":"...","Department":"...","Reporting Manager":"...","Risk_Level":"...","Reason":"..."}}],
  "summary": "<executive insight>"
}}

4) Chart:
{{
  "mode": "chart",
  "chart_type": "line"|"bar"|"pie"|"hist"|"scatter",
  "x": "<column name>",
  "y": "count" | "<numeric_column>" | "<numeric_column>_mean",
  "group_by": "<optional column name or null>",
  "time_window": "this_month"|"last_3_months"|"last_6_months"|"last_12_months"|"all",
  "filter": {{"Department": "<optional>", "Reporting Manager": "<optional>", "Name": "<optional>"}},
  "summary": "<executive insight>"
}}

Rules:
- Use full dataset unless user explicitly requests a scope.
- If user says “this month”, interpret as latest period in file: {default_month} {default_year}.
- If user asks for employees at risk, output mode="table" listing employees (not a bar chart).
- Allowed safe columns include: {safe_cols}.
"""

                    CONTEXT = f"""
Dataset summary:
- Period range: {min_p.date() if pd.notna(min_p) else "N/A"} to {max_p.date() if pd.notna(max_p) else "N/A"}
- Default "this month": {default_month} {default_year}
- Departments (sample): {dept_list}
- Managers (sample): {mgr_list}

Reminder:
- If the user didn't specify a time window for a trend/comparison, ask one clarifying question (e.g., last 3 vs last 6 months).
"""

                    # Heuristic routing hint (reduces chart-happy behavior)
                    ROUTE_HINT = f"""
Heuristic hints:
- user_explicit_chart_request = {wants_chart(user_q)}
- user_asking_for_names_list = {wants_names_list(user_q)}
"""

                    prompt = SYSTEM + "\n\n" + CONTEXT + "\n\n" + ROUTE_HINT + "\n\nUser: " + user_q

                    try:
                        raw = llm.invoke(prompt).content
                    except Exception as e:
                        raw = json.dumps({"mode":"answer","answer":f"⚠️ I hit an error calling the model: {e}"})

                    spec = extract_first_json(raw)

                    # if model output is not JSON, show it (fail-soft)
                    if not isinstance(spec, dict) or "mode" not in spec:
                        st.markdown(raw)
                        st.session_state.copilot_messages.append({"role":"assistant","content":raw})
                        st.session_state.last_copilot = {
                            "type": "text",
                            "question": user_q,
                            "summary": raw,
                            "spec": None,
                            "fig_json": None,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                    else:
                        mode = spec.get("mode")

                        # ---------------------------
                        # Clarify mode
                        # ---------------------------
                        if mode == "clarify":
                            q = spec.get("question","Could you clarify your time window or scope?")
                            st.markdown(q)
                            st.session_state.copilot_messages.append({"role":"assistant","content":q})
                            st.session_state.last_copilot = {
                                "type": "text",
                                "question": user_q,
                                "summary": q,
                                "spec": spec,
                                "fig_json": None,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }

                        # ---------------------------
                        # Answer mode
                        # ---------------------------
                        elif mode == "answer":
                            ans = (spec.get("answer") or "").strip()
                            if not ans:
                                ans = "I can help with that — could you rephrase the question?"
                            st.markdown(ans)
                            st.session_state.copilot_messages.append({"role":"assistant","content":ans})
                            st.session_state.last_copilot = {
                                "type": "text",
                                "question": user_q,
                                "summary": ans,
                                "spec": spec,
                                "fig_json": None,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }

                        # ---------------------------
                        # Table mode
                        # ---------------------------
                        elif mode == "table":
                            title = spec.get("title","Results")
                            st.markdown(f"#### {title}")

                            tw = spec.get("time_window","all")
                            table_df = apply_time_window(df_full, tw, default_year, default_month)

                            # optional exact filters (if provided)
                            f = spec.get("filter") or {}
                            for k in ["Department","Reporting Manager","Name"]:
                                if k in f and f[k] and k in table_df.columns:
                                    table_df = table_df[table_df[k].astype(str) == str(f[k])]

                            # If model provided rows explicitly, use them; else build from dataset
                            rows = spec.get("rows") or []
                            if rows:
                                out_df = pd.DataFrame(rows)
                                st.dataframe(out_df, use_container_width=True, hide_index=True)
                            else:
                                # fallback: show top risk/emphasis rows if asked for "at risk"
                                # (generic fallback; no hardcoding risk only)
                                cols = spec.get("columns") or ["Name","Department","Reporting Manager","Risk_Level","Health_Index","Sat_Score","Mood_Score","Goal Progress"]
                                cols = [c for c in cols if c in table_df.columns]
                                st.dataframe(table_df[cols].head(50), use_container_width=True, hide_index=True)

                            summary = (spec.get("summary") or "").strip()
                            if summary:
                                st.markdown(summary)

                            st.session_state.copilot_messages.append({
                                "role":"assistant",
                                "content": summary if summary else f"{title} (table)"
                            })
                            st.session_state.last_copilot = {
                                "type": "table",
                                "question": user_q,
                                "summary": summary if summary else title,
                                "spec": spec,
                                "fig_json": None,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                            }

                        # ---------------------------
                        # Chart mode
                        # ---------------------------
                        elif mode == "chart":
                            # If user did NOT ask for a chart, enforce "chart only when needed":
                            # allow charts for trends/compare/distribution, but not for "who/which employees"
                            if (not wants_chart(user_q)) and wants_names_list(user_q):
                                # override to table-like response
                                st.markdown("You’re asking for *which people* — I’ll list them instead of charting. Tell me the time window (this month / last 3 / last 6) if you want it scoped.")
                                st.session_state.copilot_messages.append({
                                    "role":"assistant",
                                    "content":"Listed people request detected; ask for time window if needed."
                                })
                            else:
                                tw = spec.get("time_window","all")
                                chart_df = apply_time_window(df_full, tw, default_year, default_month)

                                # apply optional exact filters
                                f = spec.get("filter") or {}
                                for k in ["Department","Reporting Manager","Name"]:
                                    if k in f and f[k] and k in chart_df.columns:
                                        chart_df = chart_df[chart_df[k].astype(str) == str(f[k])]

                                # sanitize spec against actual dataframe columns (fixes your invalid group_by issues)
                                clean = sanitize_chart_spec(spec, chart_df)

                                if clean.get("mode") == "answer":
                                    ans = clean.get("answer","I can answer this in text. What time window should I use?")
                                    st.markdown(ans)
                                    st.session_state.copilot_messages.append({"role":"assistant","content":ans})
                                else:
                                    try:
                                        fig = build_chart_safe(chart_df, clean)
                                        chart_key = f"copilot_chart_{len(st.session_state.copilot_messages)}_{safe_hash(json.dumps(clean, sort_keys=True))}"
                                        st.plotly_chart(fig, use_container_width=True, key=chart_key)

                                        summary = (clean.get("summary") or " ").strip()
                                        if summary:
                                            st.markdown(summary)

                                        st.session_state.copilot_messages.append({
                                            "role":"assistant",
                                            "content": summary if summary else "Chart generated."
                                        })
                                        st.session_state.last_copilot = {
                                            "type": "chart",
                                            "question": user_q,
                                            "summary": summary if summary else "Chart generated.",
                                            "spec": clean,
                                            "fig_json": fig.to_json(),
                                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                                        }
                                    except Exception as e:
                                        fallback = f"⚠️ I couldn’t render that chart. Try: (1) specify a time window (last_6_months), (2) choose x/y fields that exist. Debug: {e}"
                                        st.warning(fallback)
                                        st.session_state.copilot_messages.append({"role":"assistant","content":fallback})
                                        st.session_state.last_copilot = {
                                            "type": "text",
                                            "question": user_q,
                                            "summary": fallback,
                                            "spec": spec,
                                            "fig_json": None,
                                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                                        }
                        else:
                            # unknown mode
                            st.markdown("I’m not sure how to respond to that. Can you rephrase?")
                            st.session_state.copilot_messages.append({"role":"assistant","content":"Please rephrase your question."})

    # Pin + clear controls
    st.divider()
    col_pin, col_clear = st.columns([1, 1])

    with col_pin:
        if st.button("📌 Pin last Copilot output to Saved Insights", use_container_width=True, key="pin_last"):
            if st.session_state.get("last_copilot"):
                st.session_state.saved_insights.insert(0, st.session_state.last_copilot)
                st.success("Pinned to Saved Insights.")
            else:
                st.info("No Copilot output to pin yet.")

    with col_clear:
        if st.button("🧹 Clear Copilot conversation", use_container_width=True, key="clear_copilot"):
            st.session_state.copilot_messages = []
            st.session_state.last_copilot = None
            st.rerun()

# ============================================================
# Saved Insights (pins board + export)
# ============================================================
with tab_saved:
    st.subheader("Saved Insights")
    st.caption("Pinned Copilot insights and charts for leadership review.")

    saved = st.session_state.get("saved_insights", [])
    if not saved:
        st.info("No saved insights yet. Use the Copilot tab and click 📌 Pin.")
    else:
        # export all as JSON
        if st.button("Export Saved Insights (JSON)", key="export_saved"):
            payload = json.dumps(saved, indent=2)
            st.download_button("Download JSON", data=payload, file_name="saved_insights.json", mime="application/json", key="dl_saved")

        for i, item in enumerate(saved):
            title = f"{item.get('created_at','')} • {item.get('question','')[:80]}"
            with st.expander(title, expanded=(i==0)):
                st.markdown(f"**Question**: {item.get('question','')}")
                st.markdown("**Insight**:")
                st.markdown(item.get("summary",""))

                if item.get("type") == "chart" and item.get("fig_json"):
                    try:
                        fig = pio.from_json(item["fig_json"])
                        st.plotly_chart(fig, use_container_width=True, key=f"saved_fig_{i}_{safe_hash(item.get('question',''))}")
                    except Exception:
                        st.info("Chart could not be restored; keeping summary only.")

                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.saved_insights.pop(i)
                        st.rerun()
                with c2:
                    if st.button("Duplicate", key=f"dup_{i}"):
                        st.session_state.saved_insights.insert(i+1, dict(item))
                        st.rerun()





