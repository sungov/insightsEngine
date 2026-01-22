import os
import re
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain_openai import ChatOpenAI

# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(page_title="tsworks | People Insights", layout="wide")

st.markdown("""
<style>
/* overall spacing */
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(17,24,39,0.10); }

/* headings */
h1, h2, h3 { color: #0F2B46; }
.small-muted { color: #6B7280; font-size: 0.9rem; }

/* compact KPI tiles */
.kpi-row { margin-top: 0.25rem; margin-bottom: 0.75rem; }
.kpi-tile {
  background: #FFFFFF;
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 14px;
  padding: 10px 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.04);
}
.kpi-label { color: #6B7280; font-size: 0.78rem; margin-bottom: 4px; }
.kpi-value { color: #111827; font-size: 1.30rem; font-weight: 800; line-height: 1.1; }
.kpi-sub { color: #6B7280; font-size: 0.80rem; margin-top: 6px; }

/* section cards */
.card {
  background: #FFFFFF;
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.04);
}

/* pills */
.pill {
  display:inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
  border: 1px solid rgba(17,24,39,0.10);
  margin-right: 6px;
}
.pill-green { background:#E8F7EE; color:#166534; }
.pill-amber { background:#FFF4E5; color:#92400E; }
.pill-red { background:#FDECEC; color:#991B1B; }
.pill-blue { background:#EAF2FF; color:#1D4ED8; }
.pill-gray { background:#F3F4F6; color:#374151; }

/* chat container */
.chat-shell {
  background: #FFFFFF;
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 16px;
  padding: 10px 10px;
  box-shadow: 0 10px 20px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS + COLUMN NAMES
# =========================================================
APP_TITLE = "tsworks People Insights"
APP_SUBTITLE = "Senior Management View — Org → Department → Manager → Employee"

SAT_MAP = {
    "Extremely satisfied": 10, "Satisfied": 8, "Somewhat satisfied": 7,
    "Neutral": 5, "Somewhat dissatisfied": 3, "Dissatisfied": 2, "Extremely dissatisfied": 0
}
MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

MONTH_ORDER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
MONTHS_CANON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

COL_SAT_TEXT = "How satisfied are you working at tsworks?"
COL_MOOD_TEXT = "How are you feeling overall this month?"
COL_ACCOMPLISH = "Key Accomplishments this Month"
COL_DISAPPOINT = "What’s not going well or causing disappointment?"
COL_GOAL = "Goal Progress"
COL_WORKLOAD = "How is your current workload?"
COL_WLB = "How is your work-life balance?"

REQUIRED_COLS = ["Year", "Month", "Department", "Reporting Manager", "Name"]


# =========================================================
# HELPERS
# =========================================================
def normalize_month(x):
    return str(x).strip()[:3].title()

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def add_period_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Month"] = d["Month"].astype(str).map(normalize_month)
    d["_MonthNum"] = d["Month"].map(MONTH_ORDER)
    d["_PeriodKey"] = d["Year"] * 100 + d["_MonthNum"]
    d["PeriodDate"] = pd.to_datetime(
        d["Year"].astype("Int64").astype(str) + "-" + d["_MonthNum"].astype("Int64").astype(str) + "-01",
        errors="coerce"
    )
    d["_PeriodLabel"] = d["Month"] + " " + d["Year"].astype("Int64").astype(str)
    return d

def compute_nps(df_month):
    total = len(df_month)
    if total == 0:
        return 0
    promoters = len(df_month[df_month["Sat_Score"] >= 9])
    detractors = len(df_month[df_month["Sat_Score"] <= 6])
    return round(((promoters - detractors) / total) * 100)

def pill(text, kind="gray"):
    cls = {
        "green":"pill pill-green", "amber":"pill pill-amber", "red":"pill pill-red",
        "blue":"pill pill-blue", "gray":"pill pill-gray"
    }.get(kind, "pill pill-gray")
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)

def kpi_tile(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-tile">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

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

def apply_time_window(df: pd.DataFrame, window: str, anchor: pd.Timestamp):
    d = df.dropna(subset=["PeriodDate"]).copy()
    w = (window or "last_6_months").lower().strip()
    if w == "all":
        return d
    if w == "this_month":
        return d[d["PeriodDate"] == anchor].copy()
    months = {"last_3_months": 3, "last_6_months": 6, "last_12_months": 12}.get(w, 6)
    start = anchor - pd.DateOffset(months=months-1)
    return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= anchor)].copy()

def parse_goal_score(goal_text: str) -> float:
    """Heuristic parser for goal progress to 0..10."""
    s = str(goal_text).strip().lower()
    if s in ["n/a", "na", "none", ""]:
        return 5.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", s)
    if m:
        return float(np.clip(float(m.group(1)), 0, 10))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(np.clip(float(m.group(1))/10.0, 0, 10))
    if "on track" in s or "good" in s or "progress" in s:
        return 7.5
    if "at risk" in s or "delayed" in s or "behind" in s:
        return 4.0
    if "blocked" in s or "stuck" in s:
        return 3.0
    return 5.0

def risk_bucket(burnout_index: float) -> str:
    if burnout_index >= 13: return "Critical"
    if burnout_index >= 8:  return "Watchlist"
    return "Healthy"

def build_chart(chart_df: pd.DataFrame, spec: dict):
    """
    Chart builder for Copilot JSON mode.
    Supports:
      chart_type: line|bar|pie|hist
      x: _PeriodLabel|Department|Reporting Manager|Risk_Level|Month|Year
      y: Health_Index_mean|Sat_Score_mean|Mood_Score_mean|count
      group_by: optional
    """
    chart_type = (spec.get("chart_type") or "line").lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")

    allowed = set(chart_df.columns)
    if x not in allowed:
        raise ValueError(f"Invalid x column: {x}")
    if group_by and group_by not in allowed:
        raise ValueError(f"Invalid group_by column: {group_by}")

    agg = None
    y_col = y

    if isinstance(y, str):
        yl = y.lower()
        if yl in ["count", "responses", "total_responses"]:
            agg = "count"
            y_col = None
        elif y.endswith("_mean"):
            agg = "mean"
            y_col = y.replace("_mean", "")
        elif y.endswith("_avg"):
            agg = "mean"
            y_col = y.replace("_avg", "")
        elif y.endswith("_sum"):
            agg = "sum"
            y_col = y.replace("_sum", "")

    if y_col and y_col not in allowed:
        raise ValueError(f"Invalid y column: {y_col}")

    # time axis handling
    time_like = x in ["Month", "Year", "_PeriodLabel", "PeriodDate", "_PeriodKey"]

    if chart_type == "hist":
        col = y_col if (y_col and y_col in allowed) else "Health_Index"
        return px.histogram(chart_df, x=col, color=group_by, title=f"Distribution: {col}")

    if time_like:
        d = chart_df.dropna(subset=["PeriodDate"]).copy()
        group_cols = ["PeriodDate", "_PeriodLabel"]
        if group_by:
            group_cols.append(group_by)

        if agg == "count":
            plot_df = d.groupby(group_cols, as_index=False).size().rename(columns={"size":"Value"})
            y_plot = "Value"
        else:
            if agg is None:
                agg = "mean"
            plot_df = d.groupby(group_cols, as_index=False).agg({y_col: agg})
            y_plot = y_col

        plot_df = plot_df.sort_values("PeriodDate")

        if chart_type == "line":
            return px.line(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, markers=True, title="Trend")
        if chart_type == "bar":
            return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, title="Trend")
        return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, title="Trend")

    # non-time charts
    if chart_type == "pie":
        if agg == "count":
            plot_df = chart_df.groupby(x, as_index=False).size().rename(columns={"size":"Value"})
            return px.pie(plot_df, names=x, values="Value", hole=0.45, title="Share")
        if agg is None:
            agg = "mean"
        plot_df = chart_df.groupby(x, as_index=False).agg({y_col: agg})
        return px.pie(plot_df, names=x, values=y_col, hole=0.45, title="Share")

    if chart_type == "bar":
        if agg == "count":
            plot_df = chart_df.groupby([x] + ([group_by] if group_by else []), as_index=False).size()\
                              .rename(columns={"size":"Value"})
            return px.bar(plot_df, x=x, y="Value", color=group_by, title="Bar")
        if agg is None:
            agg = "mean"
        group_cols = [x] + ([group_by] if group_by else [])
        plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
        return px.bar(plot_df, x=x, y=y_col, color=group_by, title="Bar")

    # default line
    if agg is None:
        agg = "mean"
    group_cols = [x] + ([group_by] if group_by else [])
    plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
    return px.line(plot_df, x=x, y=y_col, color=group_by, markers=True, title="Line")


# =========================================================
# HEADER
# =========================================================
st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)


# =========================================================
# SIDEBAR (clean + grouped)
# =========================================================
with st.sidebar:
    st.header("Controls")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Employee Pulse (xlsx/csv)", type=["xlsx", "csv"], key="file_uploader")

    st.caption("Tip: Dashboard filters affect charts/tables. The Copilot uses the full dataset unless you ask otherwise.")


if not uploaded_file:
    st.info("Upload an Excel/CSV to start.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.fillna("N/A")

missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = add_period_cols(df_raw)
df = df.dropna(subset=["PeriodDate"]).copy()

# derived scores
df["Sat_Score"] = df[COL_SAT_TEXT].map(SAT_MAP).fillna(5) if COL_SAT_TEXT in df.columns else 5
df["Mood_Score"] = df[COL_MOOD_TEXT].map(MOOD_MAP).fillna(3) if COL_MOOD_TEXT in df.columns else 3
df["Goal_Score"] = df[COL_GOAL].apply(parse_goal_score) if COL_GOAL in df.columns else 5.0

# composite indices (0..100)
df["Health_Index"] = (0.45*df["Sat_Score"] + 0.35*df["Mood_Score"] + 0.20*df["Goal_Score"]) * 10
df["Burnout_Index"] = (10 - df["Sat_Score"]) + (5 - df["Mood_Score"]) + (6 - df["Goal_Score"]/2)
df["Risk_Level"] = df["Burnout_Index"].apply(risk_bucket)

# latest period defaults
latest_pd = df["PeriodDate"].max()
latest_row = df[df["PeriodDate"] == latest_pd].iloc[0]
latest_year = int(latest_row["Year"])
latest_month = str(latest_row["Month"])

# session state init (keeps UI stable on refresh)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sel_year" not in st.session_state:
    st.session_state.sel_year = latest_year
if "sel_month" not in st.session_state:
    st.session_state.sel_month = latest_month
if "trend_window" not in st.session_state:
    st.session_state.trend_window = "last_6_months"
if "scope_dept" not in st.session_state:
    st.session_state.scope_dept = "All Departments"
if "scope_mgr" not in st.session_state:
    st.session_state.scope_mgr = "All Managers"

# =========================================================
# SIDEBAR CONTROLS (very clear grouping)
# =========================================================
with st.sidebar:
    st.divider()

    with st.expander("📅 Period (what month are you reviewing?)", expanded=True):
        years = sorted(df["Year"].dropna().unique().astype(int).tolist())
        if st.session_state.sel_year not in years:
            st.session_state.sel_year = latest_year

        sel_year = st.selectbox(
            "Year",
            years,
            index=years.index(st.session_state["sel_year"]),
            key="sel_year"
        )
        
        df_year = df[df["Year"] == sel_year]
        months = [m for m in MONTHS_CANON if m in set(df_year["Month"].dropna())]
        if not months:
            months = [latest_month]
        
        # if current month becomes invalid after year switch, reset it
        if st.session_state.get("sel_month") not in months:
            st.session_state["sel_month"] = max(months, key=lambda m: MONTH_ORDER.get(m, 0))
        
        sel_month = st.selectbox(
            "Month",
            months,
            index=months.index(st.session_state["sel_month"]),
            key="sel_month"
        )


    with st.expander("📈 Trend Window (for trend charts)", expanded=False):
        trend_window = st.selectbox(
            "Window",
            ["this_month", "last_3_months", "last_6_months", "last_12_months", "all"],
            index=["this_month", "last_3_months", "last_6_months", "last_12_months", "all"].index(st.session_state.trend_window),
            key="trend_window"
        )

    with st.expander("🎯 Scope (applies to dashboard only)", expanded=True):
        st.caption("Use this to focus the dashboard. Copilot still uses the full dataset by default.")
        depts = sorted(df["Department"].dropna().unique().tolist())
        sel_dept = st.selectbox("Department", ["All Departments"] + depts, key="scope_dept")

        df_scope = df.copy()
        if sel_dept != "All Departments":
            df_scope = df_scope[df_scope["Department"] == sel_dept]

        mgrs = sorted(df_scope["Reporting Manager"].dropna().unique().tolist())
        sel_mgr = st.selectbox("Manager", ["All Managers"] + mgrs, key="scope_mgr")

# anchor period for charts
anchor_pd = pd.to_datetime(f"{sel_year}-{MONTH_ORDER.get(sel_month, 1)}-01")

# month slice (org)
df_month = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()

# scoped month slice (dashboard)
df_scoped_month = df_month.copy()
if sel_dept != "All Departments":
    df_scoped_month = df_scoped_month[df_scoped_month["Department"] == sel_dept]
if sel_mgr != "All Managers":
    df_scoped_month = df_scoped_month[df_scoped_month["Reporting Manager"] == sel_mgr]

# window slices
df_window = apply_time_window(df, trend_window, anchor_pd)
df_scoped_window = df_window.copy()
if sel_dept != "All Departments":
    df_scoped_window = df_scoped_window[df_scoped_window["Department"] == sel_dept]
if sel_mgr != "All Managers":
    df_scoped_window = df_scoped_window[df_scoped_window["Reporting Manager"] == sel_mgr]

# =========================================================
# TOP BAR CONTEXT (clean + quick)
# =========================================================
left, right = st.columns([2, 1])
with left:
    st.caption(f"Reviewing: **{sel_month} {sel_year}**  |  Trend window: **{trend_window.replace('_',' ')}**")
with right:
    if sel_dept == "All Departments" and sel_mgr == "All Managers":
        pill("Scope: Org-wide", "blue")
    elif sel_dept != "All Departments" and sel_mgr == "All Managers":
        pill(f"Scope: {sel_dept}", "blue")
    else:
        pill(f"Scope: {sel_dept} / {sel_mgr}", "blue")


# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "🏠 Executive Snapshot",
    "🏢 Department",
    "👨‍💼 Manager",
    "👤 Employee 360°",
    "📊 Trends",
    "🤖 People AI Copilot",
    "📂 Data Explorer",
])


# =========================================================
# TAB 1: EXECUTIVE SNAPSHOT (compact KPIs)
# =========================================================
with tabs[0]:
    st.markdown("### Executive Snapshot")

    base = df_scoped_month  # dashboard scope

    if base.empty:
        st.warning("No data for current period/scope selection.")
    else:
        avg_health = base["Health_Index"].mean()
        avg_sat = base["Sat_Score"].mean()
        avg_mood = base["Mood_Score"].mean()
        avg_goal = base["Goal_Score"].mean()
        nps = compute_nps(base)
        headcount = base["Name"].nunique()
        responses = len(base)
        critical = base[base["Risk_Level"] == "Critical"]["Name"].nunique()
        watch = base[base["Risk_Level"] == "Watchlist"]["Name"].nunique()

        # compact KPI tiles in one row
        st.markdown("<div class='kpi-row'></div>", unsafe_allow_html=True)
        c = st.columns(7)
        with c[0]: kpi_tile("Health Index", f"{avg_health:.1f}", "0–100 composite")
        with c[1]: kpi_tile("Employee NPS", f"{nps}", "Promoters vs Detractors")
        with c[2]: kpi_tile("Satisfaction", f"{avg_sat:.1f}/10", "Avg this month")
        with c[3]: kpi_tile("Mood", f"{avg_mood:.1f}/5", "Avg this month")
        with c[4]: kpi_tile("Goal Progress", f"{avg_goal:.1f}/10", "Derived / reported")
        with c[5]: kpi_tile("Critical", f"{critical}", "Needs action")
        with c[6]: kpi_tile("Watchlist", f"{watch}", "Monitor closely")

        st.divider()

        # org trend + dept health
        tr = df_scoped_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Responses=("Name","count")
        ).sort_values("PeriodDate")

        col1, col2 = st.columns(2)
        fig1 = px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
        fig2 = px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
        col1.plotly_chart(fig1, use_container_width=True, key="exec_health_trend")
        col2.plotly_chart(fig2, use_container_width=True, key="exec_sat_trend")

        st.divider()

        dept = base.groupby("Department", as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Headcount=("Name","nunique")
        ).sort_values("Health", ascending=True)

        if not dept.empty:
            figd = px.bar(dept, x="Department", y="Health", color="Health",
                          color_continuous_scale="RdYlGn", title="Department Health (Current Month)")
            st.plotly_chart(figd, use_container_width=True, key="exec_dept_health")

        st.divider()

        wl = base[base["Risk_Level"].isin(["Critical","Watchlist"])].copy()
        if wl.empty:
            pill("No critical/watchlist employees in this scope/month", "green")
        else:
            pill(f"Critical: {critical}", "red")
            pill(f"Watchlist: {watch}", "amber")
            show = safe_cols(wl, ["Name","Department","Reporting Manager","Health_Index","Sat_Score","Mood_Score","Goal_Score","Risk_Level",
                                 COL_WORKLOAD, COL_WLB])
            st.dataframe(wl[show].sort_values(["Risk_Level","Health_Index"]), use_container_width=True, hide_index=True)


# =========================================================
# TAB 2: DEPARTMENT
# =========================================================
with tabs[1]:
    st.markdown("### Department View")

    depts_all = sorted(df["Department"].dropna().unique().tolist())
    default_idx = depts_all.index(sel_dept) if sel_dept in depts_all else 0
    dept_sel = st.selectbox("Select Department", depts_all, index=default_idx, key="dept_tab_select")

    d_month = df_month[df_month["Department"] == dept_sel].copy()
    d_win = df_window[df_window["Department"] == dept_sel].copy()

    if d_month.empty:
        st.warning("No data for this department in selected month.")
    else:
        c = st.columns(6)
        with c[0]: kpi_tile("Health Index", f"{d_month['Health_Index'].mean():.1f}", "Dept avg")
        with c[1]: kpi_tile("NPS", f"{compute_nps(d_month)}", "Dept NPS")
        with c[2]: kpi_tile("Satisfaction", f"{d_month['Sat_Score'].mean():.1f}/10", "Avg")
        with c[3]: kpi_tile("Mood", f"{d_month['Mood_Score'].mean():.1f}/5", "Avg")
        with c[4]: kpi_tile("Headcount", f"{d_month['Name'].nunique()}", "Unique employees")
        with c[5]: kpi_tile("Critical", f"{d_month[d_month['Risk_Level']=='Critical']['Name'].nunique()}", "This month")

        st.divider()

        mgr = d_month.groupby("Reporting Manager", as_index=False).agg(
            Health=("Health_Index","mean"), Headcount=("Name","nunique")
        ).sort_values("Health")

        figm = px.bar(mgr, x="Reporting Manager", y="Health", color="Health",
                      color_continuous_scale="RdYlGn", title="Manager Health (within department)")
        st.plotly_chart(figm, use_container_width=True, key="dept_mgr_health")

        st.divider()

        tr = d_win.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"), Sat=("Sat_Score","mean"), Mood=("Mood_Score","mean")
        ).sort_values("PeriodDate")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Dept Health Trend"),
                          use_container_width=True, key="dept_health_trend")
        col2.plotly_chart(px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Dept Satisfaction Trend"),
                          use_container_width=True, key="dept_sat_trend")


# =========================================================
# TAB 3: MANAGER
# =========================================================
with tabs[2]:
    st.markdown("### Manager View")

    mgrs_all = sorted(df["Reporting Manager"].dropna().unique().tolist())
    default_idx = mgrs_all.index(sel_mgr) if sel_mgr in mgrs_all else 0
    mgr_sel = st.selectbox("Select Manager", mgrs_all, index=default_idx, key="mgr_tab_select")

    m_month = df_month[df_month["Reporting Manager"] == mgr_sel].copy()
    m_win = df_window[df_window["Reporting Manager"] == mgr_sel].copy()

    if m_month.empty:
        st.warning("No data for this manager in selected month.")
    else:
        c = st.columns(6)
        with c[0]: kpi_tile("Health Index", f"{m_month['Health_Index'].mean():.1f}", "Team avg")
        with c[1]: kpi_tile("NPS", f"{compute_nps(m_month)}", "Team NPS")
        with c[2]: kpi_tile("Satisfaction", f"{m_month['Sat_Score'].mean():.1f}/10", "Avg")
        with c[3]: kpi_tile("Mood", f"{m_month['Mood_Score'].mean():.1f}/5", "Avg")
        with c[4]: kpi_tile("Team Size", f"{m_month['Name'].nunique()}", "Unique employees")
        with c[5]: kpi_tile("Critical", f"{m_month[m_month['Risk_Level']=='Critical']['Name'].nunique()}", "This month")

        st.divider()

        # risk distribution
        fig = px.histogram(m_month, x="Risk_Level", title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True, key="mgr_risk_dist")

        st.divider()

        tr = m_win.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"), Sat=("Sat_Score","mean")
        ).sort_values("PeriodDate")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Team Health Trend"),
                          use_container_width=True, key="mgr_health_trend")
        col2.plotly_chart(px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Team Satisfaction Trend"),
                          use_container_width=True, key="mgr_sat_trend")


# =========================================================
# TAB 4: EMPLOYEE 360°
# =========================================================
with tabs[3]:
    st.markdown("### Employee 360°")

    all_emps = sorted(df["Name"].dropna().unique().tolist())
    search = st.text_input("Search employee", placeholder="Type a name…", key="emp_search").strip()
    emps = [e for e in all_emps if search.lower() in str(e).lower()] if search else all_emps

    emp_sel = st.selectbox("Select Employee", ["-- Select --"] + emps, key="emp_select")

    if emp_sel != "-- Select --":
        emp = df[df["Name"] == emp_sel].copy().sort_values("_PeriodKey")
        latest = emp.iloc[-1]

        c = st.columns(4)
        c[0].metric("Department", str(latest.get("Department","")))
        c[1].metric("Manager", str(latest.get("Reporting Manager","")))
        c[2].metric("Latest Period", str(latest.get("_PeriodLabel","")))
        c[3].metric("Risk Level", str(latest.get("Risk_Level","")))

        tr = emp.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Goal=("Goal_Score","mean"),
        ).sort_values("PeriodDate")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend"),
                          use_container_width=True, key="emp_health_trend")
        col2.plotly_chart(px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend"),
                          use_container_width=True, key="emp_sat_trend")

        st.divider()
        st.markdown("#### Latest Responses")
        cols = safe_cols(emp, [
            "_PeriodLabel", COL_SAT_TEXT, COL_MOOD_TEXT, COL_GOAL,
            COL_ACCOMPLISH, COL_DISAPPOINT, COL_WORKLOAD, COL_WLB
        ])
        st.dataframe(emp.sort_values("_PeriodKey", ascending=False)[cols], use_container_width=True, hide_index=True)


# =========================================================
# TAB 5: TRENDS
# =========================================================
with tabs[4]:
    st.markdown("### Trends & Benchmarking")

    league = df_month.groupby("Department", as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Headcount=("Name","nunique"),
    ).sort_values("Health", ascending=False)

    if league.empty:
        st.warning("No data for this month.")
    else:
        st.dataframe(league, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.box(df_month, x="Department", y="Sat_Score", title="Satisfaction Distribution (Current Month)"),
                          use_container_width=True, key="trend_sat_box")
        risk_tbl = df_month.groupby(["Department","Risk_Level"], as_index=False).agg(Employees=("Name","nunique"))
        col2.plotly_chart(px.bar(risk_tbl, x="Department", y="Employees", color="Risk_Level", title="Risk Counts by Department"),
                          use_container_width=True, key="trend_risk_bar")

        tr = df_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(Health=("Health_Index","mean")).sort_values("PeriodDate")
        st.plotly_chart(px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Organization Health Trend (Window)"),
                        use_container_width=True, key="trend_org_health")


# =========================================================
# TAB 6: PEOPLE AI COPILOT (BIG, PROFESSIONAL)
# =========================================================
with tabs[5]:
    st.markdown("### People AI Copilot")
    st.caption("Copilot uses the **full dataset** by default. Ask for scoped analysis if needed (e.g., “only for Department X”).")

    # suggested prompts
    sp1, sp2, sp3, sp4 = st.columns(4)
    if sp1.button("Org risks this month", key="sp_org_risks"):
        st.session_state._prefill = "Summarize organization risks for the latest month. Identify top 5 critical/watchlist employees and likely drivers."
    if sp2.button("Department comparison", key="sp_dept_comp"):
        st.session_state._prefill = "Compare departments on Health Index and Satisfaction for the last 6 months. Highlight outliers."
    if sp3.button("Manager hotspots", key="sp_mgr_hotspots"):
        st.session_state._prefill = "Which managers have the highest proportion of watchlist/critical employees over the last 6 months?"
    if sp4.button("Employee deep-dive", key="sp_emp_dd"):
        st.session_state._prefill = "For employee <name>, summarize trend in satisfaction/mood/goals and provide recommended actions."

    st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)

    # show messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input (prefill if a suggestion clicked)
    prefill = st.session_state.pop("_prefill", "")
    user_q = st.chat_input("Ask People AI…", key="copilot_input")
    if (not user_q) and prefill:
        user_q = prefill
        # also show it visually
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

    if user_q:
        if not prefill:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

        llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)

        SYSTEM = """
You are a senior people analytics advisor to executive leadership.

If the question is ambiguous, ask ONE clarifying question and stop.
Otherwise answer in concise executive Markdown.

If a chart would materially help, output ONLY valid JSON (no extra text):
{
  "chart_required": true,
  "chart_type": "line" | "bar" | "pie" | "hist",
  "x": "_PeriodLabel" | "Department" | "Reporting Manager" | "Risk_Level",
  "y": "Health_Index_mean" | "Sat_Score_mean" | "Mood_Score_mean" | "count",
  "group_by": "<optional column>",
  "time_window": "this_month" | "last_3_months" | "last_6_months" | "last_12_months" | "all",
  "filter": {"Department": "<optional>", "Reporting Manager": "<optional>", "Name": "<optional>"},
  "summary": "<short executive insight>"
}
"""

        latest_anchor = df["PeriodDate"].max()
        latest_label = df[df["PeriodDate"] == latest_anchor]["_PeriodLabel"].iloc[0]
        schema = {"columns": df.columns.tolist(), "latest_period": latest_label}

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = llm.invoke([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps({"schema": schema, "question": user_q})}
                ])
                text = (resp.content or "").strip()

                spec = extract_first_json(text)
                if isinstance(spec, dict) and spec.get("chart_required"):
                    chart_df = df.copy()
                    # time window applied relative to latest month
                    chart_df = apply_time_window(chart_df, spec.get("time_window","last_6_months"), latest_anchor)

                    # optional filters inside copilot request
                    filt = spec.get("filter") or {}
                    if isinstance(filt, dict):
                        for k, v in filt.items():
                            if v and k in chart_df.columns:
                                chart_df = chart_df[chart_df[k].astype(str) == str(v)]

                    fig = build_chart(chart_df, spec)
                    st.plotly_chart(fig, use_container_width=True, key=f"copilot_chart_{len(st.session_state.messages)}")

                    summary = spec.get("summary", "")
                    if summary:
                        st.markdown(summary)
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Chart generated."})
                else:
                    st.markdown(text)
                    st.session_state.messages.append({"role": "assistant", "content": text})

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 7: DATA EXPLORER (search + pagination)
# =========================================================
with tabs[6]:
    st.markdown("### Data Explorer")
    st.caption("This table follows the Dashboard Period + Scope filters.")

    exp_df = df_scoped_month.copy().sort_values(["Department","Reporting Manager","Name"])

    search = st.text_input("Search (Name / Dept / Manager)", placeholder="Type keyword…", key="explorer_search").strip()
    if search:
        s = search.lower()
        exp_df = exp_df[
            exp_df["Name"].astype(str).str.lower().str.contains(s, na=False) |
            exp_df["Department"].astype(str).str.lower().str.contains(s, na=False) |
            exp_df["Reporting Manager"].astype(str).str.lower().str.contains(s, na=False)
        ]

    st.caption(f"Rows: {len(exp_df)}")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="exp_page_size")
    total_pages = max(1, math.ceil(len(exp_df) / page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="exp_page")

    start = (page - 1) * page_size
    end = start + page_size
    page_df = exp_df.iloc[start:end].copy()

    cols = safe_cols(page_df, [
        "_PeriodLabel", "Name", "Department", "Reporting Manager",
        "Health_Index", "Risk_Level", "Sat_Score", "Mood_Score", "Goal_Score",
        COL_SAT_TEXT, COL_MOOD_TEXT, COL_GOAL, COL_WORKLOAD, COL_WLB,
        COL_ACCOMPLISH, COL_DISAPPOINT
    ])
    st.dataframe(page_df[cols], use_container_width=True, hide_index=True)

