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
# 1) CONFIG + STYLES
# =========================================================
st.set_page_config(page_title="tsworks | People Intelligence Platform", layout="wide")

APP_TITLE = "tsworks People Intelligence Platform"
APP_SUBTITLE = "Executive pulse + performance intelligence (Organization → Department → Manager → Employee)"

st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #1F3A5F; }
.small-muted { color: #6B7280; font-size: 0.9rem; }
.section-title { font-size: 1.25rem; font-weight: 700; color: #1F3A5F; margin: 0.25rem 0 0.75rem 0; }
.kpi-wrap { display:flex; gap: 12px; flex-wrap: wrap; }
.kpi {
  background: #FFFFFF;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.06);
  border: 1px solid rgba(31,58,95,0.08);
  min-width: 220px;
  flex: 1;
}
.kpi-label { color:#6B7280; font-size: 0.82rem; margin-bottom: 4px; }
.kpi-value { color:#111827; font-size: 1.55rem; font-weight: 800; line-height: 1.2; }
.kpi-sub { color:#6B7280; font-size: 0.85rem; margin-top: 6px; }
.pill {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-right: 6px;
  border: 1px solid rgba(0,0,0,0.08);
}
.pill-green { background:#E8F5EE; color:#166534; }
.pill-amber { background:#FFF4E5; color:#92400E; }
.pill-red { background:#FDECEC; color:#991B1B; }
.pill-blue { background:#EAF2FF; color:#1D4ED8; }
.pill-gray { background:#F3F4F6; color:#374151; }
.insight {
  background: #FFFFFF;
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(17,24,39,0.08);
  box-shadow: 0 6px 16px rgba(0,0,0,0.05);
  margin-bottom: 10px;
}
.insight-title { font-weight: 800; color:#111827; margin-bottom: 6px; }
.insight-body { color:#374151; font-size: 0.95rem; }
.chat-box {
  background: #FFFFFF;
  border-radius: 14px;
  border: 1px solid rgba(17,24,39,0.10);
  padding: 10px 10px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)

# =========================================================
# 2) CONSTANTS + MAPPINGS
# =========================================================
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

# =========================================================
# 3) HELPERS
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

def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def pill(text, kind="gray"):
    cls = {
        "green":"pill pill-green",
        "amber":"pill pill-amber",
        "red":"pill pill-red",
        "blue":"pill pill-blue",
        "gray":"pill pill-gray"
    }.get(kind, "pill pill-gray")
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)

def paginate(df: pd.DataFrame, page_size: int, page: int) -> pd.DataFrame:
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy()

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

def compute_nps(df_month):
    total = len(df_month)
    if total == 0:
        return 0
    promoters = len(df_month[df_month["Sat_Score"] >= 9])
    detractors = len(df_month[df_month["Sat_Score"] <= 6])
    return round(((promoters - detractors) / total) * 100)

def parse_goal_score(goal_text: str) -> float:
    s = str(goal_text).strip().lower()
    if s in ["n/a", "na", "none", ""]:
        return 5.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", s)
    if m:
        return float(np.clip(float(m.group(1)), 0, 10))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(np.clip(float(m.group(1))/10.0, 0, 10))
    if "on track" in s or "good" in s or "progressing" in s:
        return 7.5
    if "at risk" in s or "delayed" in s or "behind" in s:
        return 4.0
    if "blocked" in s or "stuck" in s:
        return 3.0
    return 5.0

def compute_employee_risk(emp_hist: pd.DataFrame):
    if emp_hist.empty:
        return "Unknown", "No data"
    emp_hist = emp_hist.sort_values("_PeriodKey")
    latest = emp_hist.iloc[-1]
    latest_sat = float(latest.get("Sat_Score", 5))
    latest_mood = float(latest.get("Mood_Score", 3))
    latest_health = float(latest.get("Health_Index", 50))

    reasons = []
    if latest_sat <= 3:
        reasons.append("low satisfaction")
    if latest_mood <= 2:
        reasons.append("low mood")
    if latest_health <= 45:
        reasons.append("low health index")
    if reasons:
        return "High", ", ".join(reasons)

    if len(emp_hist) >= 2:
        prev = emp_hist.iloc[-2]
        prev_sat = float(prev.get("Sat_Score", 5))
        prev_mood = float(prev.get("Mood_Score", 3))
        prev_health = float(prev.get("Health_Index", 50))
        if (prev_sat - latest_sat) >= 2 or (prev_mood - latest_mood) >= 2 or (prev_health - latest_health) >= 10:
            return "Medium", "recent decline vs previous month"

    return "Low", "stable"

def apply_time_window(df: pd.DataFrame, window: str, anchor_period_date: pd.Timestamp):
    d = df.dropna(subset=["PeriodDate"]).copy()
    w = (window or "all").lower().strip()
    if w == "all":
        return d
    if w == "this_month":
        return d[d["PeriodDate"] == anchor_period_date].copy()
    months = {"last_3_months": 3, "last_6_months": 6, "last_12_months": 12}.get(w, 6)
    start = anchor_period_date - pd.DateOffset(months=months-1)
    return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= anchor_period_date)].copy()

def build_chart(chart_df: pd.DataFrame, spec: dict):
    chart_type = (spec.get("chart_type") or "line").lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")

    allowed = set(chart_df.columns)
    if x and x not in allowed:
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

    time_like = x in ["Month", "Year", "_PeriodLabel", "PeriodDate", "_PeriodKey"]

    if chart_type == "hist":
        if y_col and y_col in allowed:
            return px.histogram(chart_df, x=y_col, color=group_by)
        return px.histogram(chart_df, x="Health_Index", color=group_by)

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
            return px.line(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, markers=True)
        if chart_type == "bar":
            return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)
        return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)

    if chart_type == "pie":
        if agg == "count":
            plot_df = chart_df.groupby(x, as_index=False).size().rename(columns={"size":"Value"})
            return px.pie(plot_df, names=x, values="Value", hole=0.4)
        if agg is None:
            agg = "mean"
        plot_df = chart_df.groupby(x, as_index=False).agg({y_col: agg})
        return px.pie(plot_df, names=x, values=y_col, hole=0.4)

    if chart_type == "bar":
        if agg == "count":
            plot_df = chart_df.groupby([x] + ([group_by] if group_by else []), as_index=False).size()\
                              .rename(columns={"size":"Value"})
            return px.bar(plot_df, x=x, y="Value", color=group_by)
        if agg is None:
            agg = "mean"
        group_cols = [x] + ([group_by] if group_by else [])
        plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
        return px.bar(plot_df, x=x, y=y_col, color=group_by)

    if agg is None:
        agg = "mean"
    group_cols = [x] + ([group_by] if group_by else [])
    plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
    return px.line(plot_df, x=x, y=y_col, color=group_by, markers=True)

def derive_insights_exec(df_all: pd.DataFrame, df_window: pd.DataFrame):
    insights = []
    dept = df_window.groupby("Department", as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Responses=("Name","count")
    )
    if not dept.empty:
        worst = dept.sort_values("Health").head(1).iloc[0]
        best = dept.sort_values("Health", ascending=False).head(1).iloc[0]
        insights.append({"level":"red","title":"Department needing attention",
                         "body": f"{worst['Department']} has the lowest health score (Health {worst['Health']:.1f})."})
        insights.append({"level":"green","title":"Strongest department momentum",
                         "body": f"{best['Department']} leads on health score (Health {best['Health']:.1f})."})

    trend = df_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(Health=("Health_Index","mean"))\
                    .sort_values("PeriodDate")
    if len(trend) >= 2:
        slope = trend["Health"].iloc[-1] - trend["Health"].iloc[0]
        if slope <= -5:
            insights.append({"level":"red","title":"Organization health is declining",
                             "body": f"Health index dropped by {abs(slope):.1f} points across the window."})
        elif slope >= 5:
            insights.append({"level":"green","title":"Organization health is improving",
                             "body": f"Health index improved by {slope:.1f} points across the window."})
        else:
            insights.append({"level":"gray","title":"Organization health is stable",
                             "body": f"Health index changed by {slope:.1f} points across the window."})
    return insights


# =========================================================
# 4) SIDEBAR: DATA LOAD + GLOBAL CONTROLS
# =========================================================
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Configure it in the environment.")
        st.stop()

    uploaded_file = st.file_uploader("Upload tsworks Employee Pulse", type=["xlsx","csv"], key="uploader_main")
    st.divider()
    st.subheader("Dashboard controls")
    st.caption("These controls affect charts/tables (not the AI Copilot).")

if not uploaded_file:
    st.info("Upload an Excel/CSV to start.")
    st.stop()

# Load file
df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.fillna("N/A")

required = ["Year","Month","Department","Reporting Manager","Name"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = add_period_cols(df_raw)

df["Sat_Score"] = df[COL_SAT_TEXT].map(SAT_MAP).fillna(5) if COL_SAT_TEXT in df.columns else 5
df["Mood_Score"] = df[COL_MOOD_TEXT].map(MOOD_MAP).fillna(3) if COL_MOOD_TEXT in df.columns else 3
df["Goal_Score"] = df[COL_GOAL].apply(parse_goal_score) if COL_GOAL in df.columns else 5.0

df["Health_Index"] = (0.45*df["Sat_Score"] + 0.35*df["Mood_Score"] + 0.20*df["Goal_Score"]) * 10
df["Burnout_Index"] = (10 - df["Sat_Score"]) + (5 - df["Mood_Score"]) + (6 - df["Goal_Score"]/2)

def risk_bucket(x):
    if x >= 13: return "Critical"
    if x >= 8:  return "Watchlist"
    return "Healthy"
df["Risk_Level"] = df["Burnout_Index"].apply(risk_bucket)

df = df.dropna(subset=["PeriodDate"]).copy()

latest_pd = df["PeriodDate"].max()
latest_row = df[df["PeriodDate"] == latest_pd].iloc[0]
latest_year = int(latest_row["Year"])
latest_month = str(latest_row["Month"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = uploaded_file.name
if st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    st.session_state.messages = []  # reset only on new file

# Sidebar selectors
with st.sidebar:
    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    if "sel_year" not in st.session_state:
        st.session_state.sel_year = latest_year
    if st.session_state.sel_year not in years:
        st.session_state.sel_year = latest_year

    sel_year = st.selectbox("Year", years, index=years.index(st.session_state.sel_year), key="sel_year")

    df_year = df[df["Year"] == sel_year]
    months = [m for m in MONTHS_CANON if m in set(df_year["Month"].dropna())]
    if "sel_month" not in st.session_state:
        st.session_state.sel_month = latest_month
    if st.session_state.sel_month not in months:
        st.session_state.sel_month = max(months, key=lambda m: MONTH_ORDER[m]) if months else latest_month

    sel_month = st.selectbox("Month", months, index=months.index(st.session_state.sel_month) if months else 0, key="sel_month")

    window = st.selectbox("Trend window", ["this_month","last_3_months","last_6_months","last_12_months","all"], index=2, key="trend_window")

    st.divider()
    st.subheader("Scope (for views)")
    depts = sorted(df["Department"].dropna().unique().tolist())
    sel_dept = st.selectbox("Department", ["All Departments"] + depts, key="scope_dept")

    scope_df = df.copy()
    if sel_dept != "All Departments":
        scope_df = scope_df[scope_df["Department"] == sel_dept]

    mgrs = sorted(scope_df["Reporting Manager"].dropna().unique().tolist())
    sel_manager = st.selectbox("Manager", ["All Managers"] + mgrs, key="scope_mgr")
    if sel_manager != "All Managers":
        scope_df = scope_df[scope_df["Reporting Manager"] == sel_manager]

anchor_pd = pd.to_datetime(f"{sel_year}-{MONTH_ORDER[sel_month]}-01")
df_month = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()

df_scoped_month = df_month.copy()
if sel_dept != "All Departments":
    df_scoped_month = df_scoped_month[df_scoped_month["Department"] == sel_dept]
if sel_manager != "All Managers":
    df_scoped_month = df_scoped_month[df_scoped_month["Reporting Manager"] == sel_manager]

df_window = apply_time_window(df, window, anchor_pd)
df_scoped_window = df_window.copy()
if sel_dept != "All Departments":
    df_scoped_window = df_scoped_window[df_scoped_window["Department"] == sel_dept]
if sel_manager != "All Managers":
    df_scoped_window = df_scoped_window[df_scoped_window["Reporting Manager"] == sel_manager]

df_full = df.copy()

st.caption(f"Dashboard period: **{sel_month} {sel_year}**  |  Trend window: **{window}**  |  Scope: **{sel_dept} / {sel_manager}**")

tabs = st.tabs([
    "🏠 Executive Intelligence",
    "🏢 Department Intelligence",
    "👨‍💼 Manager Intelligence",
    "👤 Employee 360°",
    "📊 Trends & Benchmarking",
    "🤖 People AI Copilot",
    "📂 Data Explorer"
])

# =========================================================
# TAB 1: EXECUTIVE
# =========================================================
with tabs[0]:
    st.markdown("<div class='section-title'>Executive Snapshot</div>", unsafe_allow_html=True)

    org_month = df_month
    org_window = df_window

    avg_health = org_month["Health_Index"].mean() if len(org_month) else 0
    avg_sat = org_month["Sat_Score"].mean() if len(org_month) else 0
    avg_mood = org_month["Mood_Score"].mean() if len(org_month) else 0
    nps = compute_nps(org_month)
    risk_critical = org_month[org_month["Risk_Level"] == "Critical"]["Name"].nunique()
    resp_count = len(org_month)
    headcount = org_month["Name"].nunique()

    st.markdown("<div class='kpi-wrap'>", unsafe_allow_html=True)
    kpi_card("Avg Health Index", f"{avg_health:.1f}", "0–100 composite of Satisfaction, Mood, Goals")
    kpi_card("Employee NPS", f"{nps}", "Promoters ≥9 vs Detractors ≤6")
    kpi_card("Avg Satisfaction", f"{avg_sat:.1f}/10", "Current month average")
    kpi_card("Avg Mood", f"{avg_mood:.1f}/5", "Current month average")
    kpi_card("Critical Risk", f"{risk_critical}", "Employees flagged this month")
    kpi_card("Responses", f"{resp_count}", f"Unique employees: {headcount}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='section-title'>Organization Trends</div>", unsafe_allow_html=True)
    trend = org_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Responses=("Name","count")
    ).sort_values("PeriodDate")

    c1, c2 = st.columns(2)
    fig1 = px.line(trend, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
    fig2 = px.line(trend, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
    c1.plotly_chart(fig1, use_container_width=True, key="exec_org_health_trend")
    c2.plotly_chart(fig2, use_container_width=True, key="exec_org_sat_trend")

    st.divider()

    st.markdown("<div class='section-title'>Department Health (Current Month)</div>", unsafe_allow_html=True)
    dept = org_month.groupby("Department", as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Responses=("Name","count")
    ).sort_values("Health")

    if dept.empty:
        st.warning("No department data for selected month.")
    else:
        fig = px.bar(dept, x="Department", y="Health", color="Health", color_continuous_scale="RdYlGn",
                     title="Department Health Index (higher is better)")
        st.plotly_chart(fig, use_container_width=True, key="exec_dept_health_bar")

    st.divider()

    st.markdown("<div class='section-title'>Leadership Insights</div>", unsafe_allow_html=True)
    insights = derive_insights_exec(df, org_window)
    for ins in insights:
        level = ins["level"]
        title = ins["title"]
        body = ins["body"]
        icon = {"red":"🔴","amber":"🟠","green":"🟢","blue":"🔵","gray":"⚪"}.get(level,"⚪")
        st.markdown(f"""
        <div class="insight">
          <div class="insight-title">{icon} {title}</div>
          <div class="insight-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Watchlist (Current Month)</div>", unsafe_allow_html=True)
    watch = org_month[(org_month["Risk_Level"].isin(["Critical","Watchlist"]))].copy()
    if watch.empty:
        pill("No watchlist employees this month", "green")
    else:
        pill(f"Critical: {watch[watch['Risk_Level']=='Critical']['Name'].nunique()}", "red")
        pill(f"Watchlist: {watch[watch['Risk_Level']=='Watchlist']['Name'].nunique()}", "amber")
        cols = safe_cols(watch, ["Name","Department","Reporting Manager","_PeriodLabel","Health_Index","Sat_Score","Mood_Score","Risk_Level",COL_WORKLOAD,COL_WLB])
        st.dataframe(watch[cols].sort_values(["Risk_Level","Health_Index"]), use_container_width=True, hide_index=True)

# =========================================================
# TAB 2: DEPARTMENT
# =========================================================
with tabs[1]:
    st.markdown("<div class='section-title'>Department Intelligence</div>", unsafe_allow_html=True)

    depts_all = sorted(df["Department"].dropna().unique().tolist())
    default_dept_idx = (depts_all.index(sel_dept) if sel_dept in depts_all else 0)
    dept_sel = st.selectbox("Select Department", depts_all, index=default_dept_idx, key="dept_tab_select")

    d_month = df_month[df_month["Department"] == dept_sel].copy()
    d_window = df_window[df_window["Department"] == dept_sel].copy()

    if d_month.empty:
        st.warning("No records for this department in the selected month.")
    else:
        avg_health = d_month["Health_Index"].mean()
        avg_sat = d_month["Sat_Score"].mean()
        avg_mood = d_month["Mood_Score"].mean()
        nps = compute_nps(d_month)
        critical = d_month[d_month["Risk_Level"]=="Critical"]["Name"].nunique()
        headcount = d_month["Name"].nunique()

        st.markdown("<div class='kpi-wrap'>", unsafe_allow_html=True)
        kpi_card("Dept Health Index", f"{avg_health:.1f}", f"Headcount: {headcount}")
        kpi_card("Dept NPS", f"{nps}", "Current month")
        kpi_card("Avg Satisfaction", f"{avg_sat:.1f}/10", "Current month")
        kpi_card("Avg Mood", f"{avg_mood:.1f}/5", "Current month")
        kpi_card("Critical Risk", f"{critical}", "Current month")
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        mgr = d_month.groupby("Reporting Manager", as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Responses=("Name","count")
        ).sort_values("Health")

        fig_mgr = px.bar(mgr, x="Reporting Manager", y="Health", color="Health", color_continuous_scale="RdYlGn",
                         title="Manager Health Index (within department)")
        st.plotly_chart(fig_mgr, use_container_width=True, key="dept_mgr_health_bar")

        st.divider()

        dt = d_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Responses=("Name","count")
        ).sort_values("PeriodDate")

        c1, c2 = st.columns(2)
        fig_d1 = px.line(dt, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
        fig_d2 = px.line(dt, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
        c1.plotly_chart(fig_d1, use_container_width=True, key="dept_health_trend")
        c2.plotly_chart(fig_d2, use_container_width=True, key="dept_sat_trend")

# =========================================================
# TAB 3: MANAGER
# =========================================================
with tabs[2]:
    st.markdown("<div class='section-title'>Manager Intelligence</div>", unsafe_allow_html=True)

    mgrs_all = sorted(df["Reporting Manager"].dropna().unique().tolist())
    default_mgr_idx = (mgrs_all.index(sel_manager) if sel_manager in mgrs_all else 0)
    mgr_sel = st.selectbox("Select Manager", mgrs_all, index=default_mgr_idx, key="mgr_tab_select")

    m_month = df_month[df_month["Reporting Manager"] == mgr_sel].copy()
    m_window = df_window[df_window["Reporting Manager"] == mgr_sel].copy()

    if m_month.empty:
        st.warning("No records for this manager in the selected month.")
    else:
        avg_health = m_month["Health_Index"].mean()
        avg_sat = m_month["Sat_Score"].mean()
        avg_mood = m_month["Mood_Score"].mean()
        nps = compute_nps(m_month)
        critical = m_month[m_month["Risk_Level"]=="Critical"]["Name"].nunique()
        headcount = m_month["Name"].nunique()

        st.markdown("<div class='kpi-wrap'>", unsafe_allow_html=True)
        kpi_card("Team Health Index", f"{avg_health:.1f}", f"Team size: {headcount}")
        kpi_card("Team NPS", f"{nps}", "Current month")
        kpi_card("Avg Satisfaction", f"{avg_sat:.1f}/10", "Current month")
        kpi_card("Avg Mood", f"{avg_mood:.1f}/5", "Current month")
        kpi_card("Critical Risk", f"{critical}", "Current month")
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        fig_hist = px.histogram(m_month, x="Risk_Level", title="Risk distribution")
        st.plotly_chart(fig_hist, use_container_width=True, key="mgr_risk_hist")

        st.divider()

        mt = m_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean"),
            Responses=("Name","count")
        ).sort_values("PeriodDate")

        c1, c2 = st.columns(2)
        fig_m1 = px.line(mt, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
        fig_m2 = px.line(mt, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
        c1.plotly_chart(fig_m1, use_container_width=True, key="mgr_health_trend")
        c2.plotly_chart(fig_m2, use_container_width=True, key="mgr_sat_trend")

# =========================================================
# TAB 4: EMPLOYEE 360°
# =========================================================
with tabs[3]:
    st.markdown("<div class='section-title'>Employee 360°</div>", unsafe_allow_html=True)

    all_emps = sorted(df["Name"].dropna().unique().tolist())
    search = st.text_input("Search employee", placeholder="Type a name…", key="emp360_search").strip()
    emps_filtered = [e for e in all_emps if search.lower() in str(e).lower()] if search else all_emps

    emp_sel = st.selectbox("Select employee", ["-- Select --"] + emps_filtered, key="emp360_select")

    if emp_sel != "-- Select --":
        emp_hist = df[df["Name"] == emp_sel].copy().sort_values("_PeriodKey")
        latest = emp_hist.iloc[-1]
        risk, reason = compute_employee_risk(emp_hist)

        c1, c2, c3, c4, c5 = st.columns([2,2,2,2,3])
        c1.metric("Employee", str(latest.get("Name","")))
        c2.metric("Department", str(latest.get("Department","")))
        c3.metric("Manager", str(latest.get("Reporting Manager","")))
        c4.metric("Latest Period", str(latest.get("_PeriodLabel","")))
        c5.metric("Risk", f"{risk} — {reason}")

        tr = emp_hist.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean"),
            Sat=("Sat_Score","mean"),
            Mood=("Mood_Score","mean")
        ).sort_values("PeriodDate")

        c1, c2 = st.columns(2)
        fig_e1 = px.line(tr, x="_PeriodLabel", y="Health", markers=True, title="Health Index Trend")
        fig_e2 = px.line(tr, x="_PeriodLabel", y="Sat", markers=True, title="Satisfaction Trend")
        c1.plotly_chart(fig_e1, use_container_width=True, key="emp_health_trend")
        c2.plotly_chart(fig_e2, use_container_width=True, key="emp_sat_trend")

# =========================================================
# TAB 5: TRENDS & BENCHMARKING
# =========================================================
with tabs[4]:
    st.markdown("<div class='section-title'>Trends & Benchmarks</div>", unsafe_allow_html=True)
    league = df_month.groupby("Department", as_index=False).agg(
        Health=("Health_Index","mean"),
        Sat=("Sat_Score","mean"),
        Mood=("Mood_Score","mean"),
        Headcount=("Name","nunique")
    ).sort_values("Health", ascending=False)

    if not league.empty:
        st.dataframe(league, use_container_width=True, hide_index=True)

        fig_box = px.box(df_month, x="Department", y="Sat_Score", title="Distribution of Satisfaction (this month)")
        st.plotly_chart(fig_box, use_container_width=True, key="bench_sat_box")

        risk_tbl = df_month.groupby(["Department","Risk_Level"], as_index=False).agg(Employees=("Name","nunique"))
        fig_risk = px.bar(risk_tbl, x="Department", y="Employees", color="Risk_Level", title="Risk counts by department")
        st.plotly_chart(fig_risk, use_container_width=True, key="bench_risk_bar")

        trend = df_window.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
            Health=("Health_Index","mean")
        ).sort_values("PeriodDate")
        fig_org = px.line(trend, x="_PeriodLabel", y="Health", markers=True, title="Org Health Trend (window)")
        st.plotly_chart(fig_org, use_container_width=True, key="bench_org_trend")

# =========================================================
# TAB 6: PEOPLE AI COPILOT (unchanged: always full dataset)
# =========================================================
with tabs[5]:
    st.markdown("<div class='section-title'>People AI Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask People AI… (Copilot uses full dataset)", key="copilot_input")
    if user_q:
        st.session_state.messages.append({"role":"user", "content":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        COPILOT_SYSTEM = """
You are a senior people analytics advisor for leadership.

If the question is ambiguous, ask ONE clarifying question first.
Otherwise answer in executive Markdown.

If a chart would help, output ONLY valid JSON using:
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

        latest_pd2 = df_full["PeriodDate"].max()
        latest_label2 = df_full[df_full["PeriodDate"] == latest_pd2]["_PeriodLabel"].iloc[0]

        schema = {"columns": df_full.columns.tolist(), "latest_period": latest_label2}
        user_context = {"schema": schema, "question": user_q}

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = llm.invoke([
                    {"role":"system","content": COPILOT_SYSTEM},
                    {"role":"user","content": json.dumps(user_context)}
                ])
                text = (resp.content or "").strip()

                spec = extract_first_json(text)
                if isinstance(spec, dict) and spec.get("chart_required"):
                    chart_df = df_full.copy()
                    anchor = chart_df["PeriodDate"].max()
                    chart_df = apply_time_window(chart_df, spec.get("time_window","all"), anchor)

                    filt = spec.get("filter") or {}
                    if isinstance(filt, dict):
                        for k, v in filt.items():
                            if v and k in chart_df.columns:
                                chart_df = chart_df[chart_df[k].astype(str) == str(v)]

                    try:
                        fig = build_chart(chart_df, spec)
                        st.plotly_chart(fig, use_container_width=True, key=f"copilot_chart_{len(st.session_state.messages)}")
                        summary = spec.get("summary","")
                        if summary:
                            st.markdown(summary)
                            st.session_state.messages.append({"role":"assistant","content": summary})
                        else:
                            st.session_state.messages.append({"role":"assistant","content":"Chart generated."})
                    except Exception as e:
                        err_msg = f"Could not build chart: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role":"assistant","content": err_msg})
                else:
                    st.markdown(text)
                    st.session_state.messages.append({"role":"assistant","content": text})

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 7: DATA EXPLORER
# =========================================================
with tabs[6]:
    st.markdown("<div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)

    exp_df = df_scoped_month.copy().sort_values(["Department","Reporting Manager","Name"])
    search = st.text_input("Search (Name / Dept / Manager)", placeholder="Type any keyword…", key="explorer_search").strip()
    if search:
        s = search.lower()
        exp_df = exp_df[
            exp_df["Name"].astype(str).str.lower().str.contains(s, na=False) |
            exp_df["Department"].astype(str).str.lower().str.contains(s, na=False) |
            exp_df["Reporting Manager"].astype(str).str.lower().str.contains(s, na=False)
        ]

    st.caption(f"Rows: {len(exp_df)}")
    page_size = st.selectbox("Rows per page", [10,25,50,100], index=1, key="exp_page_size")
    total_pages = max(1, math.ceil(len(exp_df)/page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="exp_page")

    page_df = paginate(exp_df, page_size, page)
    cols = safe_cols(page_df, ["_PeriodLabel","Name","Department","Reporting Manager","Health_Index","Risk_Level","Sat_Score","Mood_Score",
                               COL_SAT_TEXT,COL_MOOD_TEXT,COL_GOAL,COL_WORKLOAD,COL_WLB,COL_ACCOMPLISH,COL_DISAPPOINT])
    st.dataframe(page_df[cols], use_container_width=True, hide_index=True)
