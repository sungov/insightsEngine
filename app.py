import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import json
import re
import math

# ----------------------------------
# Global config / constants
# ----------------------------------
st.set_page_config(page_title="tsworks | Insights Engine", layout="wide")

SAT_MAP = {
    "Extremely satisfied": 10, "Satisfied": 8, "Somewhat satisfied": 7,
    "Neutral": 5, "Somewhat dissatisfied": 3, "Dissatisfied": 2, "Extremely dissatisfied": 0
}
MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

MONTH_ORDER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
MONTHS_CANON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ----------------------------------
# Helpers
# ----------------------------------
def extract_json_object(text: str):
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def add_period_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "Year" in d.columns:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    if "Month" in d.columns:
        d["Month"] = d["Month"].astype(str).str.strip().str[:3].str.title()
    d["_MonthNum"] = d["Month"].map(MONTH_ORDER)
    d["_PeriodKey"] = d["Year"] * 100 + d["_MonthNum"]
    d["PeriodDate"] = pd.to_datetime(
        d["Year"].astype("Int64").astype(str) + "-" + d["_MonthNum"].astype("Int64").astype(str) + "-01",
        errors="coerce"
    )
    d["_PeriodLabel"] = d["Month"] + " " + d["Year"].astype("Int64").astype(str)
    return d

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def paginate(df: pd.DataFrame, page_size: int, page: int) -> pd.DataFrame:
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy()

def apply_time_filter(d: pd.DataFrame, time_filter: str, sel_year, sel_month) -> pd.DataFrame:
    d = add_period_cols(d).dropna(subset=["PeriodDate"])
    tf = (time_filter or "all").strip().lower()

    if tf == "this_month":
        return d[(d["Year"] == sel_year) & (d["Month"] == sel_month)].copy()

    if tf == "last_6_months":
        end = d["PeriodDate"].max()
        start = end - pd.DateOffset(months=6)
        return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()

    if tf == "last_quarter":
        end = d["PeriodDate"].max()
        start = end - pd.DateOffset(months=3)
        return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()

    return d.copy()

def build_chart(chart_df: pd.DataFrame, spec: dict):
    """
    Builds charts safely + forces chronological order for time-series.
    y supports: Sat_Score_mean, Mood_Score_mean, count, <numeric>_mean, etc.
    """
    chart_df = chart_df.copy()
    chart_type = (spec.get("chart_type") or "line").lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")

    allowed_cols = set(chart_df.columns)
    if x not in allowed_cols:
        raise ValueError(f"Invalid x column: {x}")
    if group_by and group_by not in allowed_cols:
        raise ValueError(f"Invalid group_by column: {group_by}")

    agg = None
    y_col = y

    if isinstance(y, str):
        if y.lower() in ["count", "responses", "total_responses"]:
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

    if y_col and y_col not in allowed_cols:
        raise ValueError(f"Invalid y column: {y_col}")

    time_like = x in ["Month", "Year", "_PeriodLabel", "PeriodDate", "_PeriodKey"]

    if time_like:
        chart_df = add_period_cols(chart_df).dropna(subset=["PeriodDate"])

        group_cols = ["PeriodDate", "_PeriodLabel"]
        if group_by:
            group_cols.append(group_by)

        if agg == "count":
            plot_df = chart_df.groupby(group_cols, as_index=False).size().rename(columns={"size": "Value"})
            y_plot = "Value"
        else:
            if agg is None:
                agg = "mean"
            plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
            y_plot = y_col

        plot_df = plot_df.sort_values("PeriodDate")

        if chart_type == "line":
            return px.line(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, markers=True)
        if chart_type == "bar":
            return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)
        return px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)

    # Non-time charts
    if chart_type == "pie":
        if agg == "count":
            plot_df = chart_df.groupby(x, as_index=False).size().rename(columns={"size": "Value"})
            return px.pie(plot_df, names=x, values="Value", hole=0.4)
        else:
            if agg is None:
                agg = "mean"
            plot_df = chart_df.groupby(x, as_index=False).agg({y_col: agg})
            return px.pie(plot_df, names=x, values=y_col, hole=0.4)

    if chart_type == "bar":
        if agg == "count":
            plot_df = chart_df.groupby([x] + ([group_by] if group_by else []), as_index=False)\
                              .size().rename(columns={"size": "Value"})
            return px.bar(plot_df, x=x, y="Value", color=group_by)
        else:
            if agg is None:
                agg = "mean"
            group_cols = [x] + ([group_by] if group_by else [])
            plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
            return px.bar(plot_df, x=x, y=y_col, color=group_by)

    # Default line
    if agg is None:
        agg = "mean"
    group_cols = [x] + ([group_by] if group_by else [])
    plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
    return px.line(plot_df, x=x, y=y_col, color=group_by, markers=True)

def compute_employee_risk(emp_hist: pd.DataFrame):
    """
    Simple, explainable risk heuristic:
    - latest Sat <= 3 OR latest Mood <= 2 -> high risk
    - or significant drop vs last month (>=2 points) -> medium risk
    """
    if emp_hist.empty:
        return "Unknown", "No data"
    emp_hist = emp_hist.sort_values("_PeriodKey")
    latest = emp_hist.iloc[-1]
    latest_sat = float(latest.get("Sat_Score", 5))
    latest_mood = float(latest.get("Mood_Score", 3))

    reason = []
    if latest_sat <= 3:
        reason.append("low satisfaction")
    if latest_mood <= 2:
        reason.append("low mood")

    risk = "Low"
    if reason:
        risk = "High"
        return risk, ", ".join(reason)

    # drop check
    if len(emp_hist) >= 2:
        prev = emp_hist.iloc[-2]
        prev_sat = float(prev.get("Sat_Score", 5))
        prev_mood = float(prev.get("Mood_Score", 3))
        if (prev_sat - latest_sat) >= 2 or (prev_mood - latest_mood) >= 2:
            return "Medium", "recent drop vs previous month"

    return "Low", "stable"

# ----------------------------------
# Header
# ----------------------------------
st.title("📊 tsworks Insights Engine")
st.caption("Executive pulse + performance insights across Organization → Department → Employee")

# ----------------------------------
# Sidebar: Setup & Scope
# ----------------------------------
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not configured. Set OPENAI_API_KEY.")
        st.stop()

    uploaded_file = st.file_uploader("Upload 'tsworks Employee Pulse' Excel", type=["xlsx", "csv"])

if not uploaded_file:
    st.info("Upload an Excel/CSV to start.")
    st.stop()

# reset state on new file
if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    st.session_state.messages = []
    st.session_state.pop("sel_year", None)
    st.session_state.pop("sel_month", None)
    st.session_state["default_period_set"] = False

# ----------------------------------
# Load + Normalize
# ----------------------------------
df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
df.columns = df.columns.str.strip()
df = df.fillna("N/A")

# required minimal columns check
required = ["Year", "Month", "Department", "Reporting Manager", "Name"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["Sat_Score"] = df["How satisfied are you working at tsworks?"].map(SAT_MAP).fillna(5)
df["Mood_Score"] = df["How are you feeling overall this month?"].map(MOOD_MAP).fillna(3)
df = add_period_cols(df)

# ----------------------------------
# Sidebar filters: Month/Year defaults to latest
# ----------------------------------
latest_key = df["_PeriodKey"].dropna().max()
latest_row = df.loc[df["_PeriodKey"] == latest_key].iloc[0]
latest_year = int(latest_row["Year"])
latest_month = latest_row["Month"]

years = sorted(df["Year"].dropna().unique().astype(int).tolist())
if (not st.session_state.get("default_period_set", False)) or (st.session_state.get("sel_year") not in years):
    st.session_state["sel_year"] = latest_year
    st.session_state["default_period_set"] = True

sel_year = st.sidebar.selectbox("Year (Dashboard)", years, index=years.index(st.session_state["sel_year"]), key="sel_year")

df_year = df[df["Year"] == sel_year]
months = [m for m in MONTHS_CANON if m in set(df_year["Month"].dropna())]
if st.session_state.get("sel_month") not in months:
    st.session_state["sel_month"] = max(months, key=lambda m: MONTH_ORDER[m]) if months else latest_month

sel_month = st.sidebar.selectbox("Month (Dashboard)", months, index=months.index(st.session_state["sel_month"]) if months else 0, key="sel_month")

# Department + Manager scope for org/department/employee views
st.sidebar.divider()
st.sidebar.subheader("Scope (applies to Dept/Employee views)")
all_depts = sorted(df["Department"].dropna().unique().tolist())
sel_dept = st.sidebar.selectbox("Department", ["All Departments"] + all_depts)

scope_df = df.copy()
if sel_dept != "All Departments":
    scope_df = scope_df[scope_df["Department"] == sel_dept]

all_mgrs = sorted(scope_df["Reporting Manager"].dropna().unique().tolist())
sel_manager = st.sidebar.selectbox("Manager", ["All Managers"] + all_mgrs)
if sel_manager != "All Managers":
    scope_df = scope_df[scope_df["Reporting Manager"] == sel_manager]

# ----------------------------------
# Data scopes
# ----------------------------------
# Dashboard month-scoped (Year+Month + optional dept/manager scope)
df_dash = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()
if sel_dept != "All Departments":
    df_dash = df_dash[df_dash["Department"] == sel_dept]
if sel_manager != "All Managers":
    df_dash = df_dash[df_dash["Reporting Manager"] == sel_manager]

# Responses scope = all months, filtered only by dept/manager scope
df_resp_scope = scope_df.copy()

# ----------------------------------
# Main navigation tabs
# ----------------------------------
st.caption(f"Dashboard period: **{sel_month} {sel_year}**   |   Scope: **{sel_dept} / {sel_manager}**")

tab_org, tab_dept, tab_emp, tab_ai = st.tabs(["🏢 Organization", "🏬 Department", "👤 Employee", "🤖 AI Insights"])

# ----------------------------------
# ORG TAB
# ----------------------------------
with tab_org:
    st.subheader("Organization overview")

    # Month KPIs (dashboard month only)
    total = len(df[(df["Year"] == sel_year) & (df["Month"] == sel_month)])
    if total == 0:
        st.warning("No data for selected month/year.")
    else:
        df_org_month = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()
        promoters = len(df_org_month[df_org_month["Sat_Score"] >= 9])
        detractors = len(df_org_month[df_org_month["Sat_Score"] <= 6])
        nps = round(((promoters - detractors) / total) * 100)
        avg_sat = round(df_org_month["Sat_Score"].mean(), 1)
        avg_mood = round(df_org_month["Mood_Score"].mean(), 2)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses", total)
        c2.metric("Employee NPS", nps)
        c3.metric("Avg Satisfaction", f"{avg_sat}/10")
        c4.metric("Avg Mood", f"{avg_mood}/5")

        st.divider()

        # Org trend last 6 months (chronological)
        trend_df = apply_time_filter(df, "last_6_months", sel_year, sel_month)
        trend_plot = trend_df.groupby(["PeriodDate", "_PeriodLabel"], as_index=False).agg(
            Sat_Score=("Sat_Score", "mean"),
            Mood_Score=("Mood_Score", "mean")
        ).sort_values("PeriodDate")

        colA, colB = st.columns(2)
        colA.plotly_chart(px.line(trend_plot, x="_PeriodLabel", y="Sat_Score", markers=True,
                                  title="Org Satisfaction (last 6 months)"), use_container_width=True)
        colB.plotly_chart(px.line(trend_plot, x="_PeriodLabel", y="Mood_Score", markers=True,
                                  title="Org Mood (last 6 months)"), use_container_width=True)

        st.divider()

        # Department comparison for selected month
        dep_plot = df_org_month.groupby("Department", as_index=False).agg(
            Sat_Score=("Sat_Score", "mean"),
            Mood_Score=("Mood_Score", "mean"),
            Responses=("Name", "count")
        ).sort_values("Sat_Score")

        st.plotly_chart(
            px.bar(dep_plot, x="Department", y="Sat_Score", title="Department Satisfaction (this month)"),
            use_container_width=True
        )

        # Risk list (this month)
        st.markdown("### ⚠️ Watchlist (this month)")
        risk_df = df_org_month.copy()
        risk_df["RiskFlag"] = (risk_df["Sat_Score"] <= 3) | (risk_df["Mood_Score"] <= 2)
        watch = risk_df[risk_df["RiskFlag"]].copy()

        if watch.empty:
            st.success("No high-risk flags detected for this month based on satisfaction/mood thresholds.")
        else:
            cols = safe_cols(watch, ["Name","Department","Reporting Manager","Sat_Score","Mood_Score",
                                     "How is your current workload?","How is your work-life balance?"])
            st.dataframe(watch[cols].sort_values(["Sat_Score","Mood_Score"]), use_container_width=True, hide_index=True)

# ----------------------------------
# DEPARTMENT TAB
# ----------------------------------
with tab_dept:
    st.subheader("Department insights")

    if sel_dept == "All Departments":
        st.info("Select a Department in the sidebar to see deep insights.")
    else:
        dept_month = df[(df["Year"] == sel_year) & (df["Month"] == sel_month) & (df["Department"] == sel_dept)].copy()

        if dept_month.empty:
            st.warning("No records for this department in the selected month.")
        else:
            promoters = len(dept_month[dept_month["Sat_Score"] >= 9])
            detractors = len(dept_month[dept_month["Sat_Score"] <= 6])
            nps = round(((promoters - detractors) / len(dept_month)) * 100)
            avg_sat = round(dept_month["Sat_Score"].mean(), 1)
            avg_mood = round(dept_month["Mood_Score"].mean(), 2)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Responses", len(dept_month))
            c2.metric("Dept NPS", nps)
            c3.metric("Avg Satisfaction", f"{avg_sat}/10")
            c4.metric("Avg Mood", f"{avg_mood}/5")

            st.divider()

            # Manager comparison (this month)
            mgr_plot = dept_month.groupby("Reporting Manager", as_index=False).agg(
                Sat_Score=("Sat_Score", "mean"),
                Mood_Score=("Mood_Score", "mean"),
                Responses=("Name", "count")
            ).sort_values("Sat_Score")

            st.plotly_chart(px.bar(mgr_plot, x="Reporting Manager", y="Sat_Score",
                                   title="Manager-wise Satisfaction (this month)"),
                            use_container_width=True)

            st.divider()

            # Department trend (last 6 months)
            dept_trend = apply_time_filter(df[df["Department"] == sel_dept], "last_6_months", sel_year, sel_month)
            dept_trend_plot = dept_trend.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
                Sat_Score=("Sat_Score","mean"),
                Mood_Score=("Mood_Score","mean")
            ).sort_values("PeriodDate")

            colA, colB = st.columns(2)
            colA.plotly_chart(px.line(dept_trend_plot, x="_PeriodLabel", y="Sat_Score", markers=True,
                                      title="Dept Satisfaction trend (last 6 months)"),
                              use_container_width=True)
            colB.plotly_chart(px.line(dept_trend_plot, x="_PeriodLabel", y="Mood_Score", markers=True,
                                      title="Dept Mood trend (last 6 months)"),
                              use_container_width=True)

            st.divider()

            st.markdown("### 🧾 Responses (all months, scoped)")
            st.caption("This table shows all months within the selected Department/Manager scope.")
            # reuse df_resp_scope but ensure dept selected
            resp_scoped = df_resp_scope.copy()
            if sel_dept != "All Departments":
                resp_scoped = resp_scoped[resp_scoped["Department"] == sel_dept]
            if sel_manager != "All Managers":
                resp_scoped = resp_scoped[resp_scoped["Reporting Manager"] == sel_manager]

            # quick pagination here too
            search = st.text_input("Search employee name (department scope)", placeholder="Type a name...")
            if search:
                resp_scoped = resp_scoped[resp_scoped["Name"].astype(str).str.contains(search, case=False, na=False)]

            resp_scoped = resp_scoped.sort_values("_PeriodKey", ascending=False)
            page_size = st.selectbox("Rows per page", [10,25,50,100], index=1, key="dept_page_size")
            total_pages = max(1, math.ceil(len(resp_scoped)/page_size))
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="dept_page")
            page_df = paginate(resp_scoped, page_size, page)

            show_cols = safe_cols(page_df, [
                "_PeriodLabel","Name","Reporting Manager",
                "How satisfied are you working at tsworks?",
                "How are you feeling overall this month?",
                "Goal Progress","How is your current workload?","How is your work-life balance?"
            ])
            st.dataframe(page_df[show_cols], use_container_width=True, hide_index=True)

# ----------------------------------
# EMPLOYEE TAB
# ----------------------------------
with tab_emp:
    st.subheader("Employee profile")

    # Guardrail: individual employee needs manager scope (your Step 5)
    if sel_manager == "All Managers":
        st.warning("Select a **Manager** in the sidebar to view individual employee profiles.")
        st.stop()

    # Employee search + pick
    scoped_emps = sorted(df_resp_scope["Name"].dropna().unique().tolist())

    search = st.text_input("Search employee", placeholder="Start typing a name...")
    filtered_emps = [e for e in scoped_emps if search.lower() in str(e).lower()] if search else scoped_emps

    selected_emp = st.selectbox("Select employee", ["-- Select --"] + filtered_emps)

    if selected_emp == "-- Select --":
        st.info("Pick an employee to see their history and responses.")
        st.stop()

    emp_hist = df_resp_scope[df_resp_scope["Name"] == selected_emp].copy()
    emp_hist = emp_hist.sort_values("_PeriodKey")

    # Key employee KPIs
    latest = emp_hist.iloc[-1]
    latest_period = latest.get("_PeriodLabel","")
    latest_sat = float(latest.get("Sat_Score", 5))
    latest_mood = float(latest.get("Mood_Score", 3))

    risk, reason = compute_employee_risk(emp_hist)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest period", latest_period)
    c2.metric("Satisfaction", f"{round(latest_sat,1)}/10")
    c3.metric("Mood", f"{round(latest_mood,1)}/5")
    c4.metric("Risk", f"{risk} ({reason})")

    st.divider()

    # Trend charts (employee)
    trend_plot = emp_hist.groupby(["PeriodDate","_PeriodLabel"], as_index=False).agg(
        Sat_Score=("Sat_Score","mean"),
        Mood_Score=("Mood_Score","mean")
    ).sort_values("PeriodDate")

    colA, colB = st.columns(2)
    colA.plotly_chart(px.line(trend_plot, x="_PeriodLabel", y="Sat_Score", markers=True,
                              title="Satisfaction over time"), use_container_width=True)
    colB.plotly_chart(px.line(trend_plot, x="_PeriodLabel", y="Mood_Score", markers=True,
                              title="Mood over time"), use_container_width=True)

    st.divider()

    st.markdown("### 🧾 Responses history (latest first)")
    emp_hist_desc = emp_hist.sort_values("_PeriodKey", ascending=False)

    # Compact table view
    cols = safe_cols(emp_hist_desc, [
        "_PeriodLabel","How satisfied are you working at tsworks?",
        "How are you feeling overall this month?",
        "Goal Progress","How is your current workload?","How is your work-life balance?"
    ])
    st.dataframe(emp_hist_desc[cols], use_container_width=True, hide_index=True)

    st.divider()

    # Detailed card view
    question_cols = safe_cols(emp_hist_desc, [
        "Key Accomplishments this Month",
        "What’s not going well or causing disappointment?",
        "Any concerns, blockers, or risks?",
        "Do you need support from bench resources or other teams?",
        "How is your work-life balance?",
        "How is your current workload?",
        "Are you currently supporting or mentoring junior team members?",
        "Suggestions for process or workflow improvements",
        "Planned PTO this month and coverage plan",
        "Goal Progress"
    ])

    for _, r in emp_hist_desc.iterrows():
        with st.expander(f"📅 {r.get('_PeriodLabel','')}", expanded=False):
            st.write(f"**Satisfaction:** {r.get('How satisfied are you working at tsworks?', 'N/A')}")
            st.write(f"**Mood:** {r.get('How are you feeling overall this month?', 'N/A')}")
            for col in question_cols:
                st.markdown(f"**{col}**")
                st.write(r.get(col, "N/A"))

# ----------------------------------
# AI INSIGHTS TAB
# ----------------------------------
with tab_ai:
    st.subheader("AI Insights (answers + optional charts)")

    st.caption("Tip: Ask org/department questions like 'Show satisfaction trend last 6 months by department'.")

    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)

    CUSTOM_PREFIX = """
You are a Senior Manager at tsworks.

You can answer questions in Markdown.
If a chart is useful, respond with ONLY valid JSON (no extra text) using this schema:

{
  "chart_required": true,
  "chart_type": "line" | "bar" | "pie",
  "x": "Month" | "Department" | "Reporting Manager" | "Year" | "_PeriodLabel" | "<other column>",
  "y": "Sat_Score_mean" | "Mood_Score_mean" | "count" | "<numeric_column>_mean",
  "group_by": "<optional column>",
  "time_filter": "this_month" | "last_6_months" | "last_quarter" | "all",
  "summary": "<short executive insight>"
}

Rules:
- For trends, use x="Month" and y="Sat_Score_mean" or "Mood_Score_mean" (or "count").
- If the question says "this month", time_filter MUST be "this_month" and should align with the UI period.
- If chart_required is true, output ONLY JSON.
"""

    agent = create_pandas_dataframe_agent(
        llm,
        df,  # full dataset for analysis
        verbose=False,
        allow_dangerous_code=True,  # safer
        handle_parsing_errors=True,
        agent_type="openai-tools",
        prefix=CUSTOM_PREFIX
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=260)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    query = st.chat_input("Ask an insight question…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    context = f"""
UI context:
- Dashboard period: {sel_month} {sel_year}
- Scope: Department={sel_dept}, Manager={sel_manager}

Instructions:
- If user asks "this month", interpret it as the UI dashboard period.
- If user asks org-wide, use Department=All (ignore scope) unless explicitly asked.
- If user asks department/manager scoped, respect scope.
- Always state what period/scope you used.
"""
                    try:
                        result = agent.invoke({"input": context + "\n\nUser question: " + query})
                        response = (result.get("output") or "").strip()

                        chart_spec = extract_json_object(response)
                        if chart_spec and chart_spec.get("chart_required"):
                            tf = chart_spec.get("time_filter") or "all"
                            chart_df = apply_time_filter(df, tf, sel_year, sel_month)

                            # Apply scope only if user is scoped in UI and question isn't explicitly org-wide
                            if sel_dept != "All Departments":
                                chart_df = chart_df[chart_df["Department"] == sel_dept]
                            if sel_manager != "All Managers":
                                chart_df = chart_df[chart_df["Reporting Manager"] == sel_manager]

                            fig = build_chart(chart_df, chart_spec)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown(chart_spec.get("summary", ""))

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": chart_spec.get("summary", "Chart generated.")
                            })
                        else:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"Error: {e}")

