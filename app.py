import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import json
import re
import math

# -----------------------------
# Helpers
# -----------------------------
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

MONTH_ORDER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
               "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

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
    chart_type = (spec.get("chart_type") or "line").lower().strip()
    x = spec.get("x")
    y = spec.get("y")
    group_by = spec.get("group_by")

    allowed_cols = set(chart_df.columns)
    if x not in allowed_cols:
        raise ValueError(f"Invalid x column: {x}")
    if group_by and group_by not in allowed_cols:
        raise ValueError(f"Invalid group_by column: {group_by}")

    # y supports: Sat_Score_mean, Mood_Score_mean, count, etc.
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

    # If time axis requested, force chronological order
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
            plot_df = chart_df.groupby([x] + ([group_by] if group_by else []), as_index=False).size().rename(columns={"size": "Value"})
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


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="tsworks | Insights Engine", layout="wide")

SAT_MAP = {
    "Extremely satisfied": 10, "Satisfied": 8, "Somewhat satisfied": 7,
    "Neutral": 5, "Somewhat dissatisfied": 3, "Dissatisfied": 2, "Extremely dissatisfied": 0
}
MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

st.title("📊 tsworks Insights Engine")
st.caption("Custom Analytics powered by OpenAI ChatGPT")

# -----------------------------
# Sidebar setup
# -----------------------------
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not configured. Please set OPENAI_API_KEY.")
        st.stop()

    uploaded_file = st.file_uploader("Upload 'tsworks Employee Pulse' Excel", type=["xlsx", "csv"])

if not uploaded_file:
    st.info("Upload an Excel/CSV to start.")
    st.stop()

# Reset state on new file
if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    st.session_state.messages = []
    st.session_state.pop("sel_year", None)
    st.session_state.pop("sel_month", None)
    st.session_state["default_period_set"] = False

# Load file
df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

df.columns = df.columns.str.strip()
df = df.fillna("N/A")

# Derived scores
df["Sat_Score"] = df["How satisfied are you working at tsworks?"].map(SAT_MAP).fillna(5)
df["Mood_Score"] = df["How are you feeling overall this month?"].map(MOOD_MAP).fillna(3)

# Add time cols
df = add_period_cols(df)

# -----------------------------
# Sidebar filters: default latest month/year
# -----------------------------
latest_key = df["_PeriodKey"].dropna().max()
latest_row = df.loc[df["_PeriodKey"] == latest_key].iloc[0]
latest_year = int(latest_row["Year"])
latest_month = latest_row["Month"]

years = sorted(df["Year"].dropna().unique().astype(int).tolist())

if (not st.session_state.get("default_period_set", False)) or (st.session_state.get("sel_year") not in years):
    st.session_state["sel_year"] = latest_year
    st.session_state["default_period_set"] = True

sel_year = st.sidebar.selectbox(
    "Filter by Year",
    options=years,
    index=years.index(st.session_state["sel_year"]),
    key="sel_year"
)

df_year = df[df["Year"] == sel_year]
months = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
          if m in set(df_year["Month"].dropna())]

if st.session_state.get("sel_month") not in months:
    st.session_state["sel_month"] = max(months, key=lambda m: MONTH_ORDER[m]) if months else latest_month

sel_month = st.sidebar.selectbox(
    "Filter by Month",
    options=months,
    index=months.index(st.session_state["sel_month"]) if months else 0,
    key="sel_month"
)

# Apply month/year first
df_filtered = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()

# Department filter
depts = sorted(df_filtered["Department"].dropna().unique())
sel_dept = st.sidebar.selectbox("Filter by Department", ["All Departments"] + list(depts))
if sel_dept != "All Departments":
    df_filtered = df_filtered[df_filtered["Department"] == sel_dept]

# Manager filter
managers = sorted(df_filtered["Reporting Manager"].dropna().unique())
sel_manager = st.sidebar.selectbox("Filter by Manager", ["All Managers"] + list(managers))
if sel_manager != "All Managers":
    df_filtered = df_filtered[df_filtered["Reporting Manager"] == sel_manager]

# -----------------------------
# Main tabs: Dashboard + Responses
# -----------------------------
st.caption(f"Showing data for: **{sel_month} {sel_year}**")

tab_dash, tab_resp = st.tabs(["📊 Dashboard", "🧾 Responses"])

# -----------------------------
# Dashboard tab (your existing KPIs + charts + AI)
# -----------------------------
with tab_dash:
    total = len(df_filtered)
    if total > 0:
        promoters = len(df_filtered[df_filtered["Sat_Score"] >= 9])
        detractors = len(df_filtered[df_filtered["Sat_Score"] <= 6])
        nps = round(((promoters - detractors) / total) * 100)
        avg_sat = round(df_filtered["Sat_Score"].mean(), 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Employee NPS", f"{nps}")
        m2.metric("Avg Satisfaction", f"{avg_sat} / 10")
        m3.metric("Total Responses", total)

        tab1, tab2 = st.tabs(["📈 Sentiment Analysis", "📋 Textual Insights"])

        with tab1:
            c1, c2 = st.columns(2)
            group_by = "Reporting Manager" if sel_dept != "All Departments" else "Department"

            fig_sat = px.bar(
                df_filtered.groupby(group_by)["Sat_Score"].mean().reset_index(),
                x=group_by, y="Sat_Score",
                title=f"Satisfaction Drill-down: {group_by}",
                color="Sat_Score", color_continuous_scale="RdYlGn", range_color=[0, 10]
            )
            c1.plotly_chart(fig_sat, use_container_width=True)

            fig_mood = px.pie(
                df_filtered,
                names="How are you feeling overall this month?",
                title="Current Team Mood",
                hole=0.4
            )
            c2.plotly_chart(fig_mood, use_container_width=True)

        with tab2:
            show_cols = [
                "Name", "Department", "Reporting Manager",
                "How are you feeling overall this month?",
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
            ]
            show_cols = [c for c in show_cols if c in df_filtered.columns]
            st.dataframe(df_filtered[show_cols], use_container_width=True, hide_index=True)

        st.divider()

        # -----------------------------
        # AI Chatbot expander (unchanged conceptually, cleaner output)
        # -----------------------------
        with st.expander("🤖 AI ChatBot (Ask questions about the data)", expanded=False):
            llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)

            CUSTOM_PREFIX = """
You are a Senior Manager at tsworks.

You can answer questions in Markdown.
If a chart is useful, respond with ONLY valid JSON (no extra text) using this schema:

{
  "chart_required": true,
  "chart_type": "line" | "bar" | "pie",
  "x": "Month" | "Department" | "Reporting Manager" | "Year" | "<other column>",
  "y": "Sat_Score_mean" | "Mood_Score_mean" | "count" | "<numeric_column>_mean",
  "group_by": "<optional column>",
  "time_filter": "this_month" | "last_6_months" | "last_quarter" | "all",
  "summary": "<short executive insight>"
}

Rules:
- For trends over time, use x="Month" and y="Sat_Score_mean" or "Mood_Score_mean" (or "count").
- If the question asks "this month", use time_filter="this_month" (UI-selected).
- Keep it professional.
- If chart_required is true, output ONLY JSON. Do not include any explanatory text before or after.
"""

            # IMPORTANT: keep this False for safety
            agent = create_pandas_dataframe_agent(
                llm,
                df,  # full dataset
                verbose=False,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                agent_type="openai-tools",
                prefix=CUSTOM_PREFIX
            )

            chat_container = st.container(height=260)

            with chat_container:
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            query = st.chat_input("Ask about the data...")
            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(query)

                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("AI is thinking..."):
                            context = f"""
UI filters currently selected:
- Year: {sel_year}
- Month: {sel_month}
- Department: {sel_dept}
- Manager: {sel_manager}

Instructions:
- If the user asks "this month", interpret it as the UI-selected Year/Month.
- For comparisons (last quarter, last 6 months, YoY), use full dataset across months/years.
- Always make the period used explicit.
"""
                            try:
                                result = agent.invoke({"input": context + "\n\nUser question: " + query})
                                response = (result.get("output") or "").strip()

                                chart_spec = extract_json_object(response)
                                if chart_spec and chart_spec.get("chart_required"):
                                    tf = (chart_spec.get("time_filter") or "all")
                                    chart_df = apply_time_filter(df, tf, sel_year, sel_month)

                                    # Respect current UI dept/manager filters (optional)
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

    else:
        st.warning("No data available for the current filter selection.")


# -----------------------------
# Responses tab: clean employee browsing + search + pagination + drilldown
# -----------------------------
with tab_resp:
    st.subheader("Employee Responses Explorer")

    # View mode
    view_mode = st.radio(
        "View mode",
        ["All employees in selection", "By Manager", "Individual Employee"],
        index=0,
        horizontal=True
    )

    # STEP 5 guardrail: Block individual employee view unless Manager is selected
    if view_mode == "Individual Employee" and sel_manager == "All Managers":
        st.warning("Select a **Manager** from the left sidebar to view individual employee details.")
        st.stop()

    # Base data for responses should follow the SAME month/year/department/manager filters
    resp_df = df_filtered.copy()

    # If view by manager, user chooses manager inside this tab (still within the filtered slice)
    if view_mode == "By Manager":
        mgrs_in_slice = sorted(resp_df["Reporting Manager"].dropna().unique().tolist())
        chosen_mgr = st.selectbox("Choose a manager", ["All Managers"] + mgrs_in_slice)
        if chosen_mgr != "All Managers":
            resp_df = resp_df[resp_df["Reporting Manager"] == chosen_mgr]

    # If individual employee, choose employee within current filtered slice
    if view_mode == "Individual Employee":
        employees_in_slice = sorted(resp_df["Name"].dropna().unique().tolist())
        chosen_emp = st.selectbox("Choose an employee", ["-- Select --"] + employees_in_slice)
        if chosen_emp != "-- Select --":
            resp_df = resp_df[resp_df["Name"] == chosen_emp]

    # Search box (works for all modes except where already single employee)
    if view_mode != "Individual Employee":
        search = st.text_input("Search employee name", placeholder="Type to search (e.g., 'Sunil')").strip()
        if search:
            resp_df = resp_df[resp_df["Name"].astype(str).str.contains(search, case=False, na=False)]

    # Pagination controls
    st.caption(f"Rows: {len(resp_df)}")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    total_pages = max(1, math.ceil(len(resp_df) / page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * page_size
    end = start + page_size
    page_df = resp_df.iloc[start:end].copy()

    # Columns to show
    cols_to_show = [
        "Name", "Department", "Reporting Manager", "Year", "Month",
        "How satisfied are you working at tsworks?",
        "How are you feeling overall this month?",
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
    ]
    cols_to_show = [c for c in cols_to_show if c in page_df.columns]

    st.dataframe(page_df[cols_to_show], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🔍 Drill-down (Employee Detail)")

    drill_options = sorted(resp_df["Name"].dropna().unique().tolist())
    drill_emp = st.selectbox("Pick an employee from current view", ["-- Select --"] + drill_options)

    if drill_emp and drill_emp != "-- Select --":
        emp_rows = resp_df[resp_df["Name"] == drill_emp].copy()
        emp_rows = emp_rows.sort_values("_PeriodKey", ascending=False) if "_PeriodKey" in emp_rows.columns else emp_rows

        top = emp_rows.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Employee", str(top.get("Name", "")))
        c2.metric("Department", str(top.get("Department", "")))
        c3.metric("Manager", str(top.get("Reporting Manager", "")))

        for _, r in emp_rows.iterrows():
            period = f"{r.get('Month','')} {r.get('Year','')}"
            with st.expander(f"📅 {period}", expanded=True):
                st.write(f"**Satisfaction:** {r.get('How satisfied are you working at tsworks?', 'N/A')}")
                st.write(f"**Mood:** {r.get('How are you feeling overall this month?', 'N/A')}")
                st.markdown("**Key accomplishments:**")
                st.write(r.get("Key Accomplishments this Month", "N/A"))
                st.markdown("**What’s not going well / disappointments:**")
                st.write(r.get("What’s not going well or causing disappointment?", "N/A"))

                # Optional: show extra Qs if present
                extras = [
                    "Any concerns, blockers, or risks?",
                    "Do you need support from bench resources or other teams?",
                    "How is your work-life balance?",
                    "How is your current workload?",
                    "Are you currently supporting or mentoring junior team members?",
                    "Suggestions for process or workflow improvements",
                    "Planned PTO this month and coverage plan",
                    "Goal Progress"
                ]
                for col in extras:
                    if col in emp_rows.columns:
                        st.markdown(f"**{col}**")
                        st.write(r.get(col, "N/A"))
