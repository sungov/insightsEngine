import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="tsworks | Insights Engine", layout="wide")

# --- SATISFACTION MAPPING ---
SAT_MAP = {
    "Extremely satisfied": 10, "Satisfied": 8, "Somewhat satisfied": 7,
    "Neutral": 5, "Somewhat dissatisfied": 3, "Dissatisfied": 2, "Extremely dissatisfied": 0
}
MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

# --- HEADER ---
st.title("📊 tsworks Insights Engine")
st.caption("Custom Analytics powered by OpenAI ChatGPT")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Setup")
    # Switch to OpenAI Key input
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not configured. Please set OPENAI_API_KEY.")
        st.stop()
    uploaded_file = st.file_uploader("Upload 'tsworks Employee Pulse' Excel", type=["xlsx", "csv"])

if uploaded_file:
    # Logic to clear chat history when a NEW file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.current_file = uploaded_file.name
    
        # IMPORTANT: remove widget values so selectboxes can re-initialize cleanly
        st.session_state.pop("sel_year", None)
        st.session_state.pop("sel_month", None)
        st.session_state["default_period_set"] = False



    # Load and Clean Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Clean leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    df = df.fillna("N/A")
    df['Sat_Score'] = df['How satisfied are you working at tsworks?'].map(SAT_MAP).fillna(5)
    df['Mood_Score'] = df['How are you feeling overall this month?'].map(MOOD_MAP).fillna(3)

    # Sidebar Filters
        # --- NORMALIZE MONTH/YEAR COLUMNS ---       
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = df["Month"].astype(str).str.strip().str[:3].str.title()
    
    MONTH_ORDER = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                   "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    df["_MonthNum"] = df["Month"].map(MONTH_ORDER)
    df["_PeriodKey"] = df["Year"] * 100 + df["_MonthNum"]
    
    # Latest available period in the whole file
    latest_key = df["_PeriodKey"].dropna().max()
    latest_row = df.loc[df["_PeriodKey"] == latest_key].iloc[0]
    latest_year = int(latest_row["Year"])
    latest_month = latest_row["Month"]
    
    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    
    # ✅ Set defaults when:
    # - first run / refresh (keys missing)
    # - or defaults not set yet
    # - or current values are invalid
    if (not st.session_state.get("default_period_set", False)
        or st.session_state.get("sel_year") not in years):
        st.session_state["sel_year"] = latest_year
        st.session_state["default_period_set"] = True
    
    sel_year = st.sidebar.selectbox(
        "Filter by Year",
        options=years,
        index=years.index(st.session_state["sel_year"]),
        key="sel_year"
    )
    
    # Month options for selected year
    df_year = df[df["Year"] == sel_year]
    months = [m for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
              if m in set(df_year["Month"].dropna())]
    
    # ✅ Month default / validity
    if st.session_state.get("sel_month") not in months:
        # pick latest month within that year (by month number)
        if len(months) > 0:
            # choose max based on order
            st.session_state["sel_month"] = max(months, key=lambda m: MONTH_ORDER[m])
        else:
            st.session_state["sel_month"] = latest_month  # fallback
    
    sel_month = st.sidebar.selectbox(
        "Filter by Month",
        options=months,
        index=months.index(st.session_state["sel_month"]) if months else 0,
        key="sel_month"
    )
    
    # Apply Year + Month filter
    df_filtered = df[(df["Year"] == sel_year) & (df["Month"] == sel_month)].copy()


    # Department filter (after time filters)
    depts = sorted(df_filtered["Department"].dropna().unique())
    sel_dept = st.sidebar.selectbox("Filter by Department", ["All Departments"] + list(depts))

    if sel_dept != "All Departments":
        df_filtered = df_filtered[df_filtered["Department"] == sel_dept]

    # Manager filter (after dept filter)
    managers = sorted(df_filtered["Reporting Manager"].dropna().unique())
    sel_manager = st.sidebar.selectbox("Filter by Manager", ["All Managers"] + list(managers))

    if sel_manager != "All Managers":
        df_filtered = df_filtered[df_filtered["Reporting Manager"] == sel_manager]


    # --- KPI CALCULATIONS ---
    st.caption(f"Showing data for: **{st.session_state.get('sel_month', 'N/A')} {st.session_state.get('sel_year', 'N/A')}**")
    total = len(df_filtered)
    if total > 0:
        promoters = len(df_filtered[df_filtered['Sat_Score'] >= 9])
        detractors = len(df_filtered[df_filtered['Sat_Score'] <= 6])
        nps = round(((promoters - detractors) / total) * 100)
        avg_sat = round(df_filtered['Sat_Score'].mean(), 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Employee NPS", f"{nps}")
        m2.metric("Avg Satisfaction", f"{avg_sat} / 10")
        m3.metric("Total Responses", total)

        # --- VISUALS ---
        tab1, tab2 = st.tabs(["📈 Sentiment Analysis", "📋 Textual Insights"])
        with tab1:
            c1, c2 = st.columns(2)
            group_by = 'Reporting Manager' if sel_dept != "All Departments" else 'Department'
            fig_sat = px.bar(
                df_filtered.groupby(group_by)['Sat_Score'].mean().reset_index(),
                x=group_by, y='Sat_Score', 
                title=f"Satisfaction Drill-down: {group_by}",
                color='Sat_Score', color_continuous_scale='RdYlGn', range_color=[0,10]
            )
            c1.plotly_chart(fig_sat, use_container_width=True)

            fig_mood = px.pie(df_filtered, names='How are you feeling overall this month?', title="Current Team Mood", hole=0.4)
            c2.plotly_chart(fig_mood, use_container_width=True)

        with tab2:
            st.dataframe(df_filtered[[
                'Name', 
                'Department', 
                'Reporting Manager', 
                'How are you feeling overall this month?',                
                'Key Accomplishments this Month', 
                'What’s not going well or causing disappointment?',
                'Any concerns, blockers, or risks?',
                'Do you need support from bench resources or other teams?',
                'How is your work-life balance?',
                'How is your current workload?',
                'Are you currently supporting or mentoring junior team members?',
                'Suggestions for process or workflow improvements',
                'Planned PTO this month and coverage plan',
                'Goal Progress'
                
            ]])

        # --- AI AGENT SECTION ---
        st.divider()

        # Helpers for chronological ordering + time filtering
        MONTH_ORDER = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                       "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        
        def add_period_cols(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
            d["Month"] = d["Month"].astype(str).str.strip().str[:3].str.title()
            d["_MonthNum"] = d["Month"].map(MONTH_ORDER)
            d["PeriodDate"] = pd.to_datetime(
                d["Year"].astype("Int64").astype(str) + "-" + d["_MonthNum"].astype("Int64").astype(str) + "-01",
                errors="coerce"
            )
            d["_PeriodLabel"] = d["Month"] + " " + d["Year"].astype("Int64").astype(str)
            d["_PeriodKey"] = d["Year"] * 100 + d["_MonthNum"]
            return d
        
        def apply_time_filter(d: pd.DataFrame, time_filter: str, sel_year, sel_month) -> pd.DataFrame:
            d = add_period_cols(d)
            d = d.dropna(subset=["PeriodDate"])
        
            if time_filter == "this_month":
                # interpret as UI-selected
                return d[(d["Year"] == sel_year) & (d["Month"] == sel_month)].copy()
        
            if time_filter == "last_6_months":
                end = d["PeriodDate"].max()
                start = end - pd.DateOffset(months=6)
                return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()
        
            if time_filter == "last_quarter":
                # last 3 months from max date in dataset
                end = d["PeriodDate"].max()
                start = end - pd.DateOffset(months=3)
                return d[(d["PeriodDate"] >= start) & (d["PeriodDate"] <= end)].copy()
        
            # "all" or unknown
            return d.copy()
        
        def build_chart(chart_df: pd.DataFrame, spec: dict):
            chart_type = (spec.get("chart_type") or "line").lower().strip()
            x = spec.get("x")
            y = spec.get("y")
            group_by = spec.get("group_by")
        
            # Safety: allow only known columns
            allowed_cols = set(chart_df.columns)
            if x not in allowed_cols:
                raise ValueError(f"Invalid x column: {x}")
            if group_by and group_by not in allowed_cols:
                raise ValueError(f"Invalid group_by column: {group_by}")
        
            # Handle common aggregations in y
            # Examples AI can send: "Sat_Score_mean", "Mood_Score_mean", "count"
            # Or it can send raw column names like "Sat_Score"
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
        
            # If y_col is a column, validate
            if y_col and y_col not in allowed_cols:
                raise ValueError(f"Invalid y column: {y_col}")
        
            # If x is Month/Year-like, force chronological plotting
            time_like = x in ["Month", "Year", "_PeriodLabel", "PeriodDate", "_PeriodKey"]
        
            if time_like:
                chart_df = add_period_cols(chart_df).dropna(subset=["PeriodDate"])
        
                # Pre-aggregate by period (+ group_by if present)
                group_cols = ["PeriodDate", "_PeriodLabel"]
                if group_by:
                    group_cols.append(group_by)
        
                if agg == "count":
                    plot_df = chart_df.groupby(group_cols, as_index=False).size().rename(columns={"size": "Value"})
                    y_plot = "Value"
                else:
                    # default to mean for trend charts if no agg specified
                    if agg is None:
                        agg = "mean"
                    plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
                    y_plot = y_col
        
                plot_df = plot_df.sort_values("PeriodDate")
        
                if chart_type == "line":
                    fig = px.line(plot_df, x="_PeriodLabel", y=y_plot, color=group_by, markers=True)
                elif chart_type == "bar":
                    fig = px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)
                else:
                    # pie doesn't make sense over time; fallback to bar
                    fig = px.bar(plot_df, x="_PeriodLabel", y=y_plot, color=group_by)
        
                return fig
        
            # Non-time charts
            if chart_type == "pie":
                # pie expects names + values
                if agg == "count":
                    plot_df = chart_df.groupby(x, as_index=False).size().rename(columns={"size": "Value"})
                    fig = px.pie(plot_df, names=x, values="Value", hole=0.4)
                else:
                    # aggregate numeric column by x
                    if agg is None:
                        agg = "mean"
                    plot_df = chart_df.groupby(x, as_index=False).agg({y_col: agg})
                    fig = px.pie(plot_df, names=x, values=y_col, hole=0.4)
                return fig
        
            if chart_type == "bar":
                if agg == "count":
                    plot_df = chart_df.groupby([x] + ([group_by] if group_by else []), as_index=False).size().rename(columns={"size":"Value"})
                    fig = px.bar(plot_df, x=x, y="Value", color=group_by)
                else:
                    if agg is None:
                        agg = "mean"
                    group_cols = [x] + ([group_by] if group_by else [])
                    plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
                    fig = px.bar(plot_df, x=x, y=y_col, color=group_by)
                return fig
        
            # Default: line
            if agg is None:
                agg = "mean"
            group_cols = [x] + ([group_by] if group_by else [])
            plot_df = chart_df.groupby(group_cols, as_index=False).agg({y_col: agg})
            fig = px.line(plot_df, x=x, y=y_col, color=group_by, markers=True)
            return fig
        
        
        with st.expander("🤖 AI ChatBot (Ask questions about the data)", expanded=False):
            if api_key:
                try:
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
        """
        
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,  # full dataset
                        verbose=False,
                        allow_dangerous_code=True,  # safer
                        handle_parsing_errors=True,
                        agent_type="openai-tools",
                        prefix=CUSTOM_PREFIX
                    )
        
                    chat_container = st.container(height=260)  # reduce height
        
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
                                    try:
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
        
                                        result = agent.invoke({"input": context + "\n\nUser question: " + query})
                                        response = (result.get("output") or "").strip()
        
                                        # Try JSON chart mode
                                        try:
                                            chart_spec = json.loads(response)
                                            if chart_spec.get("chart_required"):
                                                tf = (chart_spec.get("time_filter") or "all").strip().lower()
                                                chart_df = apply_time_filter(df, tf, sel_year, sel_month)
        
                                                # Optional: also apply dept/manager filters if user already selected them
                                                if sel_dept != "All Departments":
                                                    chart_df = chart_df[chart_df["Department"] == sel_dept]
                                                if sel_manager != "All Managers":
                                                    chart_df = chart_df[chart_df["Reporting Manager"] == sel_manager]
        
                                                fig = build_chart(chart_df, chart_spec)
                                                st.plotly_chart(fig, use_container_width=True)
                                                st.markdown(chart_spec.get("summary", ""))
        
                                                # Store a friendly assistant message (not raw JSON)
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": chart_spec.get("summary", "Chart generated.")
                                                })
                                            else:
                                                # If JSON but chart not required, show it as text (rare)
                                                st.markdown(response)
                                                st.session_state.messages.append({"role": "assistant", "content": response})
        
                                        except json.JSONDecodeError:
                                            # Normal text response
                                            st.markdown(response)
                                            st.session_state.messages.append({"role": "assistant", "content": response})
        
                                    except Exception as e:
                                        st.error(f"Error: {e}")
        
                except Exception as init_err:
                    st.error(f"AI Setup Error: {init_err}")
            else:
                st.warning("OpenAI API key not configured.")

