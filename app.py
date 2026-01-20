import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

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
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload 'tsworks Employee Pulse' Excel", type=["xlsx", "csv"])

if uploaded_file:
    # Logic to clear chat history when a NEW file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.current_file = uploaded_file.name

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
    depts = sorted(df['Department'].dropna().unique())
    sel_dept = st.sidebar.selectbox("Filter by Department", ["All Departments"] + list(depts))

    df_filtered = df[df['Department'] == sel_dept] if sel_dept != "All Departments" else df
    
    managers = sorted(df_filtered['Reporting Manager'].dropna().unique())
    sel_manager = st.sidebar.selectbox("Filter by Manager", ["All Managers"] + list(managers))
    
    if sel_manager != "All Managers":
        df_filtered = df_filtered[df_filtered['Reporting Manager'] == sel_manager]

    # --- KPI CALCULATIONS ---
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
                'Key Accomplishments this Month', 
                'What’s not going well or causing disappointment?'
            ]])

        # --- AI AGENT SECTION ---
        st.divider()
        st.subheader("🤖 AI Data Architect")
        
        if api_key:
            try:
                # Initialize OpenAI LLM                
                llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key, temperature=0)


                CUSTOM_PREFIX = "You are a Senior Manager at tsworks. Provide professional summaries. No code. No 'df' mentions. Markdown only."
                
                agent = create_pandas_dataframe_agent(
                    llm, df_filtered, verbose=False, allow_dangerous_code=True,
                    handle_parsing_errors=True, agent_type="openai-tools", prefix=CUSTOM_PREFIX
                )

                # Fixed-size scrollable window
                chat_container = st.container(height=450)
                
                with chat_container:
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                if query := st.chat_input("Ask about the data..."):
                    st.session_state.messages.append({"role": "user", "content": query})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(query)

                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner("AI is thinking..."):
                                try:
                                    # OpenAI Agent response is usually a direct string in 'output'
                                    result = agent.invoke({"input": query})
                                    clean_text = result.get("output", "No response generated.")
                                    st.markdown(clean_text)
                                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                                except Exception as e:
                                    st.error(f"Error: {e}")
            except Exception as init_err:
                st.error(f"AI Setup Error: {init_err}")
        else:
            st.warning("Enter OpenAI API Key to begin.")




