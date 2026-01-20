import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold

# --- PAGE CONFIG ---
st.set_page_config(page_title="tsworks | Insights Engine", layout="wide")

# --- SATISFACTION MAPPING ---
# Converts text responses into numbers for NPS calculation
SAT_MAP = {
    "Extremely satisfied": 10,
    "Satisfied": 8,
    "Somewhat satisfied": 7,
    "Neutral": 5,
    "Somewhat dissatisfied": 3,
    "Dissatisfied": 2,
    "Extremely dissatisfied": 0
}

MOOD_MAP = {"Great": 5, "Good": 4, "Neutral": 3, "Challenged": 2, "Burned Out": 1}

# --- HEADER ---
st.title("📊 tsworks Insights Engine")
st.caption("Custom Analytics for tsworks Employee Pulse Surveys")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload 'tsworks Employee Pulse' Excel", type=["xlsx", "csv"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # 1. Apply Mapping for Analytics
    df['Sat_Score'] = df['How satisfied are you working at tsworks?'].map(SAT_MAP).fillna(5)
    df['Mood_Score'] = df['How are you feeling overall this month?'].map(MOOD_MAP).fillna(3)

    # 2. Sidebar Filters (Hierarchical)
    depts = sorted(df['Department'].dropna().unique())
    sel_dept = st.sidebar.selectbox("Filter by Department", ["All Departments"] + list(depts))

    if sel_dept != "All Departments":
        df_filtered = df[df['Department'] == sel_dept]
        managers = sorted(df_filtered['Reporting Manager'].dropna().unique())
    else:
        df_filtered = df
        managers = sorted(df['Reporting Manager'].dropna().unique())

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

        # --- METRICS BAR ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Employee NPS", f"{nps}")
        m2.metric("Avg Satisfaction", f"{avg_sat} / 10")
        m3.metric("Total Responses", total)

        # --- VISUALS ---
        tab1, tab2 = st.tabs(["📈 Sentiment Analysis", "📋 Textual Insights"])

        with tab1:
            c1, c2 = st.columns(2)
            
            # Chart 1: Satisfaction by Dept or Manager
            group_by = 'Reporting Manager' if sel_dept != "All Departments" else 'Department'
            fig_sat = px.bar(
                df_filtered.groupby(group_by)['Sat_Score'].mean().reset_index(),
                x=group_by, y='Sat_Score', 
                title=f"Satisfaction Drill-down: {group_by}",
                color='Sat_Score', color_continuous_scale='RdYlGn', range_color=[0,10]
            )
            c1.plotly_chart(fig_sat, use_container_width=True)
