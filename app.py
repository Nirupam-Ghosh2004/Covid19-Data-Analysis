import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from covid_eda import load_and_clean_data  # Import reusable logic from covid_eda.py


st.title("COVID-19 Data Analysis and Visualization Dashboard")
st.markdown("""
This interactive dashboard provides insights into **COVID-19 cases, deaths, and trends** using WHO official data. 
Explore daily and cumulative statistics, visualize pandemic waves, and analyze relationships between cases and deaths.
""")


@st.cache_data
def get_data():
    return load_and_clean_data()

df = get_data()


st.sidebar.header("Filter Options")
country_list = sorted(df["location"].dropna().unique())
selected_country = st.sidebar.selectbox("Select Country", country_list, index=country_list.index("India") if "India" in country_list else 0)

filtered_df = df[df["location"] == selected_country].copy()
filtered_df = filtered_df.sort_values("date")


if "new_cases" in filtered_df.columns:
    filtered_df["new_cases_7d_avg"] = filtered_df["new_cases"].rolling(window=7).mean()
if "new_deaths" in filtered_df.columns:
    filtered_df["new_deaths_7d_avg"] = filtered_df["new_deaths"].rolling(window=7).mean()


st.subheader(f"COVID-19 Overview — {selected_country}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Cases", f"{int(filtered_df['total_cases'].max()):,}")
col2.metric("Total Deaths", f"{int(filtered_df['total_deaths'].max()):,}")
col3.metric("Peak New Cases", f"{int(filtered_df['new_cases'].max()):,}")


tab1, tab2 = st.tabs(["📈 Cases Trend", "⚰️ Deaths Trend"])
with tab1:
    st.markdown(f"### Daily and 7-Day Average New Cases — {selected_country}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["date"], filtered_df["new_cases"], label="Daily Cases", alpha=0.6)
    ax.plot(filtered_df["date"], filtered_df["new_cases_7d_avg"], label="7-Day Avg", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.markdown(f"### Daily and 7-Day Average Deaths — {selected_country}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["date"], filtered_df["new_deaths"], label="Daily Deaths", color="red", alpha=0.6)
    ax.plot(filtered_df["date"], filtered_df["new_deaths_7d_avg"], label="7-Day Avg", color="black", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.legend()
    st.pyplot(fig)


st.markdown("### Correlation Between Key Variables")
numeric_cols = filtered_df.select_dtypes(include="number")
if len(numeric_cols.columns) > 1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric_cols.corr(), cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)


st.markdown("---")
st.subheader("🧠 Insights & Key Observations")
st.markdown(f"""
- **Peak daily new cases** in *{selected_country}* reached **{int(filtered_df['new_cases'].max()):,}**.
- The highest surge occurred around **{filtered_df.loc[filtered_df['new_cases'].idxmax(), 'date'].strftime('%B %Y')}**.
- **Total confirmed cases:** {int(filtered_df['total_cases'].max()):,}
- **Total deaths recorded:** {int(filtered_df['total_deaths'].max()):,}
""")


st.markdown("---")
st.markdown("Developed by **Tuhin (Nirupam Ghosh)** — Streamlit Data Visualization Project 2025")