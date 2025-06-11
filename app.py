import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent_logic import make_budget_decisions, generate_explanations
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Autonomous Campaign Manager", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("🧠 Campaign Uploader")
    uploaded_file = st.file_uploader("📤 Upload Clean Room Campaign Data CSV", type=["csv"])
    st.markdown("You can upload a file with columns like:")
    st.code("Platform, Device Type, Audience Type, Spend ($), Conversions", language="text")

    
    st.markdown("""
    ### 💡 Sample Questions
    - Which platform is delivering the highest ROI?
    - Which audience type is converting the best?
    - Which device type has the lowest cost per conversion?
    - Where should we reduce the budget?
    - Which campaigns are underperforming and why?
    - Which platform offers the best value for money?
    - Should we reallocate budget from TikTok to Meta?
    - Is the spend-to-conversion ratio good for Google Ads?
    - Are any platforms overspending with poor results?
    - If we double the spend on Meta, what might happen to conversions?
    """)

    st.markdown("- Which platform has the highest ROI?")
    st.markdown("- Where should I reduce budget?")
    st.markdown("- Which audience is converting best?")

# Main area
st.title("🤖 Autonomous Campaign Manager Agent")
st.markdown("""
This tool simulates an **agentic AI** that analyzes ad campaign performance 
(e.g., from Google Ads, Amazon DSP, Meta, TikTok) and suggests **budget reallocations**.
It uses OpenAI to generate **natural language justifications** for its decisions.
""")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Campaign Data")
    st.dataframe(df)

    df = df.head(10)  # slightly higher limit for better charts

    with st.spinner("🔍 Analyzing performance and optimizing budgets..."):
        decisions = make_budget_decisions(df)
        decisions = generate_explanations(decisions)

    st.subheader("💰 Budget Actions & AI Recommendations")
    st.dataframe(decisions[['Platform', 'Device Type', 'Audience Type', 'Spend ($)', 'ROI', 'Action', 'Explanation']])

    # Charts section
    st.subheader("📈 Visual Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Spend by Platform**")
        fig1, ax1 = plt.subplots()
        sns.barplot(data=df, x='Platform', y='Spend ($)', ax=ax1)
        ax1.set_ylabel("Spend ($)")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Conversions by Platform**")
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x='Platform', y='Conversions', ax=ax2)
        ax2.set_ylabel("Conversions")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**ROI by Audience Type**")
        fig3, ax3 = plt.subplots()
        sns.barplot(data=decisions, x='Audience Type', y='ROI', ax=ax3)
        ax3.set_ylabel("ROI")
        st.pyplot(fig3)

    with col4:
        st.markdown("**Action Count by Platform**")
        fig4, ax4 = plt.subplots()
        sns.countplot(data=decisions, x='Platform', hue='Action', ax=ax4)
        ax4.set_ylabel("Count")
        st.pyplot(fig4)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask AI about your campaign")
    question = st.text_input("Type your question here:")

    if question:
        context = df.to_csv(index=False)
        prompt = f'''You are a marketing analyst AI. Here's campaign data:

{context}

Answer this question based on the data:
{question}
'''
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error: {e}"

        st.session_state.chat_history.append((question, answer))

    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
else:
    st.info("Please upload a campaign CSV file from the sidebar to begin.")