import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-Agent Research Assistant")
st.write("Research questions using LangGraph, Gemini, and Tavily.")

question = st.text_area(
    "Enter your research question:",
    placeholder="What are the effects of artificial intelligence on education?"
)

if st.button("Start Research"):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        st.info("Research started...")

        # Backend will be connected here
        st.success("Research system is ready to be connected.")
