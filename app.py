import streamlit as st

# Import your research workflow
from multi_agent_research_assistant import research_graph


st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-Agent Research Assistant")

st.write(
    "Research questions using LangGraph, Gemini, and Tavily."
)

question = st.text_area(
    "Enter your research question:",
    placeholder="What are the effects of artificial intelligence on education?",
    height=120
)


if st.button("Start Research"):

    if not question.strip():
        st.warning("Please enter a research question.")

    else:

        with st.spinner("Researching... Please wait."):

            try:

                result = research_graph.invoke({

                    "question": question,

                    "research_plan": "",

                    "research_results": [],

                    "analysis": "",

                    "fact_check": "",

                    "final_report": ""

                })

                st.success("Research completed!")

                st.subheader("Final Research Report")

                st.write(result["final_report"])

            except Exception as e:

                st.error("Something went wrong while running the research system.")

                st.exception(e)
