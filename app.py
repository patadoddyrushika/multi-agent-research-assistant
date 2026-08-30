import streamlit as st

from multi_agent_research_assistant import research_graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 Multi-Agent Research Assistant")

st.write(
    "Research questions using LangGraph, Groq, and Tavily."
)


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_area(
    "Enter your research question:",
    placeholder="What are the effects of artificial intelligence on education?",
    height=120,
)


# ============================================================
# START RESEARCH
# ============================================================

if st.button("Start Research"):

    if not question.strip():
        st.warning("Please enter a research question.")

    else:

        with st.spinner("Researching... Please wait."):

            try:

                result = research_graph.invoke(
                    {
                        "question": question,
                        "research_plan": "",
                        "search_queries": [],
                        "research_results": [],
                        "analysis": "",
                        "fact_check": "",
                        "final_report": "",
                    }
                )

                st.success("Research completed!")

                st.subheader("Final Research Report")

                st.markdown(
                    result.get(
                        "final_report",
                        "No final report was generated.",
                    )
                )

                # Optional expandable sections
                with st.expander("View Research Plan"):
                    st.write(
                        result.get(
                            "research_plan",
                            "No research plan available.",
                        )
                    )

                with st.expander("View Sources"):
                    sources = result.get("research_results", [])

                    if sources:
                        for source in sources:
                            title = source.get("title", "Source")
                            url = source.get("url", "")

                            if url:
                                st.markdown(
                                    f"- [{title}]({url})"
                                )
                            else:
                                st.write(f"- {title}")
                    else:
                        st.write("No sources found.")

            except Exception as e:

                st.error(
                    "Something went wrong while running the research system."
                )

                st.exception(e)
