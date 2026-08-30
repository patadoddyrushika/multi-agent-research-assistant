import os
from typing import TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


# ============================================================
# 1. API KEYS
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ============================================================
# 2. LOAD KEYS FROM STREAMLIT SECRETS IF AVAILABLE
# ============================================================

try:
    import streamlit as st

    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

    if not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

    if not TAVILY_API_KEY:
        TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

except Exception:
    pass


# ============================================================
# 3. CHECK REQUIRED KEYS
# ============================================================

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")


# ============================================================
# 4. AI MODELS
# ============================================================

# MODEL 1 — GROQ
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
)


# MODEL 2 — GEMINI
# Gemini is optional because the free Gemini quota may be exhausted.
gemini_llm = None

if GOOGLE_API_KEY:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY,
    )


# ============================================================
# 5. TAVILY
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# 6. STATE
# ============================================================

class ResearchState(TypedDict):
    question: str
    research_plan: str
    research_results: List[str]
    analysis: str
    fact_check: str
    final_report: str
    research_attempts: int


# ============================================================
# 7. HELPER FUNCTION
# ============================================================

def get_text(response) -> str:
    """
    Converts LangChain/Gemini responses into normal text.
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")

                if text:
                    parts.append(text)

            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(parts)

    return str(content)


# ============================================================
# 8. SAFE GEMINI CALL
# ============================================================

def call_gemini(prompt: str) -> str:
    """
    Try Gemini first.
    If Gemini is unavailable or quota is exhausted,
    automatically use Groq instead.
    """

    if gemini_llm is not None:

        try:
            response = gemini_llm.invoke(prompt)

            return get_text(response)

        except Exception as e:

            error_message = str(e)

            print(
                "Gemini unavailable. "
                "Using Groq fallback."
            )

            print(error_message)

    response = groq_llm.invoke(prompt)

    return get_text(response)


# ============================================================
# 9. PLANNER AGENT
# ============================================================

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research planning agent.

Your job is to break the user's research question
into exactly 5 specific research tasks.

Do NOT answer the research question.

Only create a clear research plan.

Return exactly 5 numbered research tasks.
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

planner_chain = planner_prompt | groq_llm


def planner_node(state: ResearchState):

    response = planner_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "research_plan": get_text(response)
    }


# ============================================================
# 10. RESEARCHER AGENT
# ============================================================

def researcher_node(state: ResearchState):

    question = state["question"]

    plan = state["research_plan"]

    # Ask Tavily to find current information.
    search_queries = [
        question,
        f"{question} causes",
        f"{question} effects",
        f"{question} statistics",
        f"{question} solutions",
    ]

    all_results = []

    for query in search_queries:

        try:

            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=4,
            )

            results = response.get("results", [])

            for result in results:

                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")

                if content:

                    all_results.append(
                        f"TITLE: {title}\n"
                        f"SOURCE: {url}\n"
                        f"CONTENT: {content}"
                    )

        except Exception as e:

            all_results.append(
                f"Tavily search error for '{query}': {e}"
            )

    # Remove duplicates.
    unique_results = []

    seen = set()

    for result in all_results:

        key = result[:300]

        if key not in seen:

            seen.add(key)
            unique_results.append(result)

    # Limit the amount of text sent to the model.
    unique_results = unique_results[:15]

    return {
        "research_results": unique_results,
        "research_attempts": state.get(
            "research_attempts",
            0
        ) + 1,
    }


# ============================================================
# 11. ANALYST AGENT
# ============================================================

def analyst_node(state: ResearchState):

    research_text = "\n\n".join(
        state["research_results"]
    )

    prompt = f"""
You are an expert research analyst.

Research question:
{state["question"]}

Research plan:
{state["research_plan"]}

Collected research:
{research_text}

Analyze the information.

Identify:
1. Main findings
2. Important patterns
3. Benefits or positive effects
4. Risks or negative effects
5. Important statistics
6. Areas where evidence is uncertain

Do not invent facts.

Clearly distinguish evidence from assumptions.
"""

    analysis = call_gemini(prompt)

    return {
        "analysis": analysis
    }


# ============================================================
# 12. FACT-CHECK AGENT
# ============================================================

def fact_check_node(state: ResearchState):

    research_text = "\n\n".join(
        state["research_results"]
    )

    prompt = f"""
You are a fact-checking research agent.

Research question:
{state["question"]}

Analysis:
{state["analysis"]}

Source material:
{research_text}

Check the analysis for:

- unsupported claims
- exaggerated claims
- contradictions
- missing context
- statistics that need caution
- claims that appear well supported

Do not invent corrections.

Return a concise fact-check summary.
"""

    response = groq_llm.invoke(prompt)

    return {
        "fact_check": get_text(response)
    }


# ============================================================
# 13. FINAL REPORT AGENT
# ============================================================

def final_report_node(state: ResearchState):

    research_text = "\n\n".join(
        state["research_results"]
    )

    prompt = f"""
You are the final research report writer.

Create a clear, professional research report.

Research question:
{state["question"]}

Research plan:
{state["research_plan"]}

Research findings:
{research_text}

Analysis:
{state["analysis"]}

Fact check:
{state["fact_check"]}

Write the final report using this structure:

# Research Report

## Introduction

## Key Findings

## Positive Effects / Benefits

## Negative Effects / Risks

## Evidence and Statistics

## Fact-Check / Limitations

## Conclusion

## Sources

Use only information supported by the research.

Do not invent citations.

Keep the writing clear and suitable for a student or general research audience.
"""

    final_report = call_gemini(prompt)

    return {
        "final_report": final_report
    }


# ============================================================
# 14. LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(ResearchState)


graph.add_node(
    "planner",
    planner_node
)

graph.add_node(
    "researcher",
    researcher_node
)

graph.add_node(
    "analyst",
    analyst_node
)

graph.add_node(
    "fact_checker",
    fact_check_node
)

graph.add_node(
    "final_report",
    final_report_node
)


# ============================================================
# 15. GRAPH CONNECTIONS
# ============================================================

graph.add_edge(
    START,
    "planner"
)

graph.add_edge(
    "planner",
    "researcher"
)

graph.add_edge(
    "researcher",
    "analyst"
)

graph.add_edge(
    "analyst",
    "fact_checker"
)

graph.add_edge(
    "fact_checker",
    "final_report"
)

graph.add_edge(
    "final_report",
    END
)


# ============================================================
# 16. COMPILE GRAPH
# ============================================================

research_graph = graph.compile()


# ============================================================
# 17. TEST FUNCTION
# ============================================================

def run_research(question: str):

    initial_state = {
        "question": question,
        "research_plan": "",
        "research_results": [],
        "analysis": "",
        "fact_check": "",
        "final_report": "",
        "research_attempts": 0,
    }

    return research_graph.invoke(
        initial_state
    )


# ============================================================
# 18. COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What are the effects of artificial "
        "intelligence on education?"
    )

    result = run_research(question)

    print("\n")
    print("=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print("\n")

    print(
        result["final_report"]
    )
