import os
from typing import TypedDict, List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


# ============================================================
# 1. API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")


# ============================================================
# 2. TWO GROQ MODELS
# ============================================================

# Model 1: faster model for planning and analysis
planner_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)

# Model 2: stronger model for research, fact checking and writing
research_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)


# ============================================================
# 3. TAVILY
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# 4. GRAPH STATE
# ============================================================

class ResearchState(TypedDict, total=False):
    question: str
    research_plan: str
    search_queries: List[str]
    research_results: List[Dict[str, Any]]
    analysis: str
    fact_check: str
    final_report: str


# ============================================================
# 5. PLANNER
# ============================================================

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the planning agent in a multi-agent research assistant.

Your job is to create a clear research plan for the user's question.

Then create exactly 3 useful web-search queries.

Return your answer in this exact format:

RESEARCH PLAN:
- Step 1: ...
- Step 2: ...
- Step 3: ...

SEARCH QUERIES:
1. ...
2. ...
3. ...

Do not answer the research question yourself.
Focus on creating good search queries."""
        ),
        (
            "human",
            "Research question: {question}"
        ),
    ]
)


def planner_node(state: ResearchState) -> ResearchState:
    response = planner_llm.invoke(
        planner_prompt.format_messages(
            question=state["question"]
        )
    )

    text = response.content

    queries = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("1."):
            queries.append(line[2:].strip())
        elif line.startswith("2."):
            queries.append(line[2:].strip())
        elif line.startswith("3."):
            queries.append(line[2:].strip())

    # Safety fallback if the model does not format the queries perfectly
    if len(queries) < 3:
        queries = [
            state["question"],
            f"{state['question']} causes effects evidence",
            f"{state['question']} statistics studies reports",
        ]

    return {
        **state,
        "research_plan": text,
        "search_queries": queries[:3],
    }


# ============================================================
# 6. RESEARCHER
# ============================================================

def researcher_node(state: ResearchState) -> ResearchState:
    all_results = []

    for query in state["search_queries"]:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer="basic",
            include_raw_content=False,
        )

        for result in response.get("results", []):
            all_results.append(
                {
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                }
            )

    return {
        **state,
        "research_results": all_results,
    }


# ============================================================
# 7. ANALYST
# ============================================================

analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the analysis agent in a research system.

Analyze the research material provided.

Your job is to:
1. Identify the major findings.
2. Compare information across sources.
3. Identify important statistics or evidence.
4. Identify disagreements or limitations.
5. Avoid inventing information.

Use only the supplied research material."""
        ),
        (
            "human",
            """Research question:
{question}

Research plan:
{research_plan}

Research material:
{research_results}"""
        ),
    ]
)


def analyst_node(state: ResearchState) -> ResearchState:
    response = planner_llm.invoke(
        analysis_prompt.format_messages(
            question=state["question"],
            research_plan=state["research_plan"],
            research_results=str(state["research_results"]),
        )
    )

    return {
        **state,
        "analysis": response.content,
    }


# ============================================================
# 8. FACT CHECKER
# ============================================================

fact_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the fact-checking agent.

Review the analysis against the research sources.

For each major claim:
- Determine whether the sources support it.
- Flag claims that are weak or unsupported.
- Identify conflicting evidence.
- Do not invent corrections.

Return a concise fact-checking assessment."""
        ),
        (
            "human",
            """Question:
{question}

Analysis:
{analysis}

Sources:
{research_results}"""
        ),
    ]
)


def fact_checker_node(state: ResearchState) -> ResearchState:
    response = research_llm.invoke(
        fact_check_prompt.format_messages(
            question=state["question"],
            analysis=state["analysis"],
            research_results=str(state["research_results"]),
        )
    )

    return {
        **state,
        "fact_check": response.content,
    }


# ============================================================
# 9. FINAL WRITER
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the final research report writer.

Create a clear, professional research report.

The report must contain:

# Research Report

## Executive Summary

## Key Findings

## Analysis

## Fact Check

## Conclusion

## Sources

For the Sources section, list the source title and URL supplied in the research material.

Important rules:
- Do not invent sources.
- Do not invent URLs.
- Do not make claims that are contradicted by the fact check.
- Clearly distinguish evidence from interpretation.
- Use readable paragraphs and bullet points where appropriate.
- Answer the user's original question directly."""
        ),
        (
            "human",
            """Original question:
{question}

Research plan:
{research_plan}

Research sources:
{research_results}

Analysis:
{analysis}

Fact check:
{fact_check}"""
        ),
    ]
)


def writer_node(state: ResearchState) -> ResearchState:
    response = research_llm.invoke(
        writer_prompt.format_messages(
            question=state["question"],
            research_plan=state["research_plan"],
            research_results=str(state["research_results"]),
            analysis=state["analysis"],
            fact_check=state["fact_check"],
        )
    )

    return {
        **state,
        "final_report": response.content,
    }


# ============================================================
# 10. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "fact_checker")
workflow.add_edge("fact_checker", "writer")
workflow.add_edge("writer", END)

research_graph = workflow.compile()
