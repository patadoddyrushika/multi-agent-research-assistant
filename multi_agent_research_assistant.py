import os
from typing import TypedDict, List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


# ============================================================
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")


# ============================================================
# TWO GROQ MODELS
# ============================================================

planner_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)

research_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
)


# ============================================================
# TAVILY
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# STATE
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
# HELPER
# ============================================================

def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)

        return "\n".join(parts)

    return str(content)


# ============================================================
# PLANNER
# ============================================================

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the planning agent.

Create a research plan for the user's question.

Return exactly:

RESEARCH PLAN:
- Step 1: ...
- Step 2: ...
- Step 3: ...

SEARCH QUERIES:
1. ...
2. ...
3. ...

Do not answer the question."""
        ),
        (
            "human",
            "Research question: {question}"
        ),
    ]
)


def planner_node(state: ResearchState):

    response = planner_llm.invoke(
        planner_prompt.format_messages(
            question=state["question"]
        )
    )

    text = get_text(response)

    queries = []

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("1."):
            queries.append(line[2:].strip())

        elif line.startswith("2."):
            queries.append(line[2:].strip())

        elif line.startswith("3."):
            queries.append(line[2:].strip())

    if len(queries) < 3:

        queries = [
            state["question"],
            f"{state['question']} evidence studies",
            f"{state['question']} benefits risks statistics",
        ]

    return {
        "question": state["question"],
        "research_plan": text,
        "search_queries": queries[:3],
        "research_results": [],
        "analysis": "",
        "fact_check": "",
        "final_report": "",
    }


# ============================================================
# RESEARCHER
# ============================================================

def researcher_node(state: ResearchState):

    all_results = []

    for query in state["search_queries"]:

        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=2,
        )

        for result in response.get("results", []):

            content = result.get("content", "")

            # Keep each source small so Groq does not exceed TPM limits
            content = content[:1000]

            all_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": content,
                }
            )

    # Keep at most 6 sources
    all_results = all_results[:6]

    return {
        **state,
        "research_results": all_results,
    }


# ============================================================
# ANALYST
# ============================================================

analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the research analysis agent.

Analyze the supplied sources.

Identify:
- major findings
- important evidence
- benefits
- risks
- limitations
- disagreements between sources

Do not invent facts.
Use only the supplied sources."""
        ),
        (
            "human",
            """Question:
{question}

Research plan:
{research_plan}

Sources:
{research_results}"""
        ),
    ]
)


def analyst_node(state: ResearchState):

    # Keep the input safely below the model's request limit
    research_text = str(state["research_results"])[:5000]

    response = planner_llm.invoke(
        analysis_prompt.format_messages(
            question=state["question"],
            research_plan=state["research_plan"][:2500],
            research_results=research_text,
        )
    )

    return {
        **state,
        "analysis": get_text(response),
    }


# ============================================================
# FACT CHECKER
# ============================================================

fact_check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a fact-checking agent.

Compare the analysis with the supplied sources.

For the important claims:
- say whether they are Supported, Partially Supported, or Not Supported
- briefly explain why
- identify weak evidence

Do not invent evidence."""
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


def fact_checker_node(state: ResearchState):

    research_text = str(state["research_results"])[:4500]

    response = research_llm.invoke(
        fact_check_prompt.format_messages(
            question=state["question"],
            analysis=state["analysis"][:3500],
            research_results=research_text,
        )
    )

    return {
        **state,
        "fact_check": get_text(response),
    }


# ============================================================
# WRITER
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the final research report writer.

Write a professional research report.

Use these sections:

# Research Report

## Executive Summary

## Key Findings

## Detailed Analysis

## Fact Check

## Limitations

## Conclusion

## Sources

Rules:
- Use only information supplied in the sources.
- Do not invent statistics.
- Do not invent sources.
- Include the source URLs provided.
- Keep the report clear and concise."""
        ),
        (
            "human",
            """Question:
{question}

Research:
{research_results}

Analysis:
{analysis}

Fact check:
{fact_check}"""
        ),
    ]
)


def writer_node(state: ResearchState):

    # IMPORTANT:
    # Limit the total input sent to the 120B model.
    research_text = str(state["research_results"])[:3500]
    analysis_text = state["analysis"][:2500]
    fact_check_text = state["fact_check"][:2000]

    response = research_llm.invoke(
        writer_prompt.format_messages(
            question=state["question"][:1000],
            research_results=research_text,
            analysis=analysis_text,
            fact_check=fact_check_text,
        )
    )

    return {
        **state,
        "final_report": get_text(response),
    }


# ============================================================
# LANGGRAPH
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
