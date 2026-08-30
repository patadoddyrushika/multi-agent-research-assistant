import os
from typing import TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


# ============================================================
# 1. API KEYS
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set.")


# ============================================================
# 2. AI MODEL + TAVILY
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# 3. RESEARCH STATE
# ============================================================

class ResearchState(TypedDict, total=False):
    question: str
    research_plan: str
    research_results: List[str]
    analysis: str
    fact_check: str
    final_report: str
    research_attempts: int


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def get_text(response) -> str:
    """Convert an LLM response into plain text."""

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
# 5. PLANNER AGENT
# ============================================================

planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research planning agent.

Your job is to break a user's research question into
5 specific research tasks.

Do NOT answer the question.

Only create a clear research plan.

Return exactly 5 numbered research tasks."""
    ),
    (
        "human",
        "{question}"
    )
])


def planner_node(state: ResearchState):

    response = llm.invoke(
        planner_prompt.format_messages(
            question=state["question"]
        )
    )

    return {
        "research_plan": get_text(response)
    }


# ============================================================
# 6. RESEARCHER AGENT
# ============================================================

query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a web research query generator.

Given a research question and research plan,
create exactly 3 useful web search queries.

The queries should cover different aspects of the topic.

Return ONLY the 3 queries, one per line.

Do not number them.
Do not add explanations."""
    ),
    (
        "human",
        """Research question:

{question}

Research plan:

{research_plan}"""
    )
])


def researcher_node(state: ResearchState):

    question = state["question"]
    research_plan = state["research_plan"]

    response = llm.invoke(
        query_prompt.format_messages(
            question=question,
            research_plan=research_plan
        )
    )

    queries_text = get_text(response)

    queries = [
        line.strip()
        for line in queries_text.splitlines()
        if line.strip()
    ]

    queries = queries[:3]

    research_results = []

    for query in queries:

        try:
            results = tavily_client.search(
                query=query,
                max_results=3
            )

            formatted_results = []

            for result in results.get("results", []):

                formatted_results.append(
                    f"TITLE: {result.get('title', '')}\n"
                    f"URL: {result.get('url', '')}\n"
                    f"CONTENT: {result.get('content', '')}"
                )

            if formatted_results:

                research_results.append(
                    f"SEARCH QUERY: {query}\n\n"
                    + "\n\n---\n\n".join(formatted_results)
                )

        except Exception as e:

            research_results.append(
                f"SEARCH QUERY: {query}\n"
                f"SEARCH ERROR: {str(e)}"
            )

    attempts = state.get("research_attempts", 0)

    return {
        "research_results": research_results,
        "research_attempts": attempts + 1
    }


# ============================================================
# 7. ANALYST AGENT
# ============================================================

analyst_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research analyst.

You will receive:

1. A research question
2. A research plan
3. Raw web research collected by another agent

Analyze the research and identify the most important findings.

For each major finding:

- Explain the finding clearly.
- Compare information from different sources when possible.
- Mention important evidence.
- Identify disagreements or limitations.
- Do not invent facts.

Do NOT write the final report yet.

Your job is only to analyze the evidence."""
    ),
    (
        "human",
        """Research Question:

{question}

Research Plan:

{research_plan}

Raw Research:

{research_results}"""
    )
])


def analyst_node(state: ResearchState):

    combined_research = "\n\n".join(
        state.get("research_results", [])
    )

    response = llm.invoke(
        analyst_prompt.format_messages(
            question=state["question"],
            research_plan=state["research_plan"],
            research_results=combined_research
        )
    )

    return {
        "analysis": get_text(response)
    }


# ============================================================
# 8. FACT CHECKER AGENT
# ============================================================

fact_checker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a careful fact-checking research agent.

You will receive:

1. The original research question
2. An analysis produced by another agent
3. The original web research and sources

Check whether the important claims in the analysis
are supported by the available evidence.

For each important claim:

- State the claim.
- Decide whether it is Supported, Partially Supported,
  or Not Supported.
- Give a short explanation.
- Mention the relevant source URL when available.
- Do not invent evidence.
- If evidence is insufficient, clearly say so.

Be objective and critical."""
    ),
    (
        "human",
        """Research Question:

{question}

Analysis:

{analysis}

Original Research:

{research_results}"""
    )
])


def fact_checker_node(state: ResearchState):

    combined_research = "\n\n".join(
        state.get("research_results", [])
    )

    response = llm.invoke(
        fact_checker_prompt.format_messages(
            question=state["question"],
            analysis=state["analysis"],
            research_results=combined_research
        )
    )

    return {
        "fact_check": get_text(response)
    }


# ============================================================
# 9. FACT CHECK DECISION
# ============================================================

def check_fact_quality(state: ResearchState):

    fact_check = state.get("fact_check", "").lower()
    attempts = state.get("research_attempts", 0)

    # Allow at most one additional research pass.
    if attempts < 2:
        if (
            "not supported" in fact_check
            or "partially supported" in fact_check
        ):
            return "researcher"

    return "writer"


# ============================================================
# 10. WRITER AGENT
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert academic research report writer.

Write a clear, well-structured research report using
only the research evidence, analysis, and fact-checking
results provided.

The report must contain:

1. Title
2. Introduction
3. Key Findings
4. Detailed Analysis
5. Ethical and Social Considerations
6. Limitations
7. Conclusion
8. Sources

Important rules:

- Use only information supported by the provided research.
- Do not invent statistics, studies, or sources.
- Clearly distinguish evidence from interpretation.
- Use a professional academic tone.
- Keep the report readable and well organized.
- Include source URLs where appropriate.
- Do not mention that you are an AI agent."""
    ),
    (
        "human",
        """Research Question:

{question}

Research Plan:

{research_plan}

Research Evidence:

{research_results}

Analysis:

{analysis}

Fact Check:

{fact_check}

Write the final research report."""
    )
])


def writer_node(state: ResearchState):

    combined_research = "\n\n".join(
        state.get("research_results", [])
    )

    response = llm.invoke(
        writer_prompt.format_messages(
            question=state["question"],
            research_plan=state["research_plan"],
            research_results=combined_research,
            analysis=state["analysis"],
            fact_check=state["fact_check"]
        )
    )

    return {
        "final_report": get_text(response)
    }


# ============================================================
# 11. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("fact_checker", fact_checker_node)
graph.add_node("writer", writer_node)


# Workflow order

graph.add_edge(START, "planner")

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


# Fact checker decides whether to
# research again or write the report.

graph.add_conditional_edges(
    "fact_checker",
    check_fact_quality,
    {
        "researcher": "researcher",
        "writer": "writer"
    }
)


graph.add_edge(
    "writer",
    END
)


# Compile the graph

research_graph = graph.compile()


# ============================================================
# 12. OPTIONAL COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What are the effects of artificial intelligence "
        "on education?"
    )

    result = research_graph.invoke(
        {
            "question": question,
            "research_plan": "",
            "research_results": [],
            "analysis": "",
            "fact_check": "",
            "final_report": "",
            "research_attempts": 0
        }
    )

    print("\n")
    print("=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print("\n")

    print(result["final_report"])
