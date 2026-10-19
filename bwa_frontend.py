import streamlit as st
import os
import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# Tavily search tool
search_tool = TavilySearchResults(max_results=3)


# -------- STATE --------

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[str]
    draft: str
    recency_days: int


# -------- PROMPTS --------

ROUTER_SYSTEM = """
You are a routing AI.

Decide:
1) If research is needed
2) What search queries should be used

Respond ONLY in JSON format:

{
"mode": "closed_book | hybrid | open_book",
"needs_research": true,
"queries": ["query1","query2"]
}
"""

WRITER_SYSTEM = """
You are a professional blog writer.

Write a detailed blog post using the topic and evidence provided.
"""


# -------- NODES --------

def router_node(state: State):

    prompt = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}")
    ]

    response = llm.invoke(prompt).content

    try:
        decision = json.loads(response)
    except:
        decision = {
            "mode": "hybrid",
            "needs_research": True,
            "queries": [state["topic"]]
        }

    mode = decision.get("mode", "hybrid")
    needs_research = decision.get("needs_research", True)
    queries = decision.get("queries", [state["topic"]])

    if isinstance(needs_research, str):
        needs_research = needs_research.lower() == "true"

    if mode == "open_book":
        recency_days = 7
    elif mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "mode": mode,
        "needs_research": needs_research,
        "queries": queries,
        "recency_days": recency_days
    }


def research_node(state: State):

    evidence = []

    for q in state["queries"]:
        results = search_tool.invoke(q)

        for r in results:
            evidence.append(r["content"])

    return {
        "evidence": evidence
    }


def writer_node(state: State):

    evidence_text = "\n".join(state["evidence"])

    prompt = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(
            content=f"""
Topic: {state['topic']}

Evidence:
{evidence_text}

Write a high quality blog article.
"""
        )
    ]

    response = llm.invoke(prompt)

    return {
        "draft": response.content
    }


# -------- GRAPH --------

builder = StateGraph(State)

builder.add_node("router", router_node)
builder.add_node("research", research_node)
builder.add_node("writer", writer_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda x: "research" if x["needs_research"] else "writer"
)

builder.add_edge("research", "writer")
builder.add_edge("writer", END)

app = builder.compile()


# -------- STREAMLIT UI --------

st.title("AI Blog Writing Agent")

topic = st.text_input("Enter Blog Topic")

if st.button("Generate Blog"):

    with st.spinner("Generating..."):

        result = app.invoke({
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "draft": "",
            "recency_days": 365
        })

        st.subheader("Generated Blog")
        st.write(result["draft"])