import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

LANGGRAPH_DIR = Path(__file__).resolve().parent
if str(LANGGRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(LANGGRAPH_DIR))

load_dotenv(LANGGRAPH_DIR / ".env", override=True)
os.environ.setdefault("LLM_RAG_DATA_DIR", str(LANGGRAPH_DIR / "data"))

from pdf_retrieval_tool import find_pdf_document_payload, retrieve_pdf_pages_payload  # noqa: E402


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Tools ──────────────────────────────────────────────────────────────────────

search_tool = TavilySearchResults(
    max_results=3,
    description="Search the web for up-to-date information. Use when the question requires recent news or facts you're uncertain about.",
)


@tool
def find_pdf_document(document_name: str, limit: int = 5) -> str:
    """Find indexed local PDF documents by fuzzy file name.

    Use this before retrieve_pdf_pages when the user mentions a file name but
    does not know the document_id. Returns JSON with matching document IDs,
    names, page counts, and index status.
    """
    return find_pdf_document_payload(document_name=document_name, limit=limit)


@tool
def retrieve_pdf_pages(
    question: str,
    document_id: Optional[int] = None,
    document_name: Optional[str] = None,
    max_pages: int = 4,
) -> str:
    """Retrieve relevant local PDF page images for a user question.

    Use this when the user asks about an indexed local PDF. Prefer document_id
    when available; otherwise pass the user's file name as document_name.
    The tool returns JSON with page numbers, OCR text snippets, PDF URL, and JPEG
    data URLs that can be passed to a multimodal model.
    """
    return retrieve_pdf_pages_payload(
        question=question,
        document_id=document_id,
        document_name=document_name,
        max_pages=max_pages,
    )


tools = [search_tool, find_pdf_document, retrieve_pdf_pages]


# ── Model ──────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "qwen3.5-35b-a3b"),
    api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "EMPTY",
    base_url=os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def call_model(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


# ── CLI loop ───────────────────────────────────────────────────────────────────

def print_trace(messages: list[BaseMessage], prev_len: int):
    """打印本轮新增的消息，展示完整调用链。"""
    new_msgs = messages[prev_len:]
    for msg in new_msgs:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [工具调用] {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content[:200].replace("\n", " ")
            print(f"  [工具返回] {preview}...")


def run():
    print(f"{os.environ.get('LLM_MODEL', 'qwen3.5-35b-a3b')} + tools Agent (type 'quit' to exit)\n")
    history: list[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        prev_len = len(history)
        result = app.invoke({"messages": history})
        history = result["messages"]

        print_trace(history, prev_len)

        # Find the last AI message (not a tool call stub)
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"\nAgent: {msg.content}\n")
                break


if __name__ == "__main__":
    run()
