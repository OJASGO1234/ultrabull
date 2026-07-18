from typing import Literal, Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Your existing Groww tools
from groww.get_instrument import get_instrument_by_groww_symbol
from groww.get_holdings import get_holdings_for_user

# ---- NEW: quotes + calc tools ----
@tool
def get_quotes_for_symbols(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch latest LTP for the given Groww trading symbols.
    Args:
        symbols: e.g. ["CAMS","TCS"]
    Returns:
        {"CAMS": 4125.25, "TCS": 3972.10}
    """
    # TODO: swap with your Groww quotes API
    # Raise if symbols empty to keep LLM honest
    if not symbols:
        raise ValueError("symbols must be non-empty")
    # stub
    print(f"get_quotes_for_symbols called for: {symbols}")
    return {s: 4125.25 for s in symbols}

@tool
def compute_pnl(holdings: List[Dict[str, Any]], quotes: Dict[str, float]) -> Dict[str, Any]:
    """
    Compute absolute and percentage P&L per holding and overall.
    holdings item shape:
        {"symbol": "CAMS", "qty": 164, "avg_price": 4056.06}
    quotes: {"CAMS": 4125.25}
    Returns:
        {
          "per_symbol": [
            {"symbol":"CAMS","qty":164,"avg_price":4056.06,"ltp":4125.25,
             "pnl_abs":11346. , "pnl_pct":0.0169}
          ],
          "totals": {"cost": ..., "market_value": ..., "pnl_abs": ..., "pnl_pct": ...}
        }
    """
    print("compute_pnl called")
    per = []
    cost = 0.0
    mv = 0.0
    for h in holdings:
        sym = h["symbol"]
        qty = float(h["qty"])
        avg = float(h["avg_price"])
        ltp = float(quotes.get(sym))
        cost += avg * qty
        mv += ltp * qty
        pnl_abs = (ltp - avg) * qty
        pnl_pct = (ltp / avg - 1.0) if avg else 0.0
        per.append({
            "symbol": sym, "qty": qty, "avg_price": avg, "ltp": ltp,
            "pnl_abs": pnl_abs, "pnl_pct": pnl_pct
        })
    totals = {
        "cost": cost,
        "market_value": mv,
        "pnl_abs": mv - cost,
        "pnl_pct": (mv / cost - 1.0) if cost else 0.0
    }
    return {"per_symbol": per, "totals": totals}

# ------------------------------
# Models
# ------------------------------
# Recommend a stronger model than gpt-3.5 for tool use & math.
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
final_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# System prompt that forces tool-first behavior + context reuse
SYSTEM = SystemMessage(content="""\
You are a trading assistant for the Indian market.

CRITICAL:
- Reuse context from this thread. If the user asks "Am I in profit?" and prior turns
  include their holdings or symbols, do NOT ask again; use tools to fetch/compute.
- Never invent numbers, prices, or holdings. Only present figures returned by tools.
- Use tools in this sequence when relevant:
  1) get_holdings_for_user  -> fetch current holdings (no arguments)
  2) get_instrument_by_groww_symbol (only if mapping needed)
  3) get_quotes_for_symbols -> fetch LTPs for those symbols
  4) compute_pnl            -> compute per-symbol and overall P&L
- Return a concise, tabular summary with totals and % returns.
- If live quotes are unavailable, say so plainly and do not estimate.
- If absolutely missing key info, ask only one, specific follow-up.
 - Do NOT ask for confirmation for obvious requests like "What does my portfolio look like today?" or
   "Am I in profit?". Proceed to call the tools. If user answers "yes"/"please do" to a prior question,
   treat that as confirmation and proceed immediately without re-asking.
""")

# Bind tools for the AGENT model
tools = [get_instrument_by_groww_symbol, get_holdings_for_user, get_quotes_for_symbols, compute_pnl]
model = model.bind_tools(tools)

# Keep final model plain—no comedic rewrite
final_model = final_model.with_config(tags=["final_node"])

tool_node = ToolNode(tools=tools)

# ------------------------------
# Router
# ------------------------------
def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "final"

# ------------------------------
# Nodes
# ------------------------------
def call_model(state: MessagesState):
    # Add a small, programmatic nudge to avoid unnecessary confirmations and kick off tools.
    user_text = ""
    for m in reversed(state["messages"]):
        if m.__class__.__name__ == "HumanMessage":
            user_text = (m.content or "").lower()
            break
    proceed_keywords = ["portfolio", "profit", "p&l", "holdings", "today"]
    affirmations = ["yes", "y", "yeah", "yep", "please", "sure", "go ahead", "do it", "ok"]
    extra_system = None
    if any(k in user_text for k in proceed_keywords) or user_text.strip() in affirmations:
        extra_system = SystemMessage(content=(
            "The user intent is clear. Do not ask for confirmation. "
            "Immediately call tools in order: get_holdings_for_user -> get_quotes_for_symbols -> compute_pnl, "
            "then summarize results."
        ))
    messages = [SYSTEM] + ([extra_system] if extra_system else []) + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def call_final_model(state: MessagesState):
    messages = state["messages"]
    # Feed the full conversation (including tool messages) to ground the answer.
    response = final_model.invoke([SYSTEM] + messages)
    # Preserve threading by reusing the last message id when present
    for m in reversed(messages):
        if getattr(m, "id", None):
            response.id = m.id
            break
    return {"messages": [response]}

# ------------------------------
# Graph
# ------------------------------
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
builder.add_edge("final", END)

# ✅ Add a checkpointer so the thread has memory across turns
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
