import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from agent import graph  # import the compiled graph above

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="")

    # Only the *new* user message here; history comes from the checkpointer
    initial_state = {"messages": [HumanMessage(content=msg.content)]}

    async for event in graph.astream_events(
        initial_state, config=RunnableConfig(**config), version="v2"
    ):
        # We only want the final model's output (tagged in agent.py)
        if event.get("event") == "on_chat_model_end" and "final_node" in event.get("tags", []):
            out = event.get("data", {}).get("output")
            # Handle LangChain AIMessage outputs
            if isinstance(out, AIMessage):
                content = out.content
            else:
                # Fallbacks for possible dict/str outputs
                content = getattr(out, "content", None)
                if content is None and isinstance(out, dict):
                    content = out.get("content")

            if content:
                # If content is list/blocks, coerce to string
                if not isinstance(content, str):
                    content = str(content)
                await final_answer.stream_token(content)

    await final_answer.send()
