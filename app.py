import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from agent import graph

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    initial_state = {
        "messages": [HumanMessage(content=msg.content)],
        "last_stock_symbol": None
    }
    for msg, metadata in graph.stream(initial_state, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "final"
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()