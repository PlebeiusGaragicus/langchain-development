
class Colors():
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7

def cprint(string: str, color: Colors, end='\n'):
    print_this = f'\033[1;3{color}m' + string + '\033[0m'
    print(print_this, end=end)

def cput(string: str, color: Colors):
    print_this = f'\033[1;3{color}m' + string + '\033[0m'
    print(print_this, end='', flush=True)


def change_color(color: Colors):
    print(f'\033[1;3{color}m')

def reset_color():
    print('\033[0m')












import os
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# from langchain_openai import ChatOpenAI




# workflow = StateGraph(GraphState)

class State(TypedDict):
    messages: Annotated[list, add_messages]












def route_question(state: State, config):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    cprint("\n--- NODE: route_question() ---", Colors.MAGENTA)
    # convo_history = state["messages"][:-1]
    # question = state["messages"][-1].content
    messages = '\n'.join(f"{message.type}: {message.content}" for message in state["messages"])
    # print(convo_history)


    model = "llama3"
    llm = ChatOllama(model=model, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to the next appropriate workflow:\n
Use 'vectorstore' for questions related to firefighting operations, policies, terminology or knowledge.\n
Use 'friendly_chatbot' for all other user questions or replies.\n
You do not need to be stringent with the keywords in the question related to these topics.\n
Give a single choice based on the question and context of the conversation.\n
Return JSON with a single key 'datasource' and no premable or explaination.
---
Conversation History:\n
{messages}
""",
        # input_variables=["convo_history", "question"],
        input_variables=["messages"],
    )
# Conversation history: {convo_history} \n
# User question: {question}

    # prompt.pretty_print()
    # cprint(prompt.pretty_repr().format(convo_history=convo_history, question=question), Colors.CYAN)
    cprint(prompt.pretty_repr().format(messages=messages), Colors.CYAN)
    # print(prompt.pretty_repr())
    # cprint(f"Convo History: {convo_history}", Colors.RED)
    # cprint(f"Question: {question}", Colors.RED)

    question_router = prompt | llm | JsonOutputParser()

    # source = question_router.invoke({"convo_history": convo_history, "question": question}, config=config)
    source = question_router.invoke({"messages": messages}, config=config)

    if source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO vectorstore---")
        return "vectorstore"
    elif source['datasource'] == 'friendly_chatbot':
        print("---ROUTE QUESTION TO friendly chatbot---")
        return "friendly_chatbot"
    else:
        print("---ROUTING ERROR---")
        return "bad_route"





def friendly_chatbot(state: State, config):
    cprint("\n--- NODE: friendly_chatbot() ---", Colors.MAGENTA)

    model="llama3" # TODO - pull the model, temperature, etc from the config!
    llm = ChatOllama(model=model, temperature=0.8)

    # user_input = state["messages"][-1].content
    # convo_history = state["messages"][:-1]
    # user_input = state["messages"][-1].content
    # messages = state["messages"]

    # format the message history for the prompt
    messages = '\n'.join(f"{message.type}: {message.content}" for message in state["messages"])


            # Keep replies short, don't use proper grammar or punctuation.\n
    prompt = PromptTemplate(
            template="""You are the human's friend.\n
Keep the following conversation going.\n
Keep replies short.\n
---\n
Conversation History: \n
{messages}""",
            input_variables=["messages"],
        )
            # Conversation history: {messages}\n
            # Your friend said: {user_input}""",
            # input_variables=["user_input", "messages"],
        
    cprint(prompt.pretty_repr().format(messages=messages), Colors.CYAN)

    chain = prompt | llm

    # return {"messages": [chain.invoke({"user_input": user_input, "messages": convo_history}, config=config)]}
    return {"messages": [chain.invoke({"messages": messages}, config=config)]}
        # state["messages"], config=config)]}


def vectorstore(state: State, config):
    cprint("\n--- NODE: vectorstore() ---", Colors.MAGENTA)

    return {"messages": [AIMessage(content="THE DATABASE IS NOT YET IMPLEMENTED!")]}


def bad_route(state: State, config):
    cprint("\n--- NODE: bad_route() ---", Colors.MAGENTA)
    cprint("### THIS SHOULD NEVER HAPPEN!!! ###", Colors.RED)

    return {"messages": [AIMessage(content="I'm sorry, I don't understand that question.")]}


graph_builder = StateGraph(State)

graph_builder.add_node("vectorstore", vectorstore)
graph_builder.add_edge("vectorstore", END)
graph_builder.add_node("friendly_chatbot", friendly_chatbot)
graph_builder.add_edge("friendly_chatbot", END)
graph_builder.add_node("bad_route", bad_route)
graph_builder.add_edge("bad_route", END)

graph_builder.set_conditional_entry_point(
                route_question,
                {
                    "vectorstore": "vectorstore",
                    "friendly_chatbot": "friendly_chatbot",
                    "bad_route": "bad_route"
                }
            )


graph = graph_builder.compile()
























async def main():
# def main():

    convo_history = []
    while True:
        if len(convo_history) > 0:
            cprint("\nConversation History:", Colors.RED)
            for message in convo_history:
                cput(f"{message.type}: ", Colors.RED)
                cput(f"{message.content}\n", Colors.MAGENTA)

        # cprint("\nUser Question:", Colors.RED, end=" ")
        change_color(Colors.YELLOW)
        try:
            # user_input = input(">> ")
            user_input = input("User: ")
        except KeyboardInterrupt:
            cprint("\nGoodbye!", Colors.RED)
            break

        reset_color()
        if user_input.lower() in ["quit", "exit", "q"]:
            cprint("Goodbye!", Colors.RED)
            break
        if user_input.strip() == "":
            continue

        convo_history.append(HumanMessage(content=user_input))
        graph_input = {"messages": convo_history}

        # graph_input = {"messages": [HumanMessage(content=user_input)]}
        config = {"something": "yes, yes, it is!"}

        print("INVOKE GRAPH WITH <>CONVO HISTORY<>")
        for msg in convo_history:
            cput(f"{msg.type}: ", Colors.RED)
            cput(f"{msg.content}\n", Colors.MAGENTA)
            reset_color()

        last_chain_ending = ""
        async for event in graph.astream_events(
                                input=graph_input,
                                config=config,
                                version='v1'
                            ):


            if event['event'] == 'on_chat_model_stream':
                # print(event)
                # NOTE: I can't do this or it will skip single spaces in the bot's output!!
                # if event['data']['chunk'].content.strip() == "":
                #     continue
                chunk = event['data']['chunk'].content
                cput(chunk, Colors.GREEN)

            elif event['event'] == 'on_chain_end':
                # cprint(event['event'], Colors.YELLOW)
                try:
                    # TODO - this tries to get the penultimate graph output - this will break as my graph changes!
                    # last_chain_ending = event['data']["output"][event['name']]['messages'][0].content
                    last_chain_ending = event['data']["output"]['messages'][0].content
                except (KeyError, TypeError):
                    cprint("ERROR", Colors.RED)
                    pass
            else:
                # cprint(event['event'], Colors.YELLOW)
                # print(event)
                pass

            # print(event)
        convo_history.append(AIMessage(content=last_chain_ending))






if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    # main()



CONFIG = """
{'tags': [], 'metadata': {}, 'callbacks': <langchain_core.callbacks.manager.CallbackManager object at 0x1040eb400>, 'recursion_limit': 25, 'something': 'yes, yes, it is!', 'configurable': {'__pregel_send': <built-in method extend of collections.deque object at 0x1040e59c0>, '__pregel_read': functools.partial(<function _local_read at 0x10206b400>, {'v': 1, 'ts': '2024-04-22T06:36:19.820879+00:00', 'channel_values': {}, 'channel_versions': defaultdict(<class 'int'>, {'__start__': 1, 'messages': 2, 'start:chatbot': 2}), 'versions_seen': defaultdict(<function _seen_dict at 0x101b6af80>, {'__start__': defaultdict(<class 'int'>, {'__start__': 1}), 'chatbot': defaultdict(<class 'int'>, {'start:chatbot': 2})})}, {'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x104076740>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104077070>, 'chatbot': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104076830>, 'start:chatbot': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104077c40>}, deque([]))}}
"""
