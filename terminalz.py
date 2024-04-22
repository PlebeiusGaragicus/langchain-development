
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
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


from langchain_community.chat_models import ChatOllama



class State(TypedDict):
    messages: Annotated[list, add_messages]

# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}

def chatbot(state: State, config):
    model="llama3"
    llm = ChatOllama(model=model)

    print(config)

    # output = ""
    # change_color(Colors.GREEN)
    # async for chunk in llm.astream(state["messages"], config=config):
    #     # print(chunk.content, end="", flush=True)
    #     output += chunk.content
    # reset_color()
    # return {"messages": [output]}

    return {"messages": [llm.invoke(state["messages"], config=config)]}
    # return  for chunk in llm.astream(state["messages"], config=config):



graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()









async def main():
# def main():
    while True:
        cprint("\nUser Question:", Colors.RED, end=" ")
        change_color(Colors.YELLOW)
        try:
            user_input = input(">> ")
        except KeyboardInterrupt:
            cprint("\nGoodbye!", Colors.RED)
            break

        reset_color()
        if user_input.lower() in ["quit", "exit", "q"]:
            cprint("Goodbye!", Colors.RED)
            break
        if user_input.strip() == "":
            continue


        graph_input = {"messages": [HumanMessage(content=user_input)]}

        config = {"something": "yes, yes, it is!"}

        async for event in graph.astream_events(input=graph_input, config=config, version='v1'):
            # print(event)
            if event['event'] == 'on_chat_model_stream':
                chunk = event['data']['chunk'].content
                cput(chunk, Colors.GREEN)

            else:
                cprint(event['event'], Colors.YELLOW)






if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    # main()



CONFIG = """
{'tags': [], 'metadata': {}, 'callbacks': <langchain_core.callbacks.manager.CallbackManager object at 0x1040eb400>, 'recursion_limit': 25, 'something': 'yes, yes, it is!', 'configurable': {'__pregel_send': <built-in method extend of collections.deque object at 0x1040e59c0>, '__pregel_read': functools.partial(<function _local_read at 0x10206b400>, {'v': 1, 'ts': '2024-04-22T06:36:19.820879+00:00', 'channel_values': {}, 'channel_versions': defaultdict(<class 'int'>, {'__start__': 1, 'messages': 2, 'start:chatbot': 2}), 'versions_seen': defaultdict(<function _seen_dict at 0x101b6af80>, {'__start__': defaultdict(<class 'int'>, {'__start__': 1}), 'chatbot': defaultdict(<class 'int'>, {'start:chatbot': 2})})}, {'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x104076740>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104077070>, 'chatbot': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104076830>, 'start:chatbot': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x104077c40>}, deque([]))}}
"""




# ### Router

# from langchain.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOllama
# from langchain_core.output_parsers import JsonOutputParser

# # LLM
# llm = ChatOllama(model=local_llm, format="json", temperature=0)

# prompt = PromptTemplate(
#     template="""You are an expert at routing a user question to a vectorstore or web search. \n
#     Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
#     You do not need to be stringent with the keywords in the question related to these topics. \n
#     Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
#     Return JSON with a single key 'datasource' and no premable or explaination. \n
#     Question to route: {question}""",
#     input_variables=["question"],
# )

# question_router = prompt | llm | JsonOutputParser()
# question = "llm agent memory"
# docs = retriever.get_relevant_documents(question)
# doc_txt = docs[1].page_content
# print(question_router.invoke({"question": question}))