from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import END, StateGraph, add_messages
from IPython.display import display, Image
from langchain_openai import ChatOpenAI
from graph.schema import AnswerQuestion, ReviseAnswer
from graph.prompts import actor_prompt_template, REVISE_INSTRUCTION
from graph.execute_tools import execute_tools

load_dotenv()


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # Use list[BaseMessage] when your state is truly a message list — it’s clearer, safer, and matches LangGraph’s own examples.


graph = StateGraph(State)
# %% LLM initiate
llm = ChatOpenAI(model="gpt-4o")

first_responder_llm = llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
revisior_llm = llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# %% Nodes Definition
def first_responder_node(state: State):

    prompt = actor_prompt_template.format(
        first_instruction="Provide a detailed ~250 word answer",
        messages=state["messages"],
    )
    resp = first_responder_llm.invoke(prompt)
    return {"messages": [resp]}


def exe_tools(state: State):
    resp = execute_tools(state["messages"])
    return {"messages": resp}  # because resp is already a list we didnt use [resp]


def revisor_node(state: State):

    prompt = actor_prompt_template.format(
        first_instruction=REVISE_INSTRUCTION,
        messages=state["messages"],
    )
    resp = revisior_llm.invoke(prompt)
    return {"messages": [resp]}


MAX_ITERATIONS = 1  # How manytimes revision conducts


def event_loop(state: State) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    # checks whether the variable item is an instance of the class ToolMessage, then summation to see how manytimes there is a toolmessage
    # the number of tool masseges means how  many times revision happend

    if count_tool_visits >= MAX_ITERATIONS:
        return "end"
    return "more_search"


# %% Bulding Graph
graph.add_node("draft", first_responder_node)
graph.add_node("execute_tools", exe_tools)
graph.add_node("revisor", revisor_node)


graph.set_entry_point("draft")

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")
graph.add_conditional_edges(
    "revisor", event_loop, {"end": END, "more_search": "execute_tools"}
)

reflexion_graph = graph.compile()

# display(Image(reflexion_graph.get_graph().draw_mermaid_png()))
# # app.invoke()

# response = reflexion_graph.invoke(
#     {"messages": [HumanMessage(content="AI Agents taking over content creation")]}
# )

# print(first_responder_node)
