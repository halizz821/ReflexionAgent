from graph.graph import reflexion_graph
from langchain_core.messages import HumanMessage
from IPython.display import display, Image


def main():

    display(Image(reflexion_graph.get_graph().draw_mermaid_png()))

    response = reflexion_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What's the Difference Between Good and Bad Carbs?"
                )
            ]
        }
    )

    agent_answer = response["messages"][-1].tool_calls[0]["args"]["answer"]
    ref = response["messages"][-1].tool_calls[0]["args"]["references"]
    print(f"Here is the agent answer:\n{agent_answer} \n\n References:\n{ref}")


if __name__ == "__main__":
    main()
