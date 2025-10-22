import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage

# from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch

# Create the Tavily search tool
# tavily_tool = TavilySearchResults(max_results=2)

search_tool = TavilySearch(search_depth="basic", max_results=2)


# Function to execute search queries from AnswerQuestion tool calls
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    #     state is a list of messages in the conversation — a mix of HumanMessage, AIMessage, and maybe ToolMessage.
    #     The function will return a list of ToolMessages, which represent “results from tool executions.”

    last_ai_message: AIMessage = state[-1]  # The most recent message

    # Extract tool calls from the AI message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    # If there is no tool call in the last message , there’s nothing to execute — return an empty list.

    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        # Iterate over tool calls
        # Each tool_call is a dict like:
        # {
        #   "id": "abc123",
        #   "name": "AnswerQuestion",
        #   "args": { "search_queries": ["..."] }
        # }

        if tool_call["name"] in [
            "AnswerQuestion",
            "ReviseAnswer",
        ]:  # You’re only executing those tools (ignore others if they exist).
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get(
                "search_queries", []
            )  # The AnswerQuestion schema includes search_queries, so we’ll extract that list of strings.

            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                # result = tavily_tool.invoke(query)
                result = search_tool.invoke({"query": query})

                query_results[query] = result

            # Here, for each query string, it uses the TavilySearchResults tool to perform a web search and collects all results in a dictionary.
            #    {
            #   "mental health statistics": [{"title": "...", "url": "..."}, ...],
            #   "CDC depression data": [{"title": "...", "url": "..."}, ...]
            #    }

            # Create a tool message with the results
            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )
            # tool_call_id links each tool's response to its specific invocation, ensuring correct result matching in multi-tool workflows.
            # json.dumps(query_results) converts the Python dictionary query_results into a JSON-formatted string — so it can be stored or
            # sent as plain text (e.g., inside ToolMessage.content). TLDR, the output of json.dumps(query_results) is an string

    return tool_messages
