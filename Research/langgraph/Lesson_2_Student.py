#!/usr/bin/env python
# coding: utf-8

# # Lesson 2 : LangGraph Components

# In[ ]:


from dotenv import load_dotenv
_ = load_dotenv()


# In[ ]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


# In[ ]:


tool = TavilySearchResults(max_results=4) #increased number of results
print(type(tool))
print(tool.name)


# > If you are not familiar with python typing annotation, you can refer to the [python documents](https://docs.python.org/3/library/typing.html).

# In[ ]:


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# In[ ]:


class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


# In[ ]:


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4-turbo")
abot = Agent(model, [tool], system=prompt)


# In[ ]:


from IPython.display import Image

Image(abot.graph.get_graph().draw_png())


# In[ ]:


messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})


# In[ ]:


result


# In[ ]:


result['messages'][-1].content


# In[ ]:


messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})


# In[ ]:


result['messages'][-1].content


# In[ ]:


# Note, the query was modified to produce more consistent results. 
# Results may vary per run and over time as search information and models change.
query = "Who won the super bowl in 2024? What is the GDP of state where the winning team is located?" 
messages = [HumanMessage(content=query)]
result = abot.graph.invoke({"messages": messages})


# In[ ]:


result['messages'][-1].content


# > Interresting side note. If you look through the information returned by the search engine, `print(result['messages'])`, you may find the search results do not mention the state that the Kansas City Chiefs are located in. This information is then drawn from the LLM intrinsic knowledge. This is not completely trivial as Kansas City is in both Missouri and Kansas. 

# In[ ]:




