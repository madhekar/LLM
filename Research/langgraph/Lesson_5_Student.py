#!/usr/bin/env python
# coding: utf-8

# # Lesson 5: Human in the Loop

# Note: This notebook is running in a later version of langgraph that it was filmed with. The later version has a couple of key additions:
# - Additional state information is stored to memory and displayed when using `get_state()` or `get_state_history()`.
# - State is additionally stored every state transition while previously it was stored at an interrupt or at the end.
# These change the command output slightly, but are a useful addtion to the information available.

# In[1]:


from dotenv import load_dotenv

_ = load_dotenv()


# In[2]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")


# In[3]:


from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


# In[4]:


tool = TavilySearchResults(max_results=2)


# ## Manual human approval

# In[5]:


class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


# In[6]:


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)


# In[7]:


messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)


# In[8]:


abot.graph.get_state(thread)


# In[9]:


abot.graph.get_state(thread).next


# ### continue after interrupt

# In[10]:


for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)


# In[11]:


abot.graph.get_state(thread)


# In[12]:


abot.graph.get_state(thread).next


# In[13]:


messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)


# ## Modify State
# Run until the interrupt and then modify the state.

# In[14]:


messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "3"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)


# In[15]:


abot.graph.get_state(thread)


# In[16]:


current_values = abot.graph.get_state(thread)


# In[17]:


current_values.values['messages'][-1]


# In[18]:


current_values.values['messages'][-1].tool_calls


# In[19]:


_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in Louisiana'},
  'id': _id}
]


# In[20]:


abot.graph.update_state(thread, current_values.values)


# In[21]:


abot.graph.get_state(thread)


# In[22]:


for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)


# ## Time Travel

# In[23]:


states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)


# To fetch the same state as was filmed, the offset below is changed to `-3` from `-1`. This accounts for the initial state `__start__` and the first state that are now stored to state memory with the latest version of software.

# In[24]:


to_replay = states[-3]


# In[25]:


to_replay


# In[26]:


for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)


# ## Go back in time and edit

# In[27]:


to_replay


# In[28]:


_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]


# In[29]:


branch_state = abot.graph.update_state(to_replay.config, to_replay.values)


# In[30]:


for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)


# ## Add message to a state at a given time

# In[31]:


to_replay


# In[32]:


_id = to_replay.values['messages'][-1].tool_calls[0]['id']


# In[33]:


state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search_results_json",
    content="54 degree celcius",
)]}


# In[34]:


branch_and_add = abot.graph.update_state(
    to_replay.config, 
    state_update, 
    as_node="action")


# In[35]:


for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)


# # Extra Practice

# ## Build a small graph
# This is a small simple graph you can tinker with if you want more insight into controlling state memory.

# In[36]:


from dotenv import load_dotenv

_ = load_dotenv()


# In[37]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver


# Define a simple 2 node graph with the following state:
# -`lnode`: last node
# -`scratch`: a scratchpad location
# -`count` : a counter that is incremented each step

# In[38]:


class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]


# In[39]:


def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }


# The graph goes N1->N2->N1... but breaks after count reaches 3.

# In[40]:


def should_continue(state):
    return state["count"] < 3


# In[41]:


builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")


# In[42]:


memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)


# ### Run it!
# Now, set the thread and run!

# In[43]:


thread = {"configurable": {"thread_id": str(1)}}
graph.invoke({"count":0, "scratch":"hi"},thread)


# ### Look at current state

# Get the current state. Note the `values` which are the AgentState. Note the `config` and the `thread_ts`. You will be using those to refer to snapshots below.

# In[44]:


graph.get_state(thread)


# View all the statesnapshots in memory. You can use the displayed `count` agentstate variable to help track what you see. Notice the most recent snapshots are returned by the iterator first. Also note that there is a handy `step` variable in the metadata that counts the number of steps in the graph execution. This is a bit detailed - but you can also notice that the *parent_config* is the *config* of the previous node. At initial startup, additional states are inserted into memory to create a parent. This is something to check when you branch or *time travel* below.

# ### Look at state history

# In[45]:


for state in graph.get_state_history(thread):
    print(state, "\n")


# Store just the `config` into an list. Note the sequence of counts on the right. `get_state_history` returns the most recent snapshots first.

# In[46]:


states = []
for state in graph.get_state_history(thread):
    states.append(state.config)
    print(state.config, state.values['count'])


# Grab an early state.

# In[47]:


states[-3]


# This is the state after Node1 completed for the first time. Note `next` is `Node2`and `count` is 1.

# In[48]:


graph.get_state(states[-3])


# ### Go Back in Time
# Use that state in `invoke` to go back in time. Notice it uses states[-3] as *current_state* and continues to node2,

# In[49]:


graph.invoke(None, states[-3])


# Notice the new states are now in state history. Notice the counts on the far right.

# In[50]:


thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])


# You can see the details below. Lots of text, but try to find the node that start the new branch. Notice the parent *config* is not the previous entry in the stack, but is the entry from state[-3].

# In[51]:


thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state,"\n")


# ### Modify State
# Let's start by starting a fresh thread and running to clean out history.

# In[52]:


thread2 = {"configurable": {"thread_id": str(2)}}
graph.invoke({"count":0, "scratch":"hi"},thread2)


# In[53]:


from IPython.display import Image

Image(graph.get_graph().draw_png())


# In[54]:


states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    print(state.config, state.values['count'])   


# Start by grabbing a state.

# In[55]:


save_state = graph.get_state(states2[-3])
save_state


# Now modify the values. One subtle item to note: Recall when agent state was defined, `count` used `operator.add` to indicate that values are *added* to the current value. Here, `-3` will be added to the current count value rather than replace it.

# In[56]:


save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
save_state


# Now update the state. This creates a new entry at the *top*, or *latest* entry in memory. This will become the current state.

# In[57]:


graph.update_state(thread2,save_state.values)


# Current state is at the top. You can match the `thread_ts`.
# Notice the `parent_config`, `thread_ts` of the new node - it is the previous node.

# In[58]:


for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:  #print latest 3
        break
    print(state, '\n')


# ### Try again with `as_node`
# When writing using `update_state()`, you want to define to the graph logic which node should be assumed as the writer. What this does is allow th graph logic to find the node on the graph. After writing the values, the `next()` value is computed by travesing the graph using the new state. In this case, the state we have was written by `Node1`. The graph can then compute the next state as being `Node2`. Note that in some graphs, this may involve going through conditional edges!  Let's try this out.

# In[59]:


graph.update_state(thread2,save_state.values, as_node="Node1")


# In[60]:


for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:  #print latest 3
        break
    print(state, '\n')


# `invoke` will run from the current state if not given a particular `thread_ts`. This is now the entry that was just added.

# In[61]:


graph.invoke(None,thread2)


# Print out the state history, notice the `scratch` value change on the latest entries.

# In[62]:


for state in graph.get_state_history(thread2):
    print(state,"\n")


# Continue to experiment!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




