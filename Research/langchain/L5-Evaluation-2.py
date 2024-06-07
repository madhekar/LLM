#!/usr/bin/env python
# coding: utf-8

# # LangChain: Evaluation
# 
# ## Outline:
# 
# * Example generation
# * Manual evaluation (and debuging)
# * LLM-assisted evaluation
# * LangChain evaluation platform

# In[1]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[2]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# ## Create our QandA application

# In[3]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch


# In[4]:


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()


# In[5]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


# In[6]:


llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)


# ### Coming up with test datapoints

# In[7]:


data[10]


# In[8]:


data[11]


# ### Hard-coded examples

# In[9]:


examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]


# ### LLM-Generated examples

# In[10]:


from langchain.evaluation.qa import QAGenerateChain


# In[11]:


example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))


# In[12]:


# the warning below can be safely ignored


# In[13]:


new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)


# In[14]:


new_examples[0]


# In[15]:


data[0]


# ### Combine examples

# In[16]:


examples += new_examples


# In[17]:


qa.run(examples[0]["query"])


# ## Manual Evaluation

# In[18]:


import langchain
langchain.debug = True


# In[19]:


qa.run(examples[0]["query"])


# In[20]:


# Turn off the debug mode
langchain.debug = False


# ## LLM assisted evaluation

# In[21]:


predictions = qa.apply(examples)


# In[22]:


from langchain.evaluation.qa import QAEvalChain


# In[23]:


llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)


# In[24]:


graded_outputs = eval_chain.evaluate(examples, predictions)


# In[25]:


for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()


# In[26]:


graded_outputs[0]


# ## LangChain evaluation platform

# The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/.  
# Use the invite code `lang_learners_2023`

# Reminder: Download your notebook to you local computer to save your work.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




