#!/usr/bin/env python
# coding: utf-8

# # LangChain: Models, Prompts and Output Parsers
# 
# 
# ## Outline
# 
#  * Direct API calls to OpenAI
#  * API calls through LangChain:
#    * Prompts
#    * Models
#    * Output parsers

# ## Get your [OpenAI API Key](https://platform.openai.com/account/api-keys)

# In[1]:


#!pip install python-dotenv
#!pip install openai


# In[2]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[3]:


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


# ## Chat API : OpenAI
# 
# Let's start with a direct API calls to OpenAI.

# In[4]:


def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]


# In[5]:


get_completion("What is 1+1?")


# In[6]:


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""


# In[7]:


style = """American English \
in a calm and respectful tone
"""


# In[8]:


prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)


# In[9]:


response = get_completion(prompt)


# In[10]:


response


# ## Chat API : LangChain
# 
# Let's try how we can do the same using LangChain.

# In[11]:


#!pip install --upgrade langchain


# ### Model

# In[12]:


from langchain.chat_models import ChatOpenAI


# In[13]:


# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat


# ### Prompt template

# In[14]:


template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""


# In[15]:


from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)


# In[16]:


prompt_template.messages[0].prompt


# In[17]:


prompt_template.messages[0].prompt.input_variables


# In[18]:


customer_style = """American English \
in a calm and respectful tone
"""


# In[19]:


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""


# In[20]:


customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)


# In[21]:


print(type(customer_messages))
print(type(customer_messages[0]))


# In[22]:


print(customer_messages[0])


# In[23]:


# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)


# In[24]:


print(customer_response.content)


# In[25]:


service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""


# In[26]:


service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""


# In[27]:


service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print(service_messages[0].content)


# In[28]:


service_response = chat(service_messages)
print(service_response.content)


# ## Output Parsers
# 
# Let's start with defining how we would like the LLM output to look like:

# In[29]:


{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}


# In[30]:


customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""


# In[31]:


from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)


# In[32]:


messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)


# In[33]:


type(response.content)


# In[34]:


# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')


# ### Parse the LLM output string into a Python dictionary

# In[35]:


from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


# In[36]:


gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]


# In[37]:


output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


# In[38]:


format_instructions = output_parser.get_format_instructions()


# In[39]:


print(format_instructions)


# In[40]:


review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)


# In[41]:


print(messages[0].content)


# In[42]:


response = chat(messages)


# In[43]:


print(response.content)


# In[44]:


output_dict = output_parser.parse(response.content)


# In[45]:


output_dict


# In[46]:


type(output_dict)


# In[47]:


output_dict.get('delivery_days')


# Reminder: Download your notebook to you local computer to save your work.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




