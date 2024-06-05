#!/usr/bin/env python
# coding: utf-8

# # L5: Industry Applications

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
# ```
#     !pip install google-generativeai
# ```

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

import google.generativeai as genai
from google.api_core.client_options import ClientOptions
genai.configure(
    api_key=GOOGLE_API_KEY,
    transport="rest",
    client_options=ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)


# > Note: learn more about [GOOGLE_API_KEY](https://ai.google.dev/) to run it locally.

# ## Vision Function

# In[ ]:


import textwrap
import PIL.Image
from IPython.display import Markdown, Image

def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

def call_LMM(image_path: str, prompt: str, plain_text: bool=False) -> str:
    img = PIL.Image.open(image_path)

    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    
    if(plain_text):
        return response.text
    else:
        return to_markdown(response.text)


# ## Extracting Structured Data from Retreived Images
# ### Analyzing an invoice

# In[ ]:


from IPython.display import Image

Image(url="invoice.png")


# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Files and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>
# 

# In[ ]:


call_LMM("invoice.png",
    """Identify items on the invoice, Make sure you output 
    JSON with quantity, description, unit price and ammount.""")


# In[ ]:


# Ask something else
call_LMM("invoice.png",
    """How much would four sets pedal arms cost
    and 6 hours of labour?""",
    plain_text=True
)


# ### Extracting Tables from Images

# In[ ]:


Image("prosus_table.png")


# In[ ]:


call_LMM("prosus_table.png", 
    "Print the contents of the image as a markdown table.")


# In[ ]:


call_LMM("prosus_table.png", 
    """Analyse the contents of the image as a markdown table.
    Which of the business units has the highest revenue growth?""")


# ### Analyzing Flow Charts

# In[ ]:


Image("swimlane-diagram-01.png")


# In[ ]:


call_LMM("swimlane-diagram-01.png", 
    """Provide a summarized breakdown of the flow chart in the image
    in a format of a numbered list.""")


# In[ ]:


call_LMM("swimlane-diagram-01.png", 
    """Analyse the flow chart in the image,
    then output Python code
    that implements this logical flow in one function""")


# * Test the code generate above.
# > Note: please be advised that the output may include errors or the functionality may not be fully operational, as it requires additional inputs to function properly.

# In[ ]:


def order_fulfillment(client, online_shop, courier_company):
   # This function takes three objects as input:
   # - client: the client who placed the order
   # - online_shop: the online shop that received the order
   # - courier_company: the courier company that will deliver the order

   # First, the client places an order.
   order = client.place_order()

   # Then, the client makes a payment for the order.
   payment = client.make_payment(order)

   # If the payment is successful, the order is shipped.
   if payment.status == "successful":
       online_shop.ship_order(order)
       courier_company.transport_order(order)
   
   # If the payment is not successful, the order is canceled.
   else:
       online_shop.cancel_order(order)
       client.refund_order(order)

   # Finally, the order is invoiced.
   online_shop.invoice_order(order)

