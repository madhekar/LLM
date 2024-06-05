#!/usr/bin/env python
# coding: utf-8

# # L3: Large Multimodal Models (LMMs)

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
# ```
#     !pip install google-generativeai
# 
# ```
# 
# Note: don't forget to set up your GOOGLE_API_KEY to use the Gemini Vision model in the env file.
# ```
#    %env GOOGLE_API_KEY=************
# ```
# Check the [documentation](https://ai.google.dev/gemini-api/docs/api-key) for more infomation.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ## Setup
# ### Load environment variables and API keys

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')


# In[ ]:


# Set the genai library
import google.generativeai as genai
from google.api_core.client_options import ClientOptions

genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest",
        client_options=ClientOptions(
            api_endpoint=os.getenv("GOOGLE_API_BASE"),
        ),
)


# > Note: learn more about [GOOGLE_API_KEY](https://ai.google.dev/) to run it locally.

# ## Helper functions

# In[ ]:


import textwrap
import PIL.Image
from IPython.display import Markdown, Image

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# * Function to call LMM (Large Multimodal Model).

# In[ ]:


def call_LMM(image_path: str, prompt: str) -> str:
    # Load the image
    img = PIL.Image.open(image_path)

    # Call generative model
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    return to_markdown(response.text)  


# ## Analyze images with an LMM

# In[ ]:


# Pass in an image and see if the LMM can answer questions about it
Image(url= "SP-500-Index-Historical-Chart.jpg")


# In[ ]:


# Use the LMM function
call_LMM("SP-500-Index-Historical-Chart.jpg", 
    "Explain what you see in this image.")


# ## Analyze a harder image

# * Try something harder: Here's a figure we explained previously!

# In[ ]:


Image(url= "clip.png")


# In[ ]:


call_LMM("clip.png", 
    "Explain what this figure is and where is this used.")


# ## Decode the hidden message

# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Utils File and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>
# 

# In[ ]:


Image(url= "blankimage3.png")


# In[ ]:


# Ask to find the hidden message
call_LMM("blankimage3.png", 
    "Read what you see on this image.")


# ## How the model sees the picture!

# > You have to be careful! The model does not "see" in the same way that we see!

# In[ ]:


import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

image = imageio.imread("blankimage3.png")

# Convert the image to a NumPy array
image_array = np.array(image)

plt.imshow(np.where(image_array[:,:,0]>120, 0,1), cmap='gray');


# ### Try it yourself!

# **EXTRA!**  You can use the function below to create your own hidden message, into an image:

# In[ ]:


# Create a hidden text in an image
def create_image_with_text(text, font_size=20, font_family='sans-serif', text_color='#73D955', background_color='#7ED957'):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(background_color)
    ax.text(0.5, 0.5, text, fontsize=font_size, ha='center', va='center', color=text_color, fontfamily=font_family)
    ax.axis('off')
    plt.tight_layout()
    return fig


# In[ ]:


# Modify the text here to create a new hidden message image!
fig = create_image_with_text("Hello, world!") 

# Plot the image with the hidden message
plt.show()
fig.savefig("extra_output_image.png")


# In[ ]:


# Call the LMM function with the image just generated
call_LMM("extra_output_image.png", 
    "Read what you see on this image.")


# * It worked!, now plot the image decoding the message.

# In[ ]:


image = imageio.imread("extra_output_image.png")

# Convert the image to a NumPy array
image_array = np.array(image)

plt.imshow(np.where(image_array[:,:,0]>120, 0,1), cmap='gray');

