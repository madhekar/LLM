#!/usr/bin/env python
# coding: utf-8

# # L6: Multimodal Recommender System

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# * In this classroom, the libraries have been already installed for you.
# * If you would like to run this code on your own machine, you need to install the following:
# ```
#     !pip install -U weaviate-client
#     !pip install google-generativeai
#     !pip install openai
# ```

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ## Setup
# ### Load environment variables and API keys

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

MM_EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
TEXT_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASEURL = os.getenv("OPENAI_BASE_URL")


# ## Connect to Weaviate

# In[ ]:


import weaviate

client = weaviate.connect_to_embedded(
    version="1.24.4",
    environment_variables={
        "ENABLE_MODULES": "multi2vec-palm,text2vec-openai"
    },
    headers={
        "X-PALM-Api-Key": MM_EMBEDDING_API_KEY,
        "X-OpenAI-Api-Key": TEXT_EMBEDDING_API_KEY,
        "X-OpenAI-BaseURL": OPENAI_BASEURL
    }
)

client.is_ready()


# ## Create Multivector collection

# In[ ]:


from weaviate.classes.config import Configure, DataType, Property

# client.collections.delete("Movies")
client.collections.create(
    name="Movies",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="overview", data_type=DataType.TEXT),
        Property(name="vote_average", data_type=DataType.NUMBER),
        Property(name="release_year", data_type=DataType.INT),
        Property(name="tmdb_id", data_type=DataType.INT),
        Property(name="poster", data_type=DataType.BLOB),
        Property(name="poster_path", data_type=DataType.TEXT),
    ],

   # Define & configure the vector spaces
    vectorizer_config=[
        # Vectorize the movie title and overview – for text-based semantic search
        Configure.NamedVectors.text2vec_openai(
            name="txt_vector",                       # the name of the txt vector space
            source_properties=["title", "overview"], # text properties to be used for vectorization
        ),
        
        # Vectorize the movie poster – for image-based semantic search
        Configure.NamedVectors.multi2vec_palm(
            name="poster_vector",                    # the name of the image vector space
            image_fields=["poster"],                 # use poster property multivec vectorization
            
            project_id="semi-random-dev",
            location="us-central1",
            model_id="multimodalembedding@001",
            dimensions=1408,
        ),
    ]
)


# ## Load in data

# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Files and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>
# 

# In[ ]:


import pandas as pd
df = pd.read_json("movies_data.json")
df.head()


# ## Helper function

# In[ ]:


import base64

# Helper function to convert a file to base64 representation
def toBase64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


# ## Import text and image data

# In[ ]:


from weaviate.util import generate_uuid5

movies = client.collections.get("Movies")

with movies.batch.rate_limit(20) as batch:
    # for index, movie in df.sample(20).iterrows():
    for index, movie in df.iterrows():

        # In case you run it again - Don't import movies that are already in.
        if(movies.data.exists(generate_uuid5(movie.id))):
            print(f'{index}: Skipping insert. The movie "{movie.title}" is already in the database.')
            continue

        print(f'{index}: Adding "{movie.title}"')

        # construct the path to the poster image file
        poster_path = f"./posters/{movie.id}_poster.jpg"
        # generate base64 representation of the poster
        posterb64 = toBase64(poster_path)

        # Build the object payload
        movie_obj = {
            "title": movie.title,
            "overview": movie.overview,
            "vote_average": movie.vote_average,
            "tmdb_id": movie.id,
            "poster_path": poster_path,
            "poster": posterb64
        }

        # Add object to batch queue
        batch.add_object(
            properties=movie_obj,
            uuid=generate_uuid5(movie.id),
        )


# In[ ]:


# Check for failed objects
if len(movies.batch.failed_objects) > 0:
    print(f"Failed to import {len(movies.batch.failed_objects)} objects")
    for failed in movies.batch.failed_objects:
        print(f"e.g. Failed to import object with error: {failed.message}")
else:
    print("Import complete with no errors")


# ## Text-search through the text vector space

# In[ ]:


from IPython.display import Image

response = movies.query.near_text(
    query="Movie about lovable cute pets",
    target_vector="txt_vector",  # Search in the txt_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    print(item.properties["overview"])
    display(Image(item.properties["poster_path"], width=200))


# In[ ]:


# Perform query
response = movies.query.near_text(
    query="Epic super hero",
    target_vector="txt_vector",  # Search in the txt_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    print(item.properties["overview"])
    display(Image(item.properties["poster_path"], width=200))


# ## Text-search through the posters vector space

# In[ ]:


# Perform query
response = movies.query.near_text(
    query="Movie about lovable cute pets",
    target_vector="poster_vector",  # Search in the poster_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    print(item.properties["overview"])
    display(Image(item.properties["poster_path"], width=200))


# In[ ]:


# Perform query
response = movies.query.near_text(
    query="Epic super hero",
    target_vector="poster_vector",  # Search in the poster_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    print(item.properties["overview"])
    display(Image(item.properties["poster_path"], width=200))


# ## Image-search through the posters vector space

# In[ ]:


Image("test/spooky.jpg", width=300)


# In[ ]:


# Perform query
response = movies.query.near_image(
    near_image=toBase64("test/spooky.jpg"),
    target_vector="poster_vector",  # Search in the poster_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    display(Image(item.properties["poster_path"], width=200))


# #### One more example!

# In[ ]:


Image("test/superheroes.png", width=300)


# In[ ]:


# Perform query
response = movies.query.near_image(
    near_image=toBase64("test/superheroes.png"),
    target_vector="poster_vector",  # Search in the poster_vector space
    limit=3,
)

# Inspect the response
for item in response.objects:
    print(item.properties["title"])
    display(Image(item.properties["poster_path"], width=200))


# In[ ]:


client.close()

