from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from ctransformers import AutoModelForCausalLM
import chainlit as cl
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template =  """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt

# Loading the model 4bit quantized model for Llama make it running on 16GB ram
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 100,
        temperature = 0.3
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def qa_bot():
     embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'})
     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
     llm = load_llm()
     qa_prompt = set_custom_prompt()
     qa = retrieval_qa_chain(llm, qa_prompt, db)

     return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

##chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the service bot....")
    await msg.send()
    msg.content = "<< Medical Query Service >>"
    await msg.update()
    cl.user_session.set('chain', chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, 
        answer_prefix_tokens = ["FINAL", "ANSWER"]
    ) 
    cb.answer_reached=True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"] 

    if sources:
        answer += f"\nSources: " + str(sources)
    else:
        answer += f"\nNo sources found!"

    await cl.Message(content=answer).send()          
