# langchain normal openai implementation

# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )
# from langchain.chat_models import ChatOpenAI

# chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
# messages = [
#     SystemMessage(content="You are an expert lawyer well versed in Indian law"),
#     HumanMessage(content="how do i legally evict a resident in my estate who is living rent free ")
# ]
# response=chat(messages)
#
# print(response.content,end='\n')

#####################################################################
# using prompt
import os
import whisper
import requests
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from models import *
import constants
load_dotenv(find_dotenv("local.env"))


async def document_input_feeder(username: str):
    loader = DirectoryLoader('./storage/files/'+username)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model_name="ada")

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    index_name = "apnalawyer-dev"
    Pinecone.from_documents(texts, embeddings, index_name=index_name, namespace=username)

async def get_relevant_docs(query: str, namespace: str):
    embeddings = OpenAIEmbeddings(model_name="ada")
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    index_name = "apnalawyer-dev"
    docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
    docs = docsearch.similarity_search(query)

    return docs


async def audio_transcribe(user):
    model = whisper.load_model("base")
    result = model.transcribe("./storage/files/johndoe/tamil.mp3", task="translate")['text']
    input_query = QueryInput(query=result)
    return await langchain_query_processor(input_query, user)


async def langchain_query_processor(input_query: QueryInput, user):
    query_template = """
    You are an expert lawyer named "Apna Lawyer" well versed in Indian law.
    Answer the query of {query} in a detailed and complete way. 
    Reject if the query is not involving a law or constitution in any way.
    """
    query_prompt = PromptTemplate(
        input_variables=["query"],
        template=query_template,
    )

    llm = OpenAI(model_name=input_query.model)

    if input_query.kanoon:
        if user.tier != 0:
            headers = {'Authorization': f'Token '+os.getenv('KANOON_API_TOKEN')}
            response = requests.post(os.getenv('KANOON_API_URL')+input_query.query, headers=headers, json={})
            docsResponse = response.json()['docs']
            docAndUrlList = []
            for i in docsResponse:
                docAndUrlList.append([i['title'],i['url']])
            return docAndUrlList
        else:
            return [constants.BAD_REQUEST_PERMISSION_DENIED]

    if input_query.query_docs:
        if user.tier != 0:
            docs = await get_relevant_docs(input_query.query, user.username)
            prompt_template = """If the question is not related to any law or the 
            constitution, do not answer the question. If it is indeed related to a law or constitution, use the following pieces of context to answer the question at the end. If you don't 
            know the answer, try using your existing Open AI Chatgpt's general knowledge model apart form this input document to answer the question, but make sure 
            to notify that this is not in the given input context. 

            {context}

            Question: {question}
            Answer:"""
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": input_query.query}, return_only_outputs=True)
            return [response, None]
        else:
            return [constants.BAD_REQUEST_PERMISSION_DENIED]

    query_output=llm(query_prompt.format(query=input_query.query))
    negation_output=None

    if input_query.negation:
        negation_template = "Turn the {answer} and explain to me what will happen if i go against this law. Reject if query is not related to law or constitution in any way."

        negation_prompt = PromptTemplate(
            input_variables=["answer"],
            template=negation_template,
        )

        negation_output=llm(negation_prompt.format(answer=query_output))

    return [query_output, negation_output]


#
# # Run LLM with PromptTemplate
# llm = OpenAI(model_name="text-davinci-003")
#
# llm(prompt.format(query="adopting a child as a single male"))
# llm(prompt.format(concept="regularization"))

# #####################################################################
# # chain
#
# # print(chain.run("autoencoder"))
#

#
# # Define a sequential chain using the two chains above: the second chain takes the output of the first chain as input
#
# from langchain.chains import SimpleSequentialChain
# overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
#
# # Run the chain specifying only the input variable for the first chain.
# explanation = overall_chain.run("autoencoder")
# print(explanation)

 # query_chain = LLMChain(llm=llm, prompt=query_prompt, )
    # negation_chain = LLMChain(llm=llm, prompt=negation_prompt)
    # overall_chain = SimpleSequentialChain(chains=[query_chain, negation_chain], verbose=False)
    #
    # explanation = overall_chain.run(input_query)
#####################################################################

# vectorDB

# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap  = 0,
# )
#
# texts = text_splitter.create_documents([explanation])
# print(texts[0].page_content)
#
# from langchain.embeddings import OpenAIEmbeddings
#
# embeddings = OpenAIEmbeddings(model_name="ada")
#
# # Turn the first text chunk into a vector with the embedding
#
# query_result = embeddings.embed_query(texts[0].page_content)
# print(query_result)
#
# import pinecone
# from langchain.vectorstores import Pinecone
#
# pinecone.init(
#     api_key=os.getenv('PINECONE_API_KEY'),
#     environment=os.getenv('PINECONE_ENV')
# )
#
# # Upload vectors to Pinecone
#
# index_name = "langchain-tut"
# search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
#
# # Do a simple vector similarity search
#
# query = "What is magical about an autoencoder?"
# result = search.similarity_search(query)
#
# print(result)

# #####################################################################
#
# # Import Python REPL tool and instantiate Python agent
#
# from langchain.agents.agent_toolkits import create_python_agent
# from langchain.tools.python.tool import PythonREPLTool
# from langchain.python import PythonREPL
# from langchain.llms.openai import OpenAI
#
# agent_executor = create_python_agent(
#     llm=OpenAI(temperature=0, max_tokens=1000),
#     tool=PythonREPLTool(),
#     verbose=True
# )
#
# # Execute the Python agent
#
# agent_executor.run("Find the roots (zeros) if the quadratic function 3 * x**2 + 2*x -1")
