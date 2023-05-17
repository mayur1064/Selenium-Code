!pip install llama_index

! pip install langchain
! pip install openai

from llama_index import SimpleDirectoryReader , GPTListIndex , GPTVectorStoreIndex, LLMPredictor , PromptHelper, ServiceContext 
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"] 

def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    #define LLM 
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))

    #load Data
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index
    # vectorIndex = GPTVectorStoreIndex(documents=docs,llm_predictor=llmPredictor,prompt_helper=prompt_helper)
    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    vectorIndex = GPTVectorStoreIndex.from_documents(
        docs, service_context=service_context
    )

    # vectorIndex.save_to_disk("vectorIndex.json")
    vectorIndex.storage_context.persist(persist_dir="/content/knowledge")
    return vectorIndex;


vectorIndex = createVectorIndex('/content/knowledge')

from llama_index import StorageContext, load_index_from_storage

def answerMe():
  #Rebuild Storage context
  storage_context = StorageContext.from_defaults(persist_dir="/content/knowledge")
  # load index
  vIndex = load_index_from_storage(storage_context)
  query_engine = vIndex.as_query_engine()
  while(True):
    prompt= input("Plase ask something ")
    response = query_engine.query(prompt)
    print(f"Response: {response} \n")
    

answerMe()