from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_core.vectorstores import VectorStore


from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_ollama.llms import OllamaLLM
from typing import Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

def recommend_recipes(*, ingredients_list:list[Tuple[str, str]], output_path:str, vectors:VectorStore) -> None:
    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name="llama-3.1-70b-versatile",
                   temperature=0.0)

    prompt = ChatPromptTemplate.from_template(
        """
        Your job is to find a recipe from the provided context that use the given ingredients. 
        Ensure that each recipe is described with the following fields:
        - **Recipe 1: 
        - **name**: The name of the recipe.
        - **preparation_time**: The time required to prepare the recipe.
        - **directions**: A list of instructions for preparing the recipe.
        - **ingredients**: A list of ingredients required for the recipe.
        - **calories**: The total number of calories in the recipe.
        - **total fat (PDV)**: Percentage of daily value for total fat.
        - **sugar (PDV)**: Percentage of daily value for sugar.
        - **sodium (PDV)**: Percentage of daily value for sodium.
        - **protein (PDV)**: Percentage of daily value for protein.
        - **saturated fat (PDV)**: Percentage of daily value for saturated fat.
        - **carbohydrates (PDV)**: Percentage of daily value for carbohydrates.
    
        The recipes must be selected from the context provided below. If any ingredients are missing from the list, include them in the recipe details.
    
        If you cannot find a recipe that meets the criteria, please state that you don’t know.
    
        <contex>
        {context}
        </context>
    
        Questions:
        {input}
        """
    )




    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    for index, (ingredients, conditions) in enumerate(ingredients_list):
        user_message = f"I am looking for a recipe that includes {ingredients} as ingredients. {conditions}"
        user_message = "Find a recipe that includes chicken breast as an ingredient and has less than 200 calories per serving."
        print(user_message)

        response=retrieval_chain.invoke({'input':user_message})
        for answer in response["answer"].split("\n\n"):
            print(answer)

        extracted_text_path = f"{output_path}/recipe_{index}.txt"
        recommend_recipe = user_message + "\n" + response["answer"]

        # Save the text to a file
        with open(extracted_text_path, 'w') as file:
            file.write(recommend_recipe)

        break