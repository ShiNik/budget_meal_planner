from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_core.vectorstores import VectorStore

from typing import Optional, Tuple


from config import get_config
config = get_config()

def recommend_recipes(*, ingredients_list:list[Tuple[str, str]], output_path:str, vectors:VectorStore,
                      prompt_template:str) -> None:
    llm = ChatGroq(groq_api_key=config.api_configs.groq_api_key,
                   model_name=config.api_configs.groq_model_name,
                   temperature=config.api_configs.temperature)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain=create_stuff_documents_chain(llm, prompt)
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