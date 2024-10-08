from logger import get_logger
import os
from pdf_to_image import convert_pdf_to_images
from image_to_text import extract_text
from group_products import generate_product_group
from prompts import PromptManager
from recommend_recipes import recommend_recipes
from generate_vector_database import get_vector_store
from datetime import datetime
from model_factory import ModelFactory
from llm_model import LLMImage, LLMRAG
from common import TaskType
from config import get_config

config = get_config()
recipes_logger = get_logger("recipes")


# Create the directories if they do not exist
def create_directories(directories: list[str]) -> None:
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            recipes_logger.info(f"Created directory: {directory}")
        else:
            recipes_logger.info(f"Directory already exists: {directory}")


def main(*, extract_images: bool, extract_products: bool, vector_store_test:bool):
    prompt_manager = PromptManager(config)
    model_factory = ModelFactory(config)
    if extract_images:
        convert_pdf_to_images(
            pdf_path=config.data_path.pdf_path,
            output_folder=config.output_path.images_path,
        )

    if extract_products:
        page_number = 1
        model_configs = config.get_model_configs(TaskType.EXTRACT_PRODUCT)
        image_path = f"{config.output_path.images_path}/page_{page_number}.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_put_path = f"{config.output_path.products_path}/{model_configs.provider}_page_{page_number}_{timestamp}"
        extract_text(
            image_path=image_path,
            out_put_path=out_put_path,
            model=LLMImage(
                model=model_factory.get_model(TaskType.EXTRACT_PRODUCT),
                prompt=prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT),
            ),
        )

    ingredients_list = generate_product_group()

    llm_model = LLMRAG(
        model=model_factory.get_model(TaskType.RECOMMEND_RECIPES),
        prompt_template=prompt_manager.get_prompt(TaskType.RECOMMEND_RECIPES),
        vectors=get_vector_store(model_factory.get_model(TaskType.EMBEDDING)),
    )

    recommend_recipes(
        ingredients_list=ingredients_list,
        output_path=config.output_path.recipes_path,
        model=llm_model,
    )

    if vector_store_test:
        query = "Find a recipe that includes chicken breast as an ingredient and has less than 200 calories per serving."
        query = "Find a recipe that includes Andouille sausage as an ingredient."
        vectorstore_faiss = get_vector_store(model_factory.get_model(TaskType.EMBEDDING))
        relevant_documents = vectorstore_faiss.similarity_search_with_score(query,k=3, search_type="mmr")
        for i, rel_doc in enumerate(relevant_documents):
            rel_doc, distance = rel_doc[0], rel_doc[1]
            print(f"#### Document {i + 1} ####")
            print(f'{rel_doc.metadata}: \n distance:{distance} \n {rel_doc.page_content} ')

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # extracted_text_path = f"{config.output_path.recipes_path}/doc_{i}_{timestamp}.txt"
            # with open(extracted_text_path, "w") as file:
            #     file.write(rel_doc.page_content)



if __name__ == "__main__":
    main(extract_images=False, extract_products=False, vector_store_test=False)
