from logger import get_logger
import os
from pdf_to_image import convert_pdf_to_images
from  image_to_text import extract_text
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

def main(*,extract_images:bool, extract_products:bool):
    prompt_manager = PromptManager(config)
    model_factory = ModelFactory(config)
    if extract_images:
        convert_pdf_to_images(pdf_path=config.data_path.pdf_path, output_folder=config.output_path.images_path)

    if extract_products:
        page_number = 1
        model_configs = config.get_model_configs(TaskType.EXTRACT_PRODUCT)
        image_path = f"{config.output_path.images_path}/page_{page_number}.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_put_path = f"{config.output_path.products_path}/{model_configs.provider}_page_{page_number}_{timestamp}"
        extract_text(image_path=image_path,
                     out_put_path=out_put_path,
                     model = LLMImage(model=model_factory.get_model(TaskType.EXTRACT_PRODUCT),
                                      prompt=prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT)
                                    ))

    ingredients_list = generate_product_group()

    llm_model = LLMRAG(model=model_factory.get_model(TaskType.RECOMMEND_RECIPES),
    prompt_template= prompt_manager.get_prompt(TaskType.RECOMMEND_RECIPES),
    vectors= get_vector_store(model_factory.get_model(TaskType.EMBEDDING)))

    recommend_recipes(ingredients_list=ingredients_list,
                      output_path=config.output_path.recipes_path,
                     model=llm_model)


if __name__ ==  "__main__":
   main(extract_images=False, extract_products=False)


