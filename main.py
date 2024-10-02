from logger import get_logger
import os
from pdf_to_image import convert_pdf_to_images
from  groq_image_to_text import extract_text as groq_extract_text
from  openai_image_to_text import extract_text as openai_extract_text
from group_products import generate_product_group
from prompts import PromptManager, PromptType
from recommend_recipes import recommend_recipes
from generate_vector_database import get_vector_store
from datetime import datetime

from config import get_config

config = get_config()

api_name= "openai"
extract_text = groq_extract_text if api_name == "groq" else openai_extract_text


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
    prompt_manager = PromptManager()
    if extract_images:
        convert_pdf_to_images(pdf_path=config.data_path.pdf_path, output_folder=config.output_path.images_path)

    if extract_products:
        page_number = 1
        image_path = f"{config.output_path.images_path}/page_{page_number}.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_put_path = f"{config.output_path.products_path}/{api_name}_page_{page_number}_{timestamp}"
        extract_text(image_path=image_path, out_put_path=out_put_path, prompt=prompt_manager.get_prompt(PromptType.extract_product))

    ingredients_list = generate_product_group()
    vector_store = get_vector_store()
    recommend_recipes(ingredients_list=ingredients_list,
                      output_path=config.output_path.recipes_path,
                      vectors=vector_store,
                      prompt_template=prompt_manager.get_prompt(PromptType.recommend_recipes))


if __name__ ==  "__main__":
    main(extract_images=False, extract_products=False)
