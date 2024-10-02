from logger import get_logger
import os
from pdf_to_image import convert_pdf_to_images
from  groq_image_to_text import extract_text as groq_extract_text
from  openai_image_to_text import extract_text as openai_extract_text
from group_products import generate_product_group
from prompts import get_prompt, PromptType
from recommend_recipes import recommend_recipes
from generate_vector_database import get_vector_store
from datetime import datetime

api_name= "openai"
extract_text = groq_extract_text if api_name == "groq" else openai_extract_text


report_logger = get_logger("calender")

# Create the directories if they do not exist
def create_directories(directories: list[str]) -> None:
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            report_logger.info(f"Created directory: {directory}")
        else:
            report_logger.info(f"Directory already exists: {directory}")

def main(*,extract_images:bool, extract_products:bool):
    # Define the paths for the directories
    directories = [
        "./outputs/extracted_images_new",
        "./outputs/extracted_products_new",
        "./outputs/recommended_recipes"
    ]

    create_directories(directories)

    if extract_images:
        pdf_path = "./data/flyer.pdf"
        output_folder = directories[0]
        convert_pdf_to_images(pdf_path=pdf_path, output_folder=output_folder)

    if extract_products:
        page_number = 1
        prompt_type = PromptType.EXTRACT_PRODUCT
        image_path = f"{directories[0]}/page_{page_number}.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_put_path = f"{directories[1]}/{api_name}_page_{page_number}_{timestamp}"
        extract_text(image_path=image_path, out_put_path=out_put_path, prompt=get_prompt(prompt_type))

    ingredients_list = generate_product_group()
    vector_store = get_vector_store()
    recommend_recipes(ingredients_list=ingredients_list, output_path=directories[2], vectors=vector_store)


if __name__ ==  "__main__":
    main(extract_images=False, extract_products=False)