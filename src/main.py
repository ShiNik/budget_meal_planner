import asyncio
from pathlib import Path

from common import TaskType
from config import get_config
from image_to_text import extract_text
from llm_model import LLMRAG, LLMImage
from logger import get_logger
from model_factory import ModelFactory
from pdf_to_image import convert_pdf_to_images
from prompt_manager import PromptManager
from recommend_recipes import recommend_recipes
from select_products import generate_random_products_selection
from vector_database import create_vector_database

config = get_config()
recipes_logger = get_logger("recipes")


# Create the directories if they do not exist
def create_directories(directories: list[str]) -> None:
    for directory in directories:
        if not Path(directory).exists():
            Path(directory).mkdir(parents=True)
            recipes_logger.info(f"Created directory: {directory}")
        else:
            recipes_logger.info(f"Directory already exists: {directory}")


async def main(
    *,
    extract_images: bool,
    extract_products: bool,
    execute_recipe_recommendation: bool,
    vector_store_test: bool,
) -> None:
    prompt_manager = PromptManager(config)
    model_factory = ModelFactory(config)
    if extract_images:
        await convert_pdf_to_images(
            pdf_path=config.data_path.pdf_path,
            output_folder=config.output_path.images_path,
        )
    if extract_products:
        await extract_text(
            config,
            model=LLMImage(
                model=model_factory.get_model(TaskType.EXTRACT_PRODUCT),
                prompt=prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT),
            ),
        )

    if execute_recipe_recommendation:
        ingredients_list = generate_random_products_selection(config.output_path.products_path)
        await recommend_recipes(
            ingredients_list=ingredients_list,
            output_path=config.output_path.recipes_path,
            model=LLMRAG(
                model=model_factory.get_model(TaskType.RECOMMEND_RECIPES),
                prompt_template=prompt_manager.get_prompt(TaskType.RECOMMEND_RECIPES),
                vectors=create_vector_database(model_factory.get_model(TaskType.EMBEDDING)),
            ),
        )

    if vector_store_test:
        # TODO: This is a test code to find the best solution for retrieving data from a vector database.
        query = (
            "Find a recipe that includes chicken breast as an ingredient and has less than 200 calories per serving."
        )
        query = "Find a recipe that includes Andouille sausage as an ingredient."
        vectorstore_faiss = create_vector_database(model_factory.get_model(TaskType.EMBEDDING))
        relevant_documents = vectorstore_faiss.similarity_search_with_score(
            query,
            k=3,
            search_type="mmr",
        )
        for i, rel_doc_info in enumerate(relevant_documents):
            rel_doc, distance = rel_doc_info[0], rel_doc_info[1]
            recipes_logger.info(f"#### Document {i + 1} ####")
            recipes_logger.info(
                f"{rel_doc.metadata}: \n distance: {distance} \n {rel_doc.page_content} ",
            )


if __name__ == "__main__":
    asyncio.run(
        main(
            extract_images=False,
            extract_products=False,
            execute_recipe_recommendation=False,
            vector_store_test=False,
        ),
    )
