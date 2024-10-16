import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm

from config import Config
from llm_model import LLMModel
from logger import get_logger
from utils import get_name_from_path, list_files_in_folder

recipes_logger = get_logger("recipes")


async def process_image_for_extraction(*, image_path: str, out_put_path: str, model: LLMModel) -> None:
    response_data = await model.runtask(image_path)
    recipes_logger.info(response_data)
    with Path(f"{out_put_path}.txt").open("w") as file:
        file.write(response_data)


async def extract_text(config: Config, model: LLMModel):
    image_files_path = list_files_in_folder(config.output_path.images_path, "*.png")
    tasks = []
    for image_path in tqdm(image_files_path, desc="Extracting product from flyer pages"):
        recipes_logger.info(f"Extracting product from {get_name_from_path(image_path)}")
        out_put_path = f"{config.output_path.products_path}/{get_name_from_path(image_path)}"
        tasks.append(process_image_for_extraction(image_path=image_path, out_put_path=out_put_path, model=model))
    await asyncio.gather(*tasks)
