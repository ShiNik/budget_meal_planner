from pathlib import Path

from logger import get_logger
import random
recipes_logger = get_logger("recipes")

from extracte_product_info import extract_product_info
from utils import list_files_in_folder


def generate_random_products_selection(products_path: Path) -> list[str]:
    products = extract_product_info(_load_flyer_contents(products_path))
    selected_products = []
    for category, category_info in products.items():
        if len(category_info):
            random_index = random.randint(0, len(category_info) - 1)
            selected_products.append(category_info[random_index]["Category"])
            recipes_logger.info(f"Selected product from {category}: {category_info[random_index]}")
    return selected_products



def _load_flyer_contents(products_path: Path) -> str:
    flyer_contents = ""
    for file_path in list_files_in_folder(products_path, "*.txt"):
        with file_path.open('r') as file:
            flyer_contents += file.read()
    return flyer_contents