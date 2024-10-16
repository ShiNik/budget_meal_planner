import secrets
from pathlib import Path

from flyer_parser import parse_flyer_products
from logger import get_logger
from utils import list_files_in_folder

recipes_logger = get_logger("recipes")


def generate_random_products_selection(products_path: Path) -> list[str]:
    products = parse_flyer_products(_load_flyer_contents(products_path))
    selected_products = []
    for category, category_info in products.items():
        if len(category_info):
            random_index = secrets.randbelow(len(category_info))
            selected_products.append(category_info[random_index]["Category"])
            recipes_logger.info(f"Selected product from {category}: {category_info[random_index]}")
    return selected_products


def _load_flyer_contents(products_path: Path) -> str:
    flyer_contents = ""
    for file_path in list_files_in_folder(products_path, "*.txt"):
        with file_path.open("r") as file:
            flyer_contents += file.read()
    return flyer_contents
