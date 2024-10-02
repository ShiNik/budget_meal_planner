prompt_extract_product = """Analyze the following image of a flyer. The flyer contains both English and French text about various products. 
        Please extract the information for each product, including:
        - The product name or brand logo.
        - The price of the product.
        - Any promotions, discounts, or special offers associated with the product.
        - Group all the related information for each product together.
        Ensure that both English and French details are included for each product, and clearly indicate the product's price and any promotional offers."""

prompt_recommend_recipes = """
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


from enum import StrEnum, auto
import json
from config import get_config
from typing import Optional

class PromptType(StrEnum):
    extract_product = auto()
    recommend_recipes = auto()

class PromptManager:
    def __init__(self):
        self._prompts_cache = {}

    def _load_prompts(self, prompt_type: PromptType) -> Optional[dict[str, str]]:
        config = get_config()
        filename = config.prompt_files.get(prompt_type)
        if not filename:
            raise ValueError(f"No filename found for prompt type: {prompt_type}")

        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {filename} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"The file {filename} contains invalid JSON.")
        except Exception as e:  # Catch any other unexpected errors
            raise Exception(f"An unexpected error occurred: {e}")

    def get_prompt(self, prompt_type: PromptType) -> str:
        """Get the prompt based on the prompt type."""
        if prompt_type not in self._prompts_cache:
            self._prompts_cache[prompt_type] = self._load_prompts(prompt_type)
        return self._prompts_cache.get(prompt_type, {}).get('template', None)

