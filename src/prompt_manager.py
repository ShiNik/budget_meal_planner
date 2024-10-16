import json
from pathlib import Path

from common import TaskType
from config import Config

prompt_extract_product = """Analyze the following image of a flyer. The flyer contains both English and French text 
    about various products. Please extract the information for each product, including:
    - The product name or brand logo.
    - The price of the product.
    - Any promotions, discounts, or special offers associated with the product.
    - Group all the related information for each product together.
    - Categorize each product into one of the following groups:
     'Chicken,' 'Fish', 'Pork','Beef', 'Vegetable', 'Fruit,' or 'Others.'
    
    Ensure that both English and French details are included for each product, 
    clearly indicating the product's price and any promotional offers. Additionally, 
    make sure to assign each product to the correct category based on the type of product."""

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

        The recipes must be selected from the context provided below. If any ingredients are missing from the list,
         include them in the recipe details.

        If you cannot find a recipe that meets the criteria, please state that you don’t know.

        <contex>
        {context}
        </context>

        Questions:
        {input}
        """


class PromptManager:
    def __init__(self, config: Config):
        self.config = config
        self._prompts_cache = {}

    def _load_prompts(self, task_type: TaskType) -> dict[str, str] | None:
        filename = self.config.get_prompt_file_path(task_type)
        if not filename:
            raise ValueError(f"No prompy file found for task type: {task_type}")

        try:
            with Path(filename).open() as file:
                return json.load(file)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"The file {filename} was not found.") from err
        except json.JSONDecodeError as err:
            raise ValueError(f"The file {filename} contains invalid JSON.") from err
        except Exception as err:  # Catch any other unexpected errors
            raise Exception(f"An unexpected error occurred: {err}") from err

    def get_prompt(self, task_type: TaskType) -> str | None:
        """Get the prompt based on the prompt type."""
        if task_type not in self._prompts_cache:
            self._prompts_cache[task_type] = self._load_prompts(task_type)
        return self._prompts_cache.get(task_type, {}).get("template", None)
