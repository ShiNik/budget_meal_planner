from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


def recommend_recipes(
    *,
    ingredients_list: list[str],
    output_path: str,
    model: LLMModel,
) -> None:
    for index, ingredients in enumerate(tqdm(ingredients_list, desc="Find a recipes for ingredients:")):
        #TODO: We need to add a condition to specify the foods type like vegetarian
        if ingredients == "vegetable":
            user_message = "Find a vegetarian recipe that includes vegetables as ingredients."
        else:
            user_message = f"Find a recipe that includes {ingredients} as ingredients."
        recipes_logger.info(user_message)

        response_data = model.runtask(user_message)
        recipes_logger.info(response_data)
        for answer in response_data.split("\n\n"):
            recipes_logger.info(answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_text_path = f"{output_path}/recipe_{index}_{timestamp}.txt"
        recommend_recipe = user_message + "\n" + response_data

        # Save the text to a file
        with Path(extracted_text_path).open("w") as file:
            file.write(recommend_recipe)
