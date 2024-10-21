import asyncio
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


async def recommend_recipes_old(
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

        response_data = await model.runtask(user_message)
        recipes_logger.info(response_data)
        for answer in response_data.split("\n\n"):
            recipes_logger.info(answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_text_path = f"{output_path}/recipe_{index}_{timestamp}.txt"
        recommend_recipe = user_message + "\n" + response_data

        # Save the text to a file
        with Path(extracted_text_path).open("w") as file:
            file.write(recommend_recipe)


async def recommend_recipe_for_ingredients(
    ingredients: str,
    model: LLMModel,
    output_path: str,
    index: int,
) -> None:
    if ingredients == "vegetable":
        user_message = "Find a vegetarian recipe that includes vegetables as ingredients."
    else:
        user_message = f"Find a recipe that includes {ingredients} as ingredients."

    recipes_logger.info(user_message)

    # Call the model asynchronously
    response_data = await model.runtask(user_message)

    recipes_logger.info(response_data)
    for answer in response_data.split("\n\n"):
        recipes_logger.info(answer)

    # Generate a timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extracted_text_path = f"{output_path}/recipe_{index}_{timestamp}.txt"
    recommend_recipe = user_message + "\n" + response_data

    # Save the text to a file asynchronously
    with Path(extracted_text_path).open("w") as file:
        file.write(recommend_recipe)


async def recommend_recipes(
    *,
    ingredients_list: list[str],
    output_path: str,
    model: LLMModel,
) -> None:
    tasks = []

    # Iterate through ingredients and create tasks for each recipe recommendation
    for index, ingredients in enumerate(tqdm(ingredients_list, desc="Find a recipes for ingredients:")):
        tasks.append(recommend_recipe_for_ingredients(ingredients, model, output_path, index))

    # Run all tasks concurrently
    await asyncio.gather(*tasks)
