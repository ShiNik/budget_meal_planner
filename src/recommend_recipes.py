from datetime import datetime
from pathlib import Path

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


def recommend_recipes(
    *,
    ingredients_list: list[tuple[str, str]],
    output_path: str,
    model: LLMModel,
) -> None:
    for index, (ingredients, conditions) in enumerate(ingredients_list):
        user_message = f"I am looking for a recipe that includes {ingredients} as ingredients. {conditions}"
        user_message = (
            "Find a recipe that includes chicken breast as an ingredient and has less than 200 calories per serving."
        )
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

        break
