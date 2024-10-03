from llm_model import LLMModel
from typing import Tuple
from datetime import datetime


def recommend_recipes(
    *, ingredients_list: list[Tuple[str, str]], output_path: str, model: LLMModel
) -> None:
    for index, (ingredients, conditions) in enumerate(ingredients_list):
        user_message = f"I am looking for a recipe that includes {ingredients} as ingredients. {conditions}"
        user_message = "Find a recipe that includes chicken breast as an ingredient and has less than 200 calories per serving."
        print(user_message)

        response_data = model.runtask(user_message)
        for answer in response_data.split("\n\n"):
            print(answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_text_path = f"{output_path}/recipe_{index}_{timestamp}.txt"
        recommend_recipe = user_message + "\n" + response_data

        # Save the text to a file
        with open(extracted_text_path, "w") as file:
            file.write(recommend_recipe)

        break
