from llm_model import LLMModel


def extract_text(*, image_path:str,
                 out_put_path:str,
                 model: LLMModel) -> None:
    response_data = model.runtask(image_path)
    print(response_data)

    extracted_text_path = f"{out_put_path}.txt"
    with open(extracted_text_path, 'w') as file:
      file.write(response_data)








