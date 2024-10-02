from groq import Groq
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.environ['GROQ_API_KEY']

def extract_text(*, image_path:str, out_put_path:str, prompt:str) -> None:
    # Getting the base64 string
    base64_image = _encode_image(image_path)

    client = Groq(api_key=groq_api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    message_content = chat_completion.choices[0].message.content
    print(message_content)

    extracted_text_path = f"{out_put_path}.txt"
    # Save the text to a file
    with open(extracted_text_path, 'w') as file:
        file.write(message_content)


# Function to encode the image
def _encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')