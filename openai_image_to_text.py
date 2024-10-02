import base64
import requests
import json

from config import get_config
config = get_config()

def extract_text(*, image_path:str, out_put_path:str, prompt:str) -> None:

  # Getting the base64 string
  base64_image = _encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.api_keys.openai_api_key}"
  }

  payload = {
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "temperature": 0.0,
    "max_tokens": 1000  # Increase the token limit to ensure detailed extraction
  }

  try:
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # Convert the response to JSON
    response_data = response.json()

    # Check if 'choices' is in the response
    if 'choices' in response_data:
      # Extract the content from the first choice
      message_content = response_data['choices'][0]['message']['content']
      print(message_content)

      extracted_text_path = f"{out_put_path}..txt"
      # Save the text to a file
      with open(extracted_text_path, 'w') as file:
        file.write(message_content)

      # Save the response to a file
      output_file = f"{out_put_path}..json"
      with open(output_file, "w") as file:
        json.dump(response_data, file, indent=4)
      print(f"Response saved to {output_file}")
    else:
      # Print the full response for debugging
      print("Error: 'choices' not found in response")
      print(json.dumps(response_data, indent=4))

  except requests.exceptions.RequestException as e:
    print(f"Error during OpenAI API request: {e}")

  except ValueError as ve:
    print(f"Value error: {ve}")


def _encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')