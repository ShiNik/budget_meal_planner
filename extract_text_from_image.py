import base64
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
page_number= "13"
image_path = f"./outputs/extracted_images/page_{page_number}.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": """Analyze the following image of a flyer. The flyer contains both English and French text about various products. 
        Please extract the information for each product, including:
        - The product name or brand logo.
        - The price of the product.
        - Any promotions, discounts, or special offers associated with the product.
        - Group all the related information for each product together.
        Ensure that both English and French details are included for each product, and clearly indicate the product's price and any promotional offers."""
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
  "max_tokens": 1000  # Increase the token limit to ensure detailed extraction
}


response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
# Convert the response to JSON
response_data = response.json()

# Check if 'choices' is in the response
if 'choices' in response_data:
  # Extract the content from the first choice
  message_content = response_data['choices'][0]['message']['content']
  print(message_content)

  extracted_text_path = f"./outputs/page_{page_number}.txt"
  # Save the text to a file
  with open(extracted_text_path, 'w') as file:
    file.write(message_content)

  # Save the response to a file
  output_file = f"./outputs/page_{page_number}.json"
  with open(output_file, "w") as file:
    json.dump(response_data, file, indent=4)
  print(f"Response saved to {output_file}")
else:
  # Print the full response for debugging
  print("Error: 'choices' not found in response")
  print(json.dumps(response_data, indent=4))