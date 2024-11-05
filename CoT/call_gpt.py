import re
import base64
from openai import AzureOpenAI

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def request_base64_gpt4v(message_content, system_content=None, seed=42):
    api_key=""
    azure_endpoint=""
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="",
        azure_endpoint=azure_endpoint
    )
    
    if not system_content:
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ]
    else:
        messages=[
            {
                "role": "system",
                "content": system_content,

            },
            {
                "role": "user",
                "content": message_content,
            }
        ]

    response = client.chat.completions.create(
        model="gpt4v",
        messages=messages,
        max_tokens=1024,
        temperature=0,
        seed=seed
    )
    
    return response.choices[0].message.content


def request_gpt4(message_content):
    api_key=""
    azure_endpoint=""

    client = AzureOpenAI(
        api_key=api_key,
        api_version="",
        azure_endpoint=azure_endpoint
    )
    response = client.chat.completions.create(
        model="gpt4",
        messages=[
            {
            "role": "user",
            "content": message_content,
            }
        ],
        max_tokens=1024,
        temperature=0,
        seed=42
    )
    
    return response.choices[0].message.content
