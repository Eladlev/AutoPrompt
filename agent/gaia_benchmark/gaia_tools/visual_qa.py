from PIL import Image
import base64
from io import BytesIO
import json
import os
import requests
from typing import Optional
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, Tool
from transformers.agents import tool
import uuid
import mimetypes
from dotenv import load_dotenv

load_dotenv(override=True)

idefics_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty")

def process_images_and_text(image_path, query, client):
    messages = [
        {
            "role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        },
    ]

    prompt_with_template = idefics_processor.apply_chat_template(messages, add_generation_prompt=True)

    # load images from local directory

    # encode images to strings which can be sent to the endpoint
    def encode_local_image(image_path):
        # load image
        image = Image.open(image_path).convert('RGB')

        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # add string formatting required by the endpoint
        image_string = f"data:image/jpeg;base64,{base64_image}"

        return image_string
    

    image_string = encode_local_image(image_path)
    prompt_with_images = prompt_with_template.replace("<image>", "![]({}) ").format(image_string)


    payload = {
        "inputs": prompt_with_images,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 200,
        }
    }

    return json.loads(client.post(json=payload).decode())[0]

# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"
    
        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


class VisualQATool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
    }
    output_type = "string"

    client = InferenceClient("HuggingFaceM4/idefics2-8b-chatty")

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        try:
            output = process_images_and_text(image_path, question, self.client)
        except Exception as e:
            print(e)
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                output = process_images_and_text(new_image_path, question, self.client)

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output

class VisualQAGPT4Tool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to answer the question. This should be a local path to downloaded image.",
            "type": "string",
        },
    }
    output_type = "string"

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        if not isinstance(image_path, str):
            raise Exception("You should provide only one string as argument to this tool!")

        base64_image = encode_image(image_path)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": question
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
            "max_tokens": 500
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            output = response.json()['choices'][0]['message']['content']
        except Exception:
            raise Exception(f"Response format unexpected: {response.json()}")

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output


@tool
def visualizer(image_path: str, question: Optional[str] = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        question: the question to answer
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
    """

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide only one string as argument to this tool!")

    base64_image = encode_image(image_path)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
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
        "max_tokens": 500
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        output = response.json()['choices'][0]['message']['content']
    except Exception:
        raise Exception(f"Response format unexpected: {response.json()}")

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output