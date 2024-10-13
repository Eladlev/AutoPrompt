
import os

from langchain.agents import  tool, Tool
from langchain.pydantic_v1 import BaseModel, Field
import requests
from typing import Optional

url = ''
import base64
artifacts_folder = './dump/artifacts'

@tool
def Calculator(text: str) -> str:
    """A calculator tool. The input must be a single Python expression and you cannot import packages. You can use functions in the `math` package without import."""
    cur_url = url + 'Calculator'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
        'expression': text
    }

    response = requests.post(cur_url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"


@tool
def OCR(image_path: str) -> str:
    """This tool can recognize all text on the input image.
    OCR results, include bbox in x1, y1, x2, y2 format and the recognized text."""

    cur_url = url + 'OCR'

    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Prepare the file data in the 'image' field
        files = {'image': image_file}

        # Send a POST request with multipart/form-data
        response = requests.post(cur_url, files=files, headers={'accept': 'application/json'})

        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"

@tool
def CountGivenObject(image_path: str, object_description: str) -> str:
    """The tool can count the number of a certain object in the image.
    The object_description should describe the object in English.
    The output is the number of objects found in the image."""
    cur_url = url + 'CountGivenObject'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'text': object_description,
            'bbox': None
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"

@tool
def ImageDescription(image_path: str) -> str:
    """A useful tool that returns a brief description of the input image."""
    cur_url = url + 'ImageDescription'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(cur_url, files=files, headers={'accept': 'application/json'})
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"

class GoogleSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    k: str = Field(description="Select the first k results")
@tool("GoogleSearch", args_schema=GoogleSearchInput)
def GoogleSearch(query: str, k: int = 10) -> str:
    """The tool can search the input query text from Google and return the related results."""
    cur_url = url + 'GoogleSearch'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'query': query,
        'k': k
    }
    response = requests.post(cur_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"


class RegionAttributeDescriptionInput(BaseModel):
    image_path: str = Field(description="The image path")
    bbox: str = Field(description="The bbox coordinate in the format of `(x1, y1, x2, y2)`")
    attribute: str = Field(description="The attribute to describe")
@tool("RegionAttributeDescription", args_schema=RegionAttributeDescriptionInput)
def RegionAttributeDescription(image_path: str, bbox: str, attribute: str) -> str:
    """Describe the attribute of a region of the input image."""
    cur_url = url + 'RegionAttributeDescription'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'bbox': bbox,
            'attribute': attribute
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"


class TextToBboxInput(BaseModel):
    image_path: str = Field(description="The image path")
    text: str = Field(description="The object description in English.")
@tool("TextToBbox", args_schema=TextToBboxInput)
def TextToBbox(image_path: str, text: str) -> str:
    """The tool can detect the object location according to description.
     The output is the objects with the highest score, in (x1, y1, x2, y2) format, and detection score"""
    cur_url = url + 'TextToBbox'

    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Prepare the file data in the 'image' field
        files = {'image': image_file}
        data = {
            'text': text
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})


        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"

class PlotInput(BaseModel):
    command: str = Field(description="Markdown format Python code")
@tool("Plot", args_schema=PlotInput)
def Plot(command: str) -> str:
    """This tool can execute Python code to plot diagrams. The code should include a function named 'solution'. The function should return the matplotlib figure directly. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
import matplotlib.pyplot as plt
def solution():
    # labels and data
    cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
    data = [23, 17, 35, 29, 12, 41]

    # draw diagrams
    figure = plt.figure(figsize=(8, 6))
    plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Car Distribution')
    return figure
```
The tool will save the figure as a PNG file and return the file path."""
    cur_url = url + 'Plot'
    data = {
        'command': command
    }
    response = requests.post(cur_url, data=data, headers={'accept': 'application/json'})
    image_data = base64.b64decode(response.json())
    if response.status_code == 200:
        if not os.path.isdir(artifacts_folder):
            os.mkdir(artifacts_folder)
        ind = len([d for d in os.listdir(artifacts_folder) if 'plot_image' in d])
        fn = f'{artifacts_folder}/plot_image_{ind+1}.png'
        with open(fn, 'wb') as file:
            file.write(image_data)
        return fn
    else:
        return f"Error: {response.status_code}"

@tool
def MathOCR(image_path: str) -> str:
    """This tool can recognize math expressions from an image and return the latex style expression."""
    cur_url = url + 'MathOCR'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(cur_url, files=files, headers={'accept': 'application/json'})
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"

@tool("Solver", args_schema=PlotInput)
def Solver(command: str) -> str:
    """This tool can execute Python code to solve math equations. The code should include a function named 'solution'. You should use the `sympy` library in your code to solve the equations. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
from sympy import symbols, Eq, solve
def solution():
    # Define symbols
    x, y = symbols('x y')

    # Define equations
    equation1 = Eq(x**2 + y**2, 20)
    equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)

    # Solve the system of equations
    solutions = solve((equation1, equation2), (x, y), dict=True)

    # Return solutions as strings
    return str(solutions)
```"""
    cur_url = url + 'Solver'
    data = {
        'command': command
    }
    response = requests.post(cur_url, data=data, headers={'accept': 'application/json'})
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"

class DrawBoxInput(BaseModel):
    image_path: str = Field(description="The image path")
    bbox: str = Field(description="The bbox coordinate in the format of `(x1, y1, x2, y2)`")
    annotation: Optional[str] = Field(description="The extra annotation text of the bbox")
@tool("DrawBox", args_schema=DrawBoxInput)
def DrawBox(image_path: str, bbox: str, annotation: str = None) -> str:
    """A tool to draw a box on a certain region of the input image.
    The output is the image path of the image with the box drawn."""
    cur_url = url + 'DrawBox'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'bbox': bbox,
            'annotation': annotation
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})
        image_data = base64.b64decode(response.json())
        if response.status_code == 200:
            if not os.path.isdir(artifacts_folder):
                os.mkdir(artifacts_folder)
            ind = len([d for d in os.listdir(artifacts_folder) if 'draw_box' in d])
            fn = f'{artifacts_folder}/draw_box_{ind + 1}.png'
            with open(fn, 'wb') as file:
                file.write(image_data)
            return fn
        else:
            return f"Error: {response.status_code}"

class AddTextInput(BaseModel):
    image_path: str = Field(description="The image path")
    text: str = Field(description="The text to draw")
    position: str = Field(description="The left-bottom corner coordinate in the format of `(x, y)`, or a combination of [\"l\"(left), \"m\"(middle), \"r\"(right)] and [\"t\"(top), \"m\"(middle), \"b\"(bottom)] like \"mt\" for middle-top")
    color: Optional[str] = Field(description="The color of the text", default='red')
@tool("AddText", args_schema=AddTextInput)
def AddText(image_path: str, text: str, position: str, color: str = 'red') -> str:
    """A tool to draw a text on a certain region of the input image.
        The output is the image path of the image with the text drawn."""
    cur_url = url + 'AddText'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'text': text,
            'position': position,
            'color': color
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})
        image_data = base64.b64decode(response.json())
        if response.status_code == 200:
            if not os.path.isdir(artifacts_folder):
                os.mkdir(artifacts_folder)
            ind = len([d for d in os.listdir(artifacts_folder) if 'add_text' in d])
            fn = f'{artifacts_folder}/add_text_{ind + 1}.png'
            with open(fn, 'wb') as file:
                file.write(image_data)
            return fn
        else:
            return f"Error: {response.status_code}"

class TextToImageInput(BaseModel):
    keywords: str = Field(description="A series of keywords separated by comma.")
@tool("TextToImage", args_schema=TextToImageInput)
def TextToImage(keywords: str) -> str:
    """This tool can generate an image according to the input text.
    The output is the image path of the generated image."""
    cur_url = url + 'TextToImage'
    data = {
        'keywords': keywords
    }
    response = requests.post(cur_url, data=data, headers={'accept': 'application/json'})
    image_data = base64.b64decode(response.json())
    if response.status_code == 200:
        if not os.path.isdir(artifacts_folder):
            os.mkdir(artifacts_folder)
        ind = len([d for d in os.listdir(artifacts_folder) if 'text_to_image' in d])
        fn = f'{artifacts_folder}/text_to_image_{ind + 1}.png'
        with open(fn, 'wb') as file:
            file.write(image_data)
        return fn
    else:
        return f"Error: {response.status_code}"

@tool
def ImageStylization(image_path: str, instruction: str) -> str:
    """This tool can modify the input image according to the input instruction. Here are some example instructions: "turn him into cyborg", "add fireworks to the sky", "make his jacket out of leather".
    The output is the image path of the stylized image."""
    cur_url = url + 'ImageStylization'
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'instruction': instruction
        }

        # Send the POST request
        response = requests.post(cur_url, files=files, data=data, headers={'accept': 'application/json'})
        image_data = base64.b64decode(response.json())
        if response.status_code == 200:
            if not os.path.isdir(artifacts_folder):
                os.mkdir(artifacts_folder)
            ind = len([d for d in os.listdir(artifacts_folder) if 'image_stylization' in d])
            fn = f'{artifacts_folder}/image_stylization_{ind + 1}.png'
            with open(fn, 'wb') as file:
                file.write(image_data)
            return fn
        else:
            return f"Error: {response.status_code}"

@tool
def parse_yaml_code(yaml_code: str) -> dict:
    """You must use this tool before sending the final output, the input is the yaml code with the output schema. The result is the final output!"""
    return "The Yaml doesn't have a valid yaml structure, please fix it such that it can be parsed. Remember that if you have a value that is a string, you should wrap it in quotes."