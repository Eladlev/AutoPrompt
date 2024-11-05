from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from utils.config import get_llm
import base64
import os
import re
import pandas as pd
from utils.llm_chain import sync_chain_batch_run


def extract_score_and_feedback(input_string):
    # Define regular expressions to capture score and feedback
    score_pattern = r"#\s*Score:\s*(\d+(\.\d+)?)"  # Matches the score, which can be a float
    feedback_pattern = r"#\s*Feedback:\s*(.*)" # Matches the feedback text

    # Find the score
    score_match = re.search(score_pattern, input_string)
    score = float(score_match.group(1)) if score_match else None

    # Find the feedback
    feedback_match = re.search(feedback_pattern, input_string,  re.DOTALL)
    feedback = feedback_match.group(1) if feedback_match else None

    # Return the result in a dictionary
    return {"score": score, "reason": feedback}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ImChainWrapper:
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, kwargs):
        result = self.chain.invoke({})
        return extract_score_and_feedback(result.content)

class ImageEvaluator:
    """
    An image evaluator class
    """

    def __init__(self, config, mode="generation"):
        """
        Initialize a new instance of the ImageEvaluator class.
        :param config: The configuration file (EasyDict)
        """
        self.config = config
        self.llm = get_llm(config.llm)
        self.source_images = config.source_image_folder
        self.accepted_image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        self.image_list = self.get_image_list()
        self.meta_prompts_folder = self.get_meta_prompts_folder(mode)
        self.task_description = config.task_description
        self.source_prompt = self.generate_source_prompt_message()
        self.system_prompt = self.generate_system_message()

    def get_image_list(self):
        image_list = [f for f in os.listdir(self.source_images) if f.lower().endswith(self.accepted_image_extensions)]
        image_list = [os.path.join(self.source_images, f) for f in image_list]
        if len(image_list) == 0:
            raise Exception("No images found in source folder: {self.source_images}")
        return image_list

    def get_meta_prompts_folder(self, mode):
        if mode.lower() == "generation":
            meta_prompts_folder = self.config.image_generation_meta_prompt_folder
        elif mode.lower() == "description":
            meta_prompts_folder = self.config.image_description_meta_prompts_folder
        else:
            raise Exception(f"Unknown mode for ImageEvaluator: {mode}")
        return meta_prompts_folder

    def load_meta_prompt(self, mata_prompt_name):
        mata_prompt_filepath = os.path.join(self.meta_prompts_folder, mata_prompt_name + ".prompt")
        if os.path.exists(mata_prompt_filepath):
            with open(mata_prompt_filepath, 'r') as file:
                meta_prompt = file.read()
        else:
            raise Exception(f"meta prompt file not found! {mata_prompt_filepath}")
        return meta_prompt

    def generate_source_prompt_message(self):
        source_meta_prompt = self.load_meta_prompt("source_message")
        message_content = [{"type": "text",
                            "text": source_meta_prompt}]
                            # "text": "You are given images of a specific character or an object, we refer these images as the 'source images' and the object as <x>"}]
        for filename in os.listdir(self.source_images):
            # Construct full file path
            file_path = os.path.join(self.source_images, filename)

            # Check if the file is an image
            if filename.lower().endswith(self.accepted_image_extensions):
                try:
                    message_content.append({"type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,"
                                                                 f"{encode_image(file_path)}"}
                                            })

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
        return HumanMessage(content=message_content)

    def generate_system_message(self):
        system_meta_prompt = self.load_meta_prompt("system_message")
        system_message = SystemMessage(content=system_meta_prompt)
        return system_message

    def generate_eval_message(self, url):

        eval_meta_prompt = self.load_meta_prompt("eval_message").replace("{task_description}", self.task_description)

        prompt_text = f"""The user ask for the following image:{self.task_description}.
        The model generated the provided image in response. You should evaluate the quality of the generation according to the following criteria:
        1. <x> should be exactly the same in the generated image and the source images!!
        2. The generated image should adhere the user request.
        You should provide a score between 1 and 10 where 10 is only if <x> is exactly the same as in the source image and the generated image adheres to the user prompt.
        You should also provide a feedback.
         The feedback guidelines:
            - The feedback should explain the exact appearance of features in <x> that were not generate properly in the image. Do not use relative words, instead describe the features. For example instead of using 'bigger nose', describe the nose apperence 
            - This feedback should be highly details, you should assume that the model doesn't have access to the generated image and the source image, only to the provided feedback.
            - Do not use relative instructions like "more" or "less", instead describe exactly the character or object features that should be changed!!
            - The Image generator model is familiar with famous characters and objects, you can use can use similarity to famous characters and objects to guide the model. 
            - If <x> is a character focus on the face resemblance. 
        
        The format of the response should be as follows:
        #Score: <score>
        #Feedback: <feedback>
        """
        # message_content = [{"type": "text", "text": prompt_text}]
        message_content = [{"type": "text", "text": eval_meta_prompt}]
        if os.path.isfile(url):
            url = f"data:image/jpeg;base64,{encode_image(url)}"
        message_content.append({"type": "image_url", "image_url": {"url": url}})
        return HumanMessage(content=message_content)


    def generate_prompt_message(self, url):
        """
        Create a prompt message for the evaluator
        """
        prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.source_prompt,
            self.generate_eval_message(url)])
        return prompt

    def invoke(self, url):
        """
        Invoke the evaluator
        """
        chain = self.generate_prompt_message(url) | self.llm
        result = chain.invoke({})
        return extract_score_and_feedback(result.content)

    def dataset_invoke(self, record: pd.DataFrame, num_workers: int = 1):
        """
        Invoke the evaluator on a dataset
        """
        batch_inputs = []
        # prepare all the inputs for the chains
        for i, row in record.iterrows():
            batch_inputs.append({'sample_chain_input': '', 'index': i,
                                 'chain': ImChainWrapper(self.generate_prompt_message(row['prediction']) | self.llm)})
        all_results = sync_chain_batch_run(None, batch_inputs, num_workers, get_index=True)
        return all_results

    def generate_initial_image_generation_prompt(self):

        print("Generating new prompt based on source image...")
        record_proxy = pd.DataFrame({"prediction": self.image_list})
        all_results = self.dataset_invoke(record_proxy)

        prompt_suggestion = all_results[0]["result"]["reason"]

        print(f'New initial prompt:\n{prompt_suggestion}')
        return prompt_suggestion
