# A file containing the parser for the output of all the LLM chains
import re

def initial_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    pattern = r'(#### Sample \d+:)([\s\S]*?)(?=(#### Sample \d+:|$))'

    matches = re.findall(pattern, response['text'])
    results = {'samples' :[]}
    for match in matches:
        header, content = match[0], match[1]
        results['samples'].append(content.strip())
    return results

step_samples_parser = initial_parser

def step_prompt_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    pattern = re.compile( r"#### prompt:\n(?P<prompt>.*?)\n#### score:\n(?P<score>[\d.]+)", re.DOTALL)
    match = pattern.search(response['text'])
    if match:
        result = {
            'prompt': match.group('prompt'),
            'score': float(match.group('score'))
        }
        return result
    else:
        result = {
            'prompt': '',
            'score': 0.0
        }
        return result