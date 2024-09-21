from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import ToolAgentAction
from json import JSONDecodeError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
import json
from langchain_core.outputs import ChatGeneration, Generation
import yaml

from langchain_core.agents import AgentActionMessageLog, AgentFinish
import re
from typing import List, Union
from langchain_core.agents import AgentAction

def extract_yaml_content(variable):
    # Check if the input is a string
    variable = yaml.safe_load(variable)
    if isinstance(variable, str):
        # Define the regex pattern to match ```yaml <content>```
        pattern = r'```yaml\s*(.*?)\s*```'

        # Search for the pattern in the string
        match = re.search(pattern, variable, re.DOTALL)

        # If a match is found, return the extracted content
        if match:
            return yaml.safe_load(match.group(1).strip())
        else:
            raise ValueError("No YAML content found in the input string")
    else:
        return variable


def parse_ai_message_to_tool_action(
        message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )
        # Best-effort parsing
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            function = tool_call["function"]
            function_name = function["name"]
            try:
                args = json.loads(function["arguments"] or "{}")
                tool_calls.append(
                    ToolCall(name=function_name, args=args, id=tool_call["id"])
                )
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function} because "
                    f"the `arguments` is not valid JSON."
                )
    for tool_call in tool_calls:
        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        function_name = tool_call["name"]
        _tool_input = tool_call["args"]
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input
        content_msg = f"responded: {message.content}\n" if message.content else "\n"
        log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
        if function_name == 'parse_yaml_code':
            try:
                inputs = extract_yaml_content(tool_input['yaml_code'])
                return AgentFinish(return_values={'output': tool_input['yaml_code']}, log=log)
            except:
                pass

        actions.append(
            ToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call["id"],
            )
        )
    return actions


class Response(BaseModel):
    """Final YAML file response"""

    yaml_file: str = Field(description="The final YAML file")


class ToolsAgentOutputParser(MultiActionAgentOutputParser):
    """Parses a message into agent actions/finish.

    If a tool_calls parameter is passed, then that is used to get
    the tool names and tool inputs.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "tools-agent-output-parser"

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return parse_ai_message_to_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")

def build_tool_function(agent):
    def new_function(input_str):
        results = agent.invoke({'input': input_str})
        return results

    return new_function
