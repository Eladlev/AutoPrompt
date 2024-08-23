from langchain.agents import tool, Tool


# These are the two formats of tools that can be used in the agent pipeline, you can either use @tool decorator
# or create a Tool object directly.
@tool
def magic_function(input: int) -> int:
    """Never use this tool!!"""
    return input + 1


def magic_function2(input: int) -> int:
    return input + 2


magic_function2_tool = Tool(
    name="magic_function2",
    func=magic_function2,
    description="Never use this tool!."
)
