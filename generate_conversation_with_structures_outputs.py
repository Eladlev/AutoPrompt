from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
MODEL = "gpt-4o-2024-08-06"

EXAMPLE_CONVERSATIONS_PROMPT = "Please provide a conversation where the child mostly refuses to interact, until the assistant gives up"


class RoleEnum(str, Enum):
    """Conversation roles."""
    USER = "user"
    ASSISTANT = "assistant"

    def model_dump(self):
        return self.value


class ConversationElement(BaseModel):
    """Conversation output."""
    role: RoleEnum
    message: str


class Conversation(BaseModel):
    """Conversation output."""
    messages: list[ConversationElement]

    def pprint(self):
        for message in self.messages:
            print(f"{message.role.value}: {message.message}")


client = OpenAI()


def create_conversation(
    task_instructions: str,
    n: int = 1,
) -> list[list[dict]]:
    print(f"got request for {n} conversations")
    n = min(n, 5)
    user_prompt = f"Please provide a conversation to challenge this task: {task_instructions}"
    _conversations = run(user_prompt=user_prompt, n=n)
    return [[m.model_dump() for m in c.messages] for c in _conversations]


def run(
    user_prompt: str,
    n: int = 1,
    temperature: float = 0.7,
) -> list[Conversation]:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a helpful UX designer creating a challenging scenerios for voice-based conversational AI Toy"},
            {"role": "user",
             "content": user_prompt},
        ],
        response_format=Conversation,
        n=n,
        temperature=temperature,
    )

    re_val = []
    for i, choice in enumerate(completion.choices):
        if not choice.message.parsed:
            print(f"failed to parse choice {i}: {choice.message.refusal}")
            continue
        re_val.append(choice.message.parsed)
    return re_val


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    pargs = parser.parse_args()
    conversations = run(**vars(pargs))

    for i, conv in enumerate(conversations):
        print(f"# Conversation {i}")
        conv.pprint()
