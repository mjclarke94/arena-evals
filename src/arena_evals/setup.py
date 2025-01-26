import os

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_client() -> OpenAI:
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key"
    openai_client = OpenAI()
    return openai_client
