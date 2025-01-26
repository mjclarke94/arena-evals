import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_openai_client() -> OpenAI:
    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key"
    openai_client = OpenAI()
    return openai_client


def get_anthropic_client() -> Anthropic:
    assert os.getenv("ANTHROPIC_API_KEY") is not None, "You must set your Anthropic API key"
    anthropic_client = Anthropic()
    return anthropic_client
