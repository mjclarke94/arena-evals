from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from tabulate import tabulate

from arena_evals.setup import get_openai_client
from arena_evals.utils import retry_with_exponential_backoff

type Message = dict[Literal["role", "content"], str]
type Messages = list[Message]


def OpenAIWrapperFactory(
    default_model="gpt-4o-mini",
    default_temperature: float = 1,
    default_max_tokens: int = 1000,
    default_stop_sequences: list = [],
    default_max_retries: int = 20,
    default_initial_sleep_time: float = 1.0,
    default_backoff_factor: float = 1.5,
):
    @dataclass
    class OpenAIWrapper:
        @cached_property
        def client(self):
            return get_openai_client()

        def generate_response_basic(
            self,
            messages: Messages,
            model: str = default_model,
            temperature: float = default_temperature,
            max_tokens: int = default_max_tokens,
            verbose: bool = False,
            stop_sequences: list[str] = default_stop_sequences,
        ) -> str:
            if verbose:
                print(
                    tabulate([m.values() for m in messages], ["role", "content"], "simple_grid", maxcolwidths=[50, 70])
                )
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                )
                return response.choices[0].message.content

            except Exception as e:
                raise RuntimeError(f"Error in generation:\n{e}") from e

    OpenAIWrapper.generate_response = retry_with_exponential_backoff(
        OpenAIWrapper.generate_response_basic, default_max_retries, default_initial_sleep_time, default_backoff_factor
    )

    return OpenAIWrapper
