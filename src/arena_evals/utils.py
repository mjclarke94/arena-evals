import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from openai._exceptions import RateLimitError

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_exponential_backoff(
    func: Callable[P, T],
    max_retries: int = 20,
    initial_sleep_time: float = 1.0,
    backoff_factor: float = 1.5,
) -> Callable[P, T]:
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors, using an exponential backoff strategy to
    increase the wait time between retries.

    Args:
        func (callable): The function to be retried.
        max_retries (int): Maximum number of retry attempts. Defaults to 20.
        initial_sleep_time (float): Initial sleep time in seconds. Defaults to 1.
        backoff_factor (float): Factor by which the sleep time increases after each retry. Defaults to 1.5.

    Returns:
        callable: A wrapped version of the input function with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.

    Note:
        This function specifically handles rate limit errors. All other exceptions
        are re-raised immediately.
    """

    def wrapper(*args, **kwargs) -> T:
        sleep_time = initial_sleep_time

        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                sleep_time *= backoff_factor
                time.sleep(sleep_time)

        raise Exception(f"Maximum retries {max_retries} exceeded")

    return wrapper
