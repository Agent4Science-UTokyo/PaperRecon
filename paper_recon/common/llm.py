from __future__ import annotations

import json
import re
import sys
from typing import Any, TypeGuard, overload

import backoff
import litellm
from litellm import Choices, ModelResponse, completion
from litellm.exceptions import BadRequestError, NotFoundError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from paper_recon.common.log import get_logger

logger = get_logger(__file__)

MAX_NUM_TOKENS = 8192
Prompt = (
    ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
)
litellm.drop_params = True
litellm.suppress_debug_info = True


def UserPrompt(  # noqa: N802
    content: str | list[ChatCompletionContentPartParam],
) -> Prompt:
    return ChatCompletionUserMessageParam(
        content=content,
        role="user",
    )


def SystemPrompt(content: str) -> Prompt:  # noqa: N802
    return ChatCompletionSystemMessageParam(role="system", content=content)


def AssistantPrompt(content: str) -> Prompt:  # noqa: N802
    return ChatCompletionAssistantMessageParam(role="assistant", content=content)


def is_str_list(lst: list[Any]) -> TypeGuard[list[str]]:
    return all(isinstance(item, str) for item in lst)


def get_batch_responses_from_llm(
    model: str,
    user_message: str,
    system_message: str,
    msg_history: list[Prompt] | None = None,
    temperature: float | None = None,
    n_responses: int = 1,
    response_format: type[BaseModel] | None = None,
) -> tuple[list[str], list[list[Prompt]]]:
    """
    Call LLM completions and return batch responses.

    Args:
        model (str): model name
        user_message (str): user message text
        system_message (str): system message text
        msg_history (list[Prompt] | None): previous message history. Defaults to None.
        temperature (float | None): temperature for sampling. Defaults to None.
        n_responses (int): number of responses to generate. Defaults to 1.
        response_format (type[BaseModel] | None): Pydantic model class to parse the response. Defaults to None.

    Returns:
        tuple[list[str], list[list[Prompt]]]: generated contents and new message histories

    """
    messages = []
    if msg_history:
        messages.extend(msg_history)
    messages.append(UserPrompt(user_message))

    # GPT-5 only supports temperature=1.0
    if "gpt-5" in model and temperature != 1.0:
        logger.debug("Overriding temperature %.1f -> 1.0 for model %s", temperature or 0.0, model)
        temperature = 1.0

    # Switch Azure API key and base URL based on the model name
    extra_kwargs: dict[str, Any] = {}
    if "gpt-5.4" in model or "gpt-54" in model:
        import os

        api_key = os.environ.get("AZURE_GPT54_API_KEY")
        api_base = os.environ.get("AZURE_GPT54_API_BASE")
        if api_key:
            extra_kwargs["api_key"] = api_key
        if api_base:
            extra_kwargs["api_base"] = api_base

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[SystemPrompt(system_message), *messages],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
                response_format=response_format,
                **extra_kwargs,
            )
            break
        except NotFoundError:
            logger.critical(
                "Model %s not found. Please check your model name and API configuration.", model
            )
            sys.exit(1)
        except BadRequestError as e:
            if "internal error" in str(e).lower() and attempt < max_retries - 1:
                import time

                wait = 10 * (attempt + 1)
                logger.warning(
                    "Azure internal error (attempt %d/%d), retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    assert isinstance(response, ModelResponse)
    content = [c.message.content for c in response.choices if isinstance(c, Choices)]
    assert is_str_list(content)

    # new msg histories for each response
    new_msg_histories = [[*messages, AssistantPrompt(c)] for c in content]
    return content, new_msg_histories


@backoff.on_exception(backoff.constant, ValidationError, max_tries=3, interval=0)
def get_batch_responses_from_llm_with_validation[T: BaseModel](
    model: str,
    user_message: str,
    system_message: str,
    response_format: type[T],
    n_responses: int,
    temperature: float | None = None,
    msg_history: list[Prompt] | None = None,
) -> tuple[list[T], list[list[Prompt]]]:
    responses, msg_histories = get_batch_responses_from_llm(
        model=model,
        user_message=user_message,
        system_message=system_message,
        msg_history=msg_history,
        temperature=temperature,
        n_responses=n_responses,
        response_format=response_format,
    )
    models = [response_format.model_validate_json(r) for r in responses]
    return models, msg_histories


@overload
def get_response_from_llm(
    model: str,
    user_message: str,
    system_message: str,
    response_format: None = None,
    temperature: float | None = None,
    msg_history: list[Prompt] | None = None,
) -> tuple[str, list[Prompt]]: ...


@overload
def get_response_from_llm[T: BaseModel](
    model: str,
    user_message: str,
    system_message: str,
    response_format: type[T],
    temperature: float | None = None,
    msg_history: list[Prompt] | None = None,
) -> tuple[T, list[Prompt]]: ...


def get_response_from_llm[T: BaseModel](
    model: str,
    user_message: str,
    system_message: str,
    response_format: type[T] | None = None,
    temperature: float | None = None,
    msg_history: list[Prompt] | None = None,
) -> tuple[str | T, list[Prompt]]:
    """
    Call LLM completion and return response and new message history.

    Args:
        model (str): model name
        user_message (str): user message text
        system_message (str): system message text
        response_format (type[BaseModel] | None, optional): Pydantic model class to parse the response. Defaults to None.
        temperature (float | None): temperature for sampling. Defaults to None.
        msg_history (list[Prompt] | None, optional): previous message history. Defaults to None.

    Returns:
        tuple[str| BaseModel, list[Prompt]]: generated content and new message history

    """
    logger.debug("=" * 40 + " LLM REQUEST " + "=" * 40)
    logger.debug("[System Prompt]\n%s", system_message)
    logger.debug("[User Prompt]\n%s", user_message)
    logger.debug(
        "[Model] %s  [Temperature] %s  [Response Format] %s",
        model,
        temperature,
        response_format.__name__ if response_format else "None",
    )
    logger.debug("=" * 93)

    if response_format is None:
        responses, msg_histories = get_batch_responses_from_llm(
            model=model,
            user_message=user_message,
            system_message=system_message,
            msg_history=msg_history,
            temperature=temperature,
            n_responses=1,
            response_format=response_format,
        )
    else:
        responses, msg_histories = get_batch_responses_from_llm_with_validation(
            model=model,
            user_message=user_message,
            system_message=system_message,
            response_format=response_format,
            msg_history=msg_history,
            temperature=temperature,
            n_responses=1,
        )
    result = responses[0]
    logger.debug("=" * 40 + " LLM RESPONSE " + "=" * 39)
    logger.debug("[Response]\n%s", result)
    logger.debug("=" * 93)

    return result, msg_histories[0]


def extract_json_between_markers(llm_output: str) -> dict:
    """
    Extract JSON content from LLM output between ```json and ``` markers.

    Args:
        llm_output (str): The output string from the LLM.

    Returns:
        dict: The extracted JSON content as a dictionary. \
            Returns an empty dictionary if no valid JSON is found.

    """
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    assert is_str_list(matches)

    for json_string in matches:
        clean_json_str = json_string.strip()
        try:
            parsed_json = dict(json.loads(clean_json_str))
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", clean_json_str)
                parsed_json = json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue  # Try next match
            else:
                return parsed_json
        else:
            return parsed_json

    return {}  # No valid JSON found
