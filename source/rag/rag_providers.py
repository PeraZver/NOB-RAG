from __future__ import annotations

import os
from functools import lru_cache

import anthropic
from openai import OpenAI


def build_prompts(title: str, question: str, context: str) -> tuple[str, str]:
    developer_prompt = (
        "You answer questions about one war-history book using only the provided sources. "
        "If the sources do not support a claim, say so plainly. "
        "Cite supporting pages inline in square brackets like [p. 12] or [pp. 12-14]. "
        "Do not invent citations. Prefer concise, grounded answers."
    )
    user_prompt = (
        f"Book title: {title}\n\n"
        f"Question: {question}\n\n"
        "Sources:\n"
        f"{context}\n\n"
        "Answer using the sources above and include page citations."
    )
    return developer_prompt, user_prompt


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI()


@lru_cache(maxsize=1)
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def generate_with_provider(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1600,
) -> str:
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        response = get_openai_client().responses.create(
            model=model,
            input=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        return response.output_text

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

        message = get_anthropic_client().messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )

    raise ValueError(f"Unsupported provider: {provider}")


def answer_with_provider(provider: str, model: str, title: str, question: str, context: str) -> str:
    developer_prompt, user_prompt = build_prompts(title=title, question=question, context=context)
    return generate_with_provider(
        provider=provider,
        model=model,
        system_prompt=developer_prompt,
        user_prompt=user_prompt,
        max_output_tokens=1600,
    )
