"""
ChatAuxen — LangChain chat model for Auxen instances.

Subclasses langchain_openai.ChatOpenAI with Auxen-specific defaults.
All LangChain ChatModel features (streaming, tool calling, structured
output, with_structured_output, bind_tools, async, batch) work as-is
because Auxen is OpenAI-compatible at the wire level.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import Field, model_validator


class ChatAuxen(ChatOpenAI):
    """LangChain chat model backed by an Auxen instance.

    Auxen issues a per-instance ``base_url`` and ``api_key`` at
    provision time. This class reads those from the AUXEN_BASE_URL
    and AUXEN_API_KEY environment variables if not passed explicitly.

    Example:
        .. code-block:: python

            from langchain_auxen import ChatAuxen

            model = ChatAuxen(
                base_url="https://api.auxen.ai/v1/inst_xxx/v1",
                api_key="auxk_...",
                model="llama-3.1-8b",
                temperature=0.7,
            )
            response = model.invoke("Hello")
    """

    # Re-declare these to give them AUXEN_* environment-variable
    # defaults — pydantic's Field validation runs before our
    # validator, so this lets us avoid the "OPENAI_API_KEY is missing"
    # error users hit if they only set AUXEN_API_KEY.
    auxen_base_url: Optional[str] = Field(default=None, exclude=True)
    auxen_api_key: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_auxen_env(cls, values: Any) -> Any:
        """Translate AUXEN_* env vars into the OpenAI client fields.

        Runs before ChatOpenAI's own validation. Reads AUXEN_BASE_URL
        and AUXEN_API_KEY from the environment (or from the
        constructor kwargs) and maps them to openai_api_base /
        openai_api_key, which the parent class understands. Also
        accepts ``base_url`` / ``api_key`` as parameter names for
        ergonomics.
        """
        if not isinstance(values, dict):
            return values

        # Resolve base_url
        base_url = (
            values.get("base_url")
            or values.get("openai_api_base")
            or values.get("auxen_base_url")
            or os.environ.get("AUXEN_BASE_URL")
        )
        if base_url:
            # Auxen instances expose the OpenAI-compatible endpoint
            # under /v1/chat/completions relative to the per-instance
            # base. ChatOpenAI appends /chat/completions, so base_url
            # needs to end at /v1.
            url = base_url.rstrip("/")
            if not url.endswith("/v1"):
                url = url + "/v1"
            values["openai_api_base"] = url
            # Pop the aliases so Pydantic's later field-alias resolution
            # doesn't overwrite our normalized openai_api_base with the
            # original unprocessed string.
            values.pop("base_url", None)
            values.pop("auxen_base_url", None)

        # Resolve api_key
        api_key = (
            values.get("api_key")
            or values.get("openai_api_key")
            or values.get("auxen_api_key")
            or os.environ.get("AUXEN_API_KEY")
        )
        if api_key:
            values["openai_api_key"] = api_key
            values.pop("api_key", None)
            values.pop("auxen_api_key", None)

        # Default to a sensible model if none is given. The model name
        # is mostly cosmetic — Auxen routes inference to whichever
        # model the instance was provisioned with, regardless of what
        # the client passes here. We still want a sensible default
        # so the LangChain init validator doesn't complain.
        if not values.get("model") and not values.get("model_name"):
            values["model"] = "llama-3.1-8b"

        return values

    @property
    def _llm_type(self) -> str:
        return "auxen-chat"
