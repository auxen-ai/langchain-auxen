"""
LangChain provider for Auxen.

Auxen (https://auxen.ai) hosts private AI model endpoints on dedicated
GPUs. Each instance is OpenAI-compatible — same chat-completions
request/response shape as OpenAI, just at your own per-instance URL —
so this provider is a thin wrapper over `langchain_openai.ChatOpenAI`
with Auxen-specific defaults for the base URL, auth header, and
environment variables.

Quickstart:

    from langchain_auxen import ChatAuxen

    model = ChatAuxen(
        base_url="https://api.auxen.ai/v1/inst_xxx/v1",
        api_key="auxk_...",
        model="llama-3.1-8b",
    )

    response = model.invoke("Hello")
    print(response.content)

Environment variables (used if base_url / api_key are not passed):

    AUXEN_BASE_URL   per-instance base URL from the Auxen dashboard
    AUXEN_API_KEY    per-instance bearer token (auxk_*)

All LangChain features (streaming, tool calling, structured output,
async, batch, with_structured_output, bind_tools) work via the
ChatOpenAI surface — no Auxen-specific code needed.
"""

from langchain_auxen.chat_models import ChatAuxen
from langchain_auxen._version import __version__

__all__ = ["ChatAuxen", "__version__"]
