# `langchain-auxen`

LangChain provider for [Auxen](https://auxen.ai) — private AI models on dedicated GPUs, OpenAI-compatible, pay-per-minute.

## Install

```bash
pip install langchain-auxen
```

## Quickstart

Provision an instance at [auxen.ai/dashboard](https://auxen.ai/dashboard) and copy the endpoint URL + API key.

```python
from langchain_auxen import ChatAuxen

model = ChatAuxen(
    base_url="https://api.auxen.ai/v1/inst_xxx",
    api_key="auxk_...",
    model="llama-3.1-8b",
)

response = model.invoke("Hello")
print(response.content)
```

Or with environment variables (`AUXEN_BASE_URL` + `AUXEN_API_KEY`):

```python
from langchain_auxen import ChatAuxen

model = ChatAuxen(model="llama-3.1-8b")
```

## Streaming

```python
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="")
```

## Tool calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")
```

## Structured output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

structured = model.with_structured_output(Person)
result = structured.invoke("Extract: Alice is 30 years old")
print(result)  # Person(name='Alice', age=30)
```

## Async

```python
import asyncio
from langchain_auxen import ChatAuxen

async def main():
    model = ChatAuxen()
    response = await model.ainvoke("Hello")
    print(response.content)

asyncio.run(main())
```

## How it works

`ChatAuxen` is a thin subclass of `langchain_openai.ChatOpenAI` with Auxen-specific defaults — it reads `AUXEN_BASE_URL` / `AUXEN_API_KEY` from the environment and auto-appends `/v1` to the base URL where needed. All ChatOpenAI features (streaming, batch, tool calling, structured output, async) inherit for free because Auxen is OpenAI-compatible at the wire level.

## Pricing

Auxen bills per minute of GPU runtime, not per token:

| Tier | Models | Rate (1× capacity) |
|---|---|---|
| Small (≤7B) | gemma2-2b, mistral-7b, llama3.2-3b | $0.10/hr |
| Medium (8–14B) | llama3.1-8b, qwen2.5-14b | $0.20/hr |
| Large (24–32B) | mistral-small-24b, qwen2.5-32b | $0.65/hr |
| XL (70B+) | llama3.1-70b, qwen2.5-72b | $1.50/hr |

## Links

- Auxen homepage: <https://auxen.ai>
- Documentation: <https://auxen.ai/docs>
- Source: <https://github.com/auxen-ai/langchain-auxen>
- Related: [`auxen`](https://github.com/auxen-ai/auxen-python) (standalone Python SDK), [`@auxen-ai/ai-sdk-provider`](https://github.com/auxen-ai/ai-sdk-provider) (Vercel AI SDK)

## License

Apache-2.0
