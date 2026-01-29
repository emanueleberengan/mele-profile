from smolagents import LiteLLMModel
from litellm import completion


response = completion(
    model="ollama_chat/qwen2:7b",                 # LiteLLM model name for Ollama
    api_base="http://127.0.0.1:11434",            # Ollama server URL
    messages=[{"role": "user", "content": "Does a cat with lion legs exists?"}],
    num_ctx=8192
)

print(response.choices[0].message["content"])
