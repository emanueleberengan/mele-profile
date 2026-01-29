#bin/bash

curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2:7b

ollama serve

sudo lsof -i :11434