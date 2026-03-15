# src/api/claude_client.py

import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
DEFAULT_MODEL = "claude-sonnet-4-6"

def ask_claude(prompt: str, model: str = DEFAULT_MODEL,max_tokens: int = 600, temperature: float = 0.0) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


