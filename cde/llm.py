"""Single LLM wrapper — backend swappable via .env.

Backend order:
  1. OPENAI_API_KEY set  -> OpenAI chat (CDE_CHAT_MODEL, default gpt-4o-mini)
                          + text-embedding-3-small for embeddings
  2. otherwise           -> `claude -p` CLI (print mode), local TF-IDF embeddings
                            handled in cde/embed.py

All extraction/judging goes through chat_json() which returns a parsed dict.
"""
import json
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env():
    env = {}
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("OPENAI_API_KEY", "CDE_CHAT_MODEL", "CDE_EMBED_MODEL", "CDE_BACKEND"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


ENV = load_env()
BACKEND = ENV.get("CDE_BACKEND") or ("openai" if ENV.get("OPENAI_API_KEY") else "claude")
CHAT_MODEL = ENV.get("CDE_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = ENV.get("CDE_EMBED_MODEL", "text-embedding-3-small")

_client = None
if BACKEND == "openai":
    from openai import OpenAI
    _client = OpenAI(api_key=ENV["OPENAI_API_KEY"])


def _extract_json(text):
    """Parse JSON leniently: strip fences, take outermost {…} or […]."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in response: {text[:200]}")


def chat(prompt, timeout=240):
    """Raw text completion."""
    if BACKEND == "openai":
        r = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return r.choices[0].message.content
    p = subprocess.run(
        ["claude", "-p", "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=timeout, cwd=ROOT,
    )
    if p.returncode != 0:
        raise RuntimeError(f"claude -p failed: {p.stderr[:300]}")
    return p.stdout


def chat_json(prompt, timeout=240, retries=2):
    """Completion parsed as JSON, with one retry appending a reminder."""
    last = None
    for attempt in range(retries + 1):
        out = chat(prompt, timeout=timeout)
        try:
            return _extract_json(out)
        except ValueError as e:
            last = e
            prompt += "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
    raise last


def embed(texts):
    """OpenAI embeddings, or None if backend has no embedding API (caller falls back)."""
    if BACKEND != "openai" or not texts:
        return None
    import numpy as np
    r = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)
