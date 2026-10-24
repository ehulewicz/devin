"""Single LLM wrapper — backend swappable via .env.

Backend order (CDE_BACKEND in .env forces one):
  1. OPENAI_API_KEY set        -> OpenAI chat (CDE_CHAT_MODEL, default gpt-4o-mini)
                                + text-embedding-3-small for embeddings
  2. `codex` CLI on PATH       -> `codex exec -o <outfile>` (ChatGPT-account CLI)
  3. `claude` CLI on PATH      -> `claude -p` print mode
Embeddings exist only on the OpenAI backend; otherwise callers fall back to the
local hashed TF-IDF in cde/embed.py.

chat_json() parses lenient JSON and falls through to the next backend on failure.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

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
CHAT_MODEL = ENV.get("CDE_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = ENV.get("CDE_EMBED_MODEL", "text-embedding-3-small")

_available = []
if ENV.get("OPENAI_API_KEY"):
    _available.append("openai")
if shutil.which("codex"):
    _available.append("codex")
if shutil.which("claude"):
    _available.append("claude")

BACKEND = ENV.get("CDE_BACKEND") or (_available[0] if _available else "none")

_client = None
if "openai" in _available:
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


def _chat_openai(prompt, timeout):
    r = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return r.choices[0].message.content


def _chat_codex(prompt, timeout):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        outf = f.name
    try:
        p = subprocess.run(
            ["codex", "exec", "-c", 'model_reasoning_effort="low"',
             "--sandbox", "read-only", "-o", outf, prompt],
            capture_output=True, text=True, timeout=timeout, cwd=ROOT,
        )
        if p.returncode != 0:
            raise RuntimeError(f"codex exec failed: {(p.stderr or p.stdout)[:300]}")
        out = open(outf).read().strip()
        if not out:  # fall back to parsing stdout tail
            m = re.split(r"tokens used\s*\n[\d,]+\s*\n", p.stdout)
            out = m[-1].strip() if len(m) > 1 else p.stdout.strip()
        if "session limit" in out.lower() or "usage limit" in out.lower():
            raise RuntimeError(f"codex limited: {out[:200]}")
        return out
    finally:
        os.path.exists(outf) and os.unlink(outf)


def _chat_claude(prompt, timeout):
    p = subprocess.run(
        ["claude", "-p", "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=timeout, cwd=ROOT,
    )
    if p.returncode != 0:
        raise RuntimeError(f"claude -p failed: {p.stderr[:300]}")
    if "session limit" in p.stdout.lower() or "usage limit" in p.stdout.lower():
        raise RuntimeError(f"claude limited: {p.stdout[:200]}")
    return p.stdout


_FUNCS = {"openai": _chat_openai, "codex": _chat_codex, "claude": _chat_claude}
_working = []  # backends that have succeeded, tried first


def chat(prompt, timeout=240):
    """Raw text completion; tries each backend in order, sticky on success."""
    order = _working + [b for b in _available if b not in _working]
    if ENV.get("CDE_BACKEND"):
        order = [ENV["CDE_BACKEND"]] + [b for b in order if b != ENV["CDE_BACKEND"]]
    last = None
    for b in order:
        try:
            out = _FUNCS[b](prompt, timeout)
            if b not in _working:
                _working.append(b)
            return out
        except Exception as e:
            last = e
    raise last or RuntimeError("no LLM backend available")


def chat_json(prompt, timeout=240, retries=1):
    """Completion parsed as JSON; retries once with a nudge, then next backend."""
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
    if "openai" not in _available or not texts:
        return None
    import numpy as np
    r = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)
