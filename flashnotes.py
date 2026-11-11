#!/usr/bin/env python3
"""FlashNotes -- turn long notes and documents into structured summaries.

The summarisation runs on Nebius Token Factory with an open model
(default: DeepSeek V4.1 Flash 1017). Only the Python standard library
is used, so the tool runs anywhere with Python 3.10+ and no installs.

Configuration (environment variables):
    NEBIUS_API_KEY   API key from the Token Factory console (required)
    NEBIUS_BASE_URL  API base URL (default: https://api.tokenfactory.nebius.com/v1)
    NEBIUS_MODEL     Model id (default: deepseek/deepseek-v4.1-flash-1017)

Examples:
    export NEBIUS_API_KEY="sk-..."
    python flashnotes.py check
    python flashnotes.py summarize examples/sample-notes.txt --lang zh
    python flashnotes.py summarize notes.txt -o summary.md --lang zh
    cat notes.txt | python flashnotes.py summarize --lang zh
    python flashnotes.py summarize notes.txt --dry-run   # no API call
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
CHUNK_CHARS = 6000
CHUNK_OVERLAP = 400
DEFAULT_MAX_TOKENS = 2048
REQUEST_TIMEOUT = 180

SYSTEM_PROMPTS = {
    "zh": (
        "你是一个会议记录整理助手。把输入的笔记整理成结构化摘要，"
        "用 Markdown 输出，包含：TL;DR（一句话）、要点、行动项（含负责人，如有）、"
        "待确认问题。只写输入里有的内容，不要编造；输入里没有的信息不要补。"
    ),
    "en": (
        "You are a meeting-notes assistant. Turn the input notes into a "
        "structured Markdown summary with: TL;DR (one sentence), Key points, "
        "Action items (with owners, if any), Open questions. Only use "
        "information present in the input; do not invent anything."
    ),
}

MERGE_PROMPTS = {
    "zh": (
        "下面是同一份笔记各部分的摘要。请合并成一份最终结构化摘要，"
        "用 Markdown 输出，包含：TL;DR（一句话）、要点、行动项（含负责人，如有）、"
        "待确认问题。合并重复内容，保持简洁；只写摘要里有的内容，不要编造。"
    ),
    "en": (
        "Below are the summaries of each part of the same notes. Merge them "
        "into one final structured Markdown summary with: TL;DR (one sentence), "
        "Key points, Action items (with owners, if any), Open questions. "
        "Deduplicate and keep it concise; do not invent anything."
    ),
}

PART_LABELS = {
    "zh": "第 {i}/{n} 部分笔记：\n\n{chunk}\n\n请按上述格式输出本部分的摘要。",
    "en": "Part {i} of {n}:\n\n{chunk}\n\nSummarize this part in the format described.",
}


def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split text into chunks on paragraph boundaries, with a small overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) == 1:
        paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) > size:
            # Very long paragraph: hard-slice it, keeping overlap.
            start = 0
            while start < len(para):
                end = min(len(para), start + size)
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = end - overlap
            current = ""
        else:
            current = para
    if current:
        chunks.append(current)
    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, chunk in zip(chunks, chunks[1:]):
            if len(prev) > overlap:
                overlapped.append(prev[-overlap:] + "\n\n" + chunk)
            else:
                overlapped.append(chunk)
        return overlapped
    return chunks


def estimate_tokens(text):
    """Rough token estimate (about 4 characters per token)."""
    return max(1, len(text) // 4)


def extract_content(data):
    """Pull the assistant message out of a /chat/completions response."""
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected API response: {data!r}") from exc


def chat(messages, model, base_url, api_key, max_tokens,
         dry_run=False, timeout=REQUEST_TIMEOUT):
    """Send one chat-completions request; return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if dry_run:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return extract_content(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        hint = {
            401: "check NEBIUS_API_KEY",
            404: "check NEBIUS_MODEL / base URL",
            429: "rate limit or capacity -- retry later",
        }.get(exc.code, "")
        raise RuntimeError(
            f"API request failed (HTTP {exc.code}): {body} {hint}".strip()
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {base_url}: {exc.reason}") from exc


def summarize_text(text, lang, model, base_url, api_key, max_tokens,
                   dry_run=False, log=None):
    """Summarize long text: chunk -> summarize each part -> merge."""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("no text to summarize")
    part_summaries = []
    for i, chunk in enumerate(chunks, 1):
        if log:
            log(f"[{i}/{len(chunks)}] summarizing {len(chunk)} chars "
                f"(~{estimate_tokens(chunk)} tokens)...")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[lang]},
            {"role": "user", "content": PART_LABELS[lang].format(
                i=i, n=len(chunks), chunk=chunk)},
        ]
        part_summaries.append(
            chat(messages, model, base_url, api_key, max_tokens, dry_run))
    if len(part_summaries) == 1:
        return part_summaries[0]
    if log:
        log("merging part summaries...")
    messages = [
        {"role": "system", "content": MERGE_PROMPTS[lang]},
        {"role": "user", "content": "\n\n---\n\n".join(part_summaries)},
    ]
    return chat(messages, model, base_url, api_key, max_tokens, dry_run)


def read_input(path):
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def require_key(args):
    api_key = args.api_key or os.environ.get("NEBIUS_API_KEY")
    if not api_key and not args.dry_run:
        print("error: NEBIUS_API_KEY is not set (or pass --api-key).\n"
              "Get a key from the Token Factory console (API Keys page).",
              file=sys.stderr)
        return None
    # In --dry-run mode no key is needed; return "" so callers proceed.
    return api_key or ""


def cmd_summarize(args):
    api_key = require_key(args)
    if api_key is None:
        return 2
    text = read_input(args.input)
    if not text.strip():
        print("error: input is empty", file=sys.stderr)
        return 2
    try:
        result = summarize_text(
            text, args.lang, args.model, args.base_url, api_key,
            args.max_tokens, args.dry_run,
            log=lambda m: print(m, file=sys.stderr))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not result.endswith("\n"):
        result += "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(result, end="")
    return 0


def cmd_check(args):
    api_key = require_key(args)
    if api_key is None:
        return 2
    try:
        reply = chat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            args.model, args.base_url, api_key, 16, args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"model reachable: {args.model}")
    print(f"reply: {reply.strip()}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="flashnotes",
        description="Summarize long notes with DeepSeek V4.1 Flash "
                    "on Nebius Token Factory.")
    parser.add_argument("--version", action="version",
                        version="flashnotes 0.1.0")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", default=None,
                        help="Token Factory API key (or set NEBIUS_API_KEY)")
    common.add_argument("--base-url",
                        default=os.environ.get("NEBIUS_BASE_URL",
                                               DEFAULT_BASE_URL),
                        help="API base URL")
    common.add_argument("--model",
                        default=os.environ.get("NEBIUS_MODEL", DEFAULT_MODEL),
                        help="model id")
    common.add_argument("--dry-run", action="store_true",
                        help="print the request payload instead of calling "
                             "the API")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summarize", parents=[common],
                           help="summarize a text file or stdin")
    p_sum.add_argument("input", nargs="?", default=None,
                       help="input file (omit or '-' to read stdin)")
    p_sum.add_argument("-o", "--output", default=None,
                       help="write the summary to this file")
    p_sum.add_argument("--lang", choices=["zh", "en"], default="zh",
                       help="output language (default: zh)")
    p_sum.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help="max output tokens per request (default: 2048)")
    p_sum.set_defaults(func=cmd_summarize)
    p_check = sub.add_parser("check", parents=[common],
                             help="verify the API key and model are reachable")
    p_check.set_defaults(func=cmd_check)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
