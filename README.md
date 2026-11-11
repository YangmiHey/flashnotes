# FlashNotes

Turn long notes and documents into structured summaries with
DeepSeek V4.1 Flash running on
[Nebius Token Factory](https://dev.nebius.com/builders).

FlashNotes reads a text file (or stdin), splits long inputs into
chunks, asks the model to summarise each part, then merges the parts
into one Markdown summary: **TL;DR, key points, action items, open
questions**. It works in Chinese and English, needs no third-party
packages — Python 3.10+ and the standard library are enough.

## Why

Long meeting notes are hard to re-read and easy to lose. FlashNotes
keeps the summarisation pipeline small and inspectable: plain prompts,
visible chunking, and every output grounded in the input text (the
model is told not to invent anything).

## Quick start

```bash
# 1. Get an API key from the Nebius Token Factory console (API Keys page)
export NEBIUS_API_KEY="sk-..."

# 2. Check the key and the model
python flashnotes.py check

# 3. Summarize the bundled example (Chinese)
python flashnotes.py summarize examples/sample-notes.txt --lang zh

# 4. Summarize your own file, save as Markdown
python flashnotes.py summarize notes.txt -o summary.md --lang zh
```

No install step: there are no dependencies.

## Usage

```bash
# Read from stdin
cat notes.txt | python flashnotes.py summarize --lang zh

# English output
python flashnotes.py summarize notes.txt --lang en -o summary.md

# See the exact request that would be sent, without calling the API
python flashnotes.py summarize notes.txt --dry-run

# Use a different model or base URL
NEBIUS_MODEL="deepseek/deepseek-v3.2" python flashnotes.py summarize notes.txt
```

Options: `-o/--output`, `--lang zh|en` (default `zh`),
`--max-tokens` (default 2048), `--dry-run`.

## How it works

1. **Chunking** — text is split at paragraph boundaries into ~6,000
   character chunks with a 400 character overlap, so context is not
   lost at the seams.
2. **Summarize** — each chunk is summarised on its own with a
   fixed-format prompt (TL;DR, key points, action items, open
   questions).
3. **Merge** — the part summaries are combined into one final
   Markdown document, deduplicated.

The default model is `deepseek/deepseek-v4.1-flash-1017` on
`https://api.tokenfactory.nebius.com/v1` (OpenAI-compatible
`/chat/completions`). Token counts in progress messages are rough
estimates (`len(text) // 4`).

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API key | `NEBIUS_API_KEY` | — (required) |
| Base URL | `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` |
| Model | `NEBIUS_MODEL` | `deepseek/deepseek-v4.1-flash-1017` |

You can also pass `--api-key`, `--base-url`, `--model` on the command line.

## Project layout

```
flashnotes.py                  the CLI (stdlib only)
tests/test_flashnotes.py       unit tests for the pure functions
examples/sample-notes.txt      example input (fictional weekly notes)
examples/sample-output.md      example output (illustrative)
pyproject.toml                 packaging metadata (no dependencies)
```

## Testing

```bash
python -m unittest discover -s tests -v
```

Tests cover chunking, prompts and response parsing only — they never
call the API.

## License

MIT — see `LICENSE`.
