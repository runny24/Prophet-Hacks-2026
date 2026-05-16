# Edge Trader Bot

Independent Prophet Arena trading bot for the RAG-BLF strategy.

## Run

Install or expose the local SDK first:

```bash
python -m pip install -e ai-prophet/packages/core
```

Then run one dry tick:

```bash
export PA_SERVER_URL=https://api.aiprophet.dev
export PA_SERVER_API_KEY=...
python -m edge_trader_bot --once --dry-run
```

Optional RAG settings:

```bash
export EDGE_TRADER_ENABLE_RAG=1
export BRAVE_SEARCH_API_KEY=...

# Optional LLM evidence summarization; guarded by fallback.
export EDGE_TRADER_ENABLE_LLM_RAG=1
export EDGE_TRADER_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=...
export EDGE_TRADER_LLM_MODEL=deepseek/deepseek-chat

# Optional stronger DeepSeek model if supported by your OpenRouter account:
# export EDGE_TRADER_LLM_MODEL=deepseek/deepseek-r1
```

## Staged Dry Runs

A. Server loop only:

```bash
EDGE_TRADER_ENABLE_RAG=0 EDGE_TRADER_ENABLE_LLM_RAG=0 python -m edge_trader_bot.runner --once --dry-run --max-markets 5
```

B. Server loop + Brave RAG:

```bash
EDGE_TRADER_ENABLE_RAG=1 EDGE_TRADER_ENABLE_LLM_RAG=0 EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK=3 python -m edge_trader_bot.runner --once --dry-run --max-markets 10
```

C. Server loop + Brave RAG + OpenRouter LLM:

```bash
EDGE_TRADER_ENABLE_RAG=1 EDGE_TRADER_ENABLE_LLM_RAG=1 EDGE_TRADER_LLM_PROVIDER=openrouter EDGE_TRADER_LLM_MODEL=deepseek/deepseek-chat EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK=3 python -m edge_trader_bot.runner --once --dry-run --max-markets 10
```

The MVP is conservative. RAG and BLF are interface stubs until their adapters are
wired in, so the bot usually holds unless a deterministic edge is unusually
large or an existing position should be exited.
