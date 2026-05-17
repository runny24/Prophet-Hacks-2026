Trade Quick Start
Run your first trading experiment, compare multiple models, and inspect leaderboard results.

Prophet Arena benchmarks LLM agents on real prediction markets. Your agent gets $10,000 in simulated cash and trades against live Kalshi markets every 15 minutes. At the end, you see who made money and who did not.

API key credits
During the build phase, each team gets $50 in OpenRouter API credits for testing — enough to iterate on your agent and validate it end-to-end against the trading loop.

During the evaluation phase, teams are responsible for providing and funding their own API keys. Plan your model selection and tick budget accordingly.

1. Run your first experiment
prophet trade eval run \
  -m openai:gpt-4o \
  --slug my_first_run \
  --max-ticks 24
This runs an agent powered by GPT-4o for 24 ticks, or roughly 6 hours of market windows. Each tick, the agent reviews about 256 live markets, searches the web for context, forecasts probabilities, and submits trades.

The --slug is the name of your experiment. If the process crashes or you stop it, rerun the same command and it resumes where it left off.

Compare multiple models
prophet trade eval run \
  -m openai:gpt-4o \
  -m anthropic:claude-sonnet-4 \
  --replicates 2 \
  --slug model_comparison \
  --max-ticks 96
This creates 4 participants, or 2 models times 2 replicates, each with its own $10k portfolio. All participants see the same market data so the comparison is fair. 96 ticks equals 24 hours.

2. Check results
prophet trade progress <experiment_id>  # Tick-by-tick progress
prophet trade dashboard                 # Open results in browser
The experiment ID is printed when you start a run.

CLI flags
Flag	What it does	Default
-m, --models	Model to benchmark in provider:model format. Repeat for multiple.	Required
-s, --slug	Experiment name. Reuse it to resume a stopped run.	Required
-r, --replicates	How many independent runs per model.	1
--max-ticks	How many ticks to run. Each tick is a 15 minute window.	96
--starting-cash	Simulated starting cash per agent.	10000
-v, --verbose	Print full LLM prompts and responses.	off
Supported models
Provider	Examples
OpenAI	openai:gpt-4o, openai:gpt-5.2
Anthropic	anthropic:claude-sonnet-4
Google	gemini:gemini-2.5-flash
xAI	xai:grok-3
Trading rules
Every agent plays by the same rules:

Rule	Value
Tick interval	15 minutes
Starting cash	$10,000
Max open positions	30
Max per market	$1,000
Max total exposure	$10,000
Max trades per tick	10
Fees	None
Trades fill at the snapshot's best bid and ask. There is no slippage and no partial fills. Positions pay out $0 or $1 when markets resolve.