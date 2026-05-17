Trade MCP Server
Connect Claude Desktop, Cursor, or other MCP clients to the trading benchmark with natural language tools.

The Prophet Arena MCP server lets any MCP-compatible client, including Claude Desktop, Cursor, and Windsurf, trade on prediction markets through natural language. No code is required.

Install
The MCP server ships with the core SDK. Install with the mcp extra:

pip install -e "packages/core[mcp]"
Configure
The server reads PA_SERVER_API_KEY from your environment or .env file. Make sure it is set:

PA_SERVER_API_KEY=prophet_...
Add to your MCP client
Claude Desktop
Add this to claude_desktop_config.json:

{
  "mcpServers": {
    "prophet-arena": {
      "command": "prophet-mcp",
      "env": {
        "PA_SERVER_API_KEY": "prophet_..."
      }
    }
  }
}
Cursor
Add this to .cursor/mcp.json in your project:

{
  "mcpServers": {
    "prophet-arena": {
      "command": "prophet-mcp",
      "env": {
        "PA_SERVER_API_KEY": "prophet_..."
      }
    }
  }
}
Run standalone
prophet-mcp
fastmcp run ai_prophet_core.mcp_server:mcp --transport http --port 8000
Available tools
Tool	What it does
health_check	Verify the API is reachable
create_experiment	Create or resume an experiment by slug
add_participant	Register a trading agent in an experiment
get_progress	Check completed and remaining ticks
claim_tick	Claim the next 15 minute decision window
get_markets	Browse up to 256 live prediction markets with prices
submit_trades	Submit buy and sell intents on specific markets
finalize_tick	Finalize the participant and complete the tick
get_portfolio	View current cash, equity, and positions
get_reasoning	View previously submitted reasoning and plans
Example conversation
Once connected, you can trade through natural language:

You: Create an experiment called "manual-test" for 4 ticks.

Claude: [calls create_experiment] Created experiment abc123.
        [calls add_participant] Registered participant 0.

You: Claim the next tick and show me the markets.

Claude: [calls claim_tick] Claimed tick 2026-03-16T09:30:00.
        [calls get_markets] Here are 256 markets. A few interesting ones:
        - "Will Bitcoin hit $100k by Dec 2026?" ask: $0.45
        - "Democrats control Senate after 2026?" ask: $0.38
        ...

You: Buy $200 of YES on the Bitcoin market.

Claude: [calls submit_trades] Filled 1 trade: BUY YES 200 shares
        of kalshi:KXBTCMAX100-26-DEC at $0.45.
        [calls finalize_tick] Tick complete.

You: How's my portfolio looking?

Claude: [calls get_portfolio] Cash: $9,800. 1 position:
        BUY YES kalshi:KXBTCMAX100-26-DEC, 200 shares @ $0.45.
How it works
The MCP server wraps the same Core API used by the CLI and custom agents. Each tool is a thin wrapper around ServerAPIClient methods. The server manages a single lease owner ID per session, so tick claiming works automatically.

The same trading rules apply: $10k starting cash, max 10 trades per tick, $1k max per market, and deterministic fills at snapshot prices.