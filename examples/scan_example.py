"""Example: scan tickers and print verdicts + agent ideas (network needed)."""
from trade_sentiment import scan
from trade_sentiment.adapters import to_agent_ideas

if __name__ == "__main__":
    pops = scan(["AAPL", "NVDA", "TSLA"])
    for pop in pops:
        print(pop.verdict)
    print(to_agent_ideas(pops))
