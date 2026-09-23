"""CLI: trade-sentiment scan/score/sources/license/update-check."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, licensing
from . import sources as src
from .adapters import to_agent_ideas, to_signal_overlay
from .pipeline import scan
from .scoring import score_text


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="JSON config file (symbols, sources, window)")


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return json.load(fh)


def cmd_scan(args, cfg: dict) -> int:
    symbols = args.symbols or cfg.get("symbols", [])
    if not symbols:
        print("no symbols given (args or config 'symbols')", file=sys.stderr)
        return 2
    pops = scan(
        symbols,
        source_names=args.sources or cfg.get("sources"),
        window_hours=args.window or cfg.get("window_hours", 24),
        min_mentions=args.min_mentions,
        limit_per_source=args.limit,
    )
    if args.format == "ideas":
        print(json.dumps(to_agent_ideas(pops), indent=2))
    elif args.format == "overlay":
        print(json.dumps(to_signal_overlay(pops), indent=2))
    else:
        for pop in pops:
            print(pop.verdict)
        if args.verbose:
            print(json.dumps([p.to_dict() for p in pops], indent=2))
    return 0


def cmd_score(args, cfg: dict) -> int:
    text = args.text or sys.stdin.read()
    sm = score_text(text)
    print(f"polarity={sm.polarity} label={sm.label.value} "
          f"magnitude={sm.magnitude}")
    if sm.hits:
        print("hits: " + ", ".join(sm.hits))
    return 0


def cmd_sources(args, cfg: dict) -> int:
    for name, cls in src.SOURCES.items():
        print(f"{name}: {cls.rate_limit_note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trade-sentiment",
                                description="Social/news sentiment pops")
    _add_config(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan symbols for sentiment pops")
    _add_config(s)
    s.add_argument("symbols", nargs="*", help="tickers, e.g. AAPL NVDA")
    s.add_argument("--sources", nargs="*", default=None,
                   choices=list(src.SOURCES),
                   help="subset of sources (default: all)")
    s.add_argument("--window", type=int, default=None,
                   help="trailing window hours (default 24)")
    s.add_argument("--min-mentions", type=int, default=5)
    s.add_argument("--limit", type=int, default=50,
                   help="mentions per source per symbol")
    s.add_argument("--format", choices=["verdicts", "json", "ideas", "overlay"],
                   default="verdicts")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("score", help="score one piece of text")
    _add_config(s)
    s.add_argument("text", nargs="?", help="text (or stdin)")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("sources", help="list available sources")
    _add_config(s)
    s.set_defaults(func=cmd_sources)

    s = sub.add_parser("license", help="license status")
    _add_config(s)
    s.set_defaults(func=lambda a, c: (print(json.dumps(
        licensing.check_license(), indent=2)), 0)[1])

    s = sub.add_parser("update-check", help="check for a newer release")
    _add_config(s)
    s.set_defaults(func=lambda a, c: (print(json.dumps(
        licensing.check_update(), indent=2)), 0)[1])

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
