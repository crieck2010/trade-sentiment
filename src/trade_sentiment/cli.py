"""CLI: trade-sentiment scan/score/sources/archive/license/update-check."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from . import __version__, licensing
from . import sources as src
from .adapters import to_agent_ideas, to_signal_overlay
from .archive import Archive, DEFAULT_ARCHIVE_PATH
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
    rescorer = None
    if (args.rescore or cfg.get("rescore", "none")) == "finbert":
        from .finbert import finbert_rescorer
        rescorer = finbert_rescorer(
            blend=args.blend if args.blend is not None
            else cfg.get("blend", 0.5))
    archive = None
    if args.archive or cfg.get("archive"):
        archive = Archive(args.archive or cfg["archive"])
    pops = scan(
        symbols,
        source_names=args.sources or cfg.get("sources"),
        window_hours=args.window or cfg.get("window_hours", 24),
        min_mentions=args.min_mentions,
        limit_per_source=args.limit,
        rescorer=rescorer,
        archive=archive,
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
    model = args.model or cfg.get("model", "lexicon")
    if model == "finbert":
        from .finbert import FinBERTError, FinBERTScorer
        try:
            polarity = FinBERTScorer()(text)
        except FinBERTError as exc:
            print(f"finbert unavailable: {exc}", file=sys.stderr)
            return 3
        from .models import label_for
        label = label_for(polarity)
        print(f"polarity={polarity:.4f} label={label.value} (finbert, opaque)")
        return 0
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


def cmd_archive_query(args, cfg: dict) -> int:
    arc = Archive(args.db)
    try:
        obs = arc.query(args.symbol, args.as_of, sources=args.sources,
                        limit=args.limit)
    except ValueError as exc:
        print(f"bad --as-of value: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps([o.to_dict() for o in obs], indent=2))
    else:
        for o in obs:
            pol = "n/a" if o.polarity is None else f"{o.polarity:+.2f}"
            print(f"{o.observed_at:%Y-%m-%d %H:%M}Z {o.symbol} "
                  f"{o.source:10s} polarity={pol} {o.text[:80]}")
    return 0


def cmd_archive_coverage(args, cfg: dict) -> int:
    arc = Archive(args.db)
    cov = arc.coverage(args.symbol)
    if cov is None:
        print(json.dumps(
            {"covered": False,
             "note": "archive is empty: no sentiment history exists "
                     "before deployment; history is never fabricated"},
            indent=2))
        return 0
    cov["covered"] = True
    print(json.dumps(cov, indent=2))
    return 0


def cmd_archive_prune(args, cfg: dict) -> int:
    arc = Archive(args.db)
    removed = arc.prune(args.days)
    print(f"pruned {removed} observations older than {args.days} days")
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
    s.add_argument("--rescore", choices=["none", "finbert"], default=None,
                   help="blend lexicon tone with FinBERT (needs torch + "
                        "transformers; default none)")
    s.add_argument("--blend", type=float, default=None,
                   help="FinBERT weight 0..1 (default 0.5)")
    s.add_argument("--archive", default=None, metavar="DB",
                   help="record every scored mention into the point-in-time "
                        "archive (SQLite; default off)")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("score", help="score one piece of text")
    _add_config(s)
    s.add_argument("text", nargs="?", help="text (or stdin)")
    s.add_argument("--model", choices=["lexicon", "finbert"],
                   default=None, help="scorer (default lexicon)")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("sources", help="list available sources")
    _add_config(s)
    s.set_defaults(func=cmd_sources)

    s = sub.add_parser("archive", help="point-in-time archive queries")
    _add_config(s)
    asub = s.add_subparsers(dest="archive_cmd", required=True)

    q = asub.add_parser("query", help="observations knowable as of a time")
    q.add_argument("symbol", help="ticker, e.g. AAPL")
    q.add_argument("--as-of", required=True, metavar="ISO",
                   help="decision time; only observations recorded at or "
                        "before this are returned (no-lookahead contract)")
    q.add_argument("--db", default=str(DEFAULT_ARCHIVE_PATH),
                   help="archive database path")
    q.add_argument("--sources", nargs="*", default=None,
                   help="restrict to these sources")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--format", choices=["text", "json"], default="text")
    q.set_defaults(func=cmd_archive_query)

    c = asub.add_parser("coverage",
                        help="what the archive covers (honest gap statement)")
    c.add_argument("symbol", nargs="?", default=None, help="ticker (optional)")
    c.add_argument("--db", default=str(DEFAULT_ARCHIVE_PATH),
                   help="archive database path")
    c.set_defaults(func=cmd_archive_coverage)

    pr = asub.add_parser("prune",
                         help="delete observations older than N days")
    pr.add_argument("--db", default=str(DEFAULT_ARCHIVE_PATH),
                    help="archive database path")
    pr.add_argument("--days", type=int, required=True,
                    help="keep observations observed within the last N days")
    pr.set_defaults(func=cmd_archive_prune)
    s.set_defaults(func=lambda a, c: 0)  # unreachable; subcommand required

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
