from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from .engine import RuleEngine, explain_participant
from .utils import (
    load_rules,
    setup_logger,
    write_csv,
    write_json,
    write_jsonl,
    write_manifest,
)


def _load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"Data file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p, low_memory=False)

    raise click.ClickException("Supported input formats are .parquet and .csv")


@click.group()
def main() -> None:
    """
    Rules & Safety Engine CLI.
    """
    pass


@main.command("validate")
@click.option("--rules", "rules_path", required=True, type=str, help="Path to YAML/JSON ruleset.")
def validate_ruleset(rules_path: str) -> None:
    """
    Validate ruleset loading and rule object creation.
    """
    logger = setup_logger(level="INFO")
    try:
        rules = load_rules(rules_path)
        logger.info(f"Loaded {len(rules)} rules from {rules_path}")
        for r in rules:
            r.validate()
        click.echo(json.dumps({
            "status": "ok",
            "rules_loaded": len(rules),
            "rules_path": rules_path,
        }, indent=2))
    except Exception as e:
        raise click.ClickException(str(e))


@main.command("run")
@click.option("--rules", "rules_path", required=True, type=str, help="Path to YAML/JSON ruleset.")
@click.option("--data", "data_path", required=True, type=str, help="Path to engineered .parquet or .csv input.")
@click.option("--out", "out_dir", required=True, type=str, help="Output directory.")
@click.option("--log-level", default="INFO", type=str, help="Logging level.")
@click.option("--limit", default=None, type=int, help="Optional row limit for debugging.")
def run_engine(rules_path: str, data_path: str, out_dir: str, log_level: str, limit: Optional[int]) -> None:
    """
    Run engine on input dataframe and write outputs.
    """
    logger = setup_logger(level=log_level)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    try:
        rules = load_rules(rules_path)
        logger.info(f"Loaded {len(rules)} rules from {rules_path}")

        df = _load_dataframe(data_path)
        if limit is not None:
            df = df.head(limit).copy()
            logger.info(f"Applying debug row limit: {limit}")

        logger.info(f"Loaded dataframe: {df.shape[0]} rows x {df.shape[1]} cols")

        engine = RuleEngine(rules=rules)
        result = engine.evaluate(df=df, input_path=data_path)

        # outputs
        fired_rules_path = out_path / "fired_rules.csv"
        recommendations_path = out_path / "recommendations.jsonl"
        conflicts_path = out_path / "conflicts.jsonl"
        manifest_path = out_path / "manifest.json"
        warnings_path = out_path / "warnings.json"

        write_csv(result.fired_rules, fired_rules_path)
        write_jsonl(result.recommendations, recommendations_path)
        write_jsonl(result.conflicts, conflicts_path)
        write_manifest(result.manifest, manifest_path)
        write_json({"warnings": result.warnings}, warnings_path)

        elapsed = time.perf_counter() - start

        summary = {
            "status": "ok",
            "rows_processed": result.manifest.rows_processed,
            "rules_loaded": result.manifest.rules_loaded,
            "fired_rules_count": len(result.fired_rules),
            "recommendations_count": len(result.recommendations),
            "conflicts_count": len(result.conflicts),
            "elapsed_seconds": round(elapsed, 4),
            "outputs": {
                "fired_rules_csv": str(fired_rules_path),
                "recommendations_jsonl": str(recommendations_path),
                "conflicts_jsonl": str(conflicts_path),
                "manifest_json": str(manifest_path),
                "warnings_json": str(warnings_path),
            },
        }

        click.echo(json.dumps(summary, indent=2))
    except Exception as e:
        raise click.ClickException(str(e))


@main.command("explain")
@click.option("--rules", "rules_path", required=True, type=str, help="Path to YAML/JSON ruleset.")
@click.option("--data", "data_path", required=True, type=str, help="Path to engineered .parquet or .csv input.")
@click.option("--participant", "participant_key", required=True, type=str, help="ParticipantKey to explain.")
@click.option("--limit", default=None, type=int, help="Optional row limit for debugging.")
def explain_one(rules_path: str, data_path: str, participant_key: str, limit: Optional[int]) -> None:
    """
    Explain one participant by rerunning the engine and filtering outputs.
    """
    try:
        rules = load_rules(rules_path)
        df = _load_dataframe(data_path)
        if limit is not None:
            df = df.head(limit).copy()

        engine = RuleEngine(rules=rules)
        result = engine.evaluate(df=df, input_path=data_path)
        payload = explain_participant(result, participant_key)
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        raise click.ClickException(str(e))


@main.command("benchmark")
@click.option("--rules", "rules_path", required=True, type=str, help="Path to YAML/JSON ruleset.")
@click.option("--data", "data_path", required=True, type=str, help="Path to engineered .parquet or .csv input.")
@click.option("--repeat", default=3, type=int, help="Number of benchmark repetitions.")
@click.option("--limit", default=None, type=int, help="Optional row limit for benchmarking.")
def benchmark(rules_path: str, data_path: str, repeat: int, limit: Optional[int]) -> None:
    """
    Benchmark engine runtime.
    """
    if repeat < 1:
        raise click.ClickException("--repeat must be >= 1")

    try:
        rules = load_rules(rules_path)
        df = _load_dataframe(data_path)
        if limit is not None:
            df = df.head(limit).copy()

        timings = []
        for _ in range(repeat):
            start = time.perf_counter()
            engine = RuleEngine(rules=rules)
            _ = engine.evaluate(df=df, input_path=data_path)
            timings.append(time.perf_counter() - start)

        payload = {
            "status": "ok",
            "repeat": repeat,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "timings_seconds": [round(x, 6) for x in timings],
            "min_seconds": round(min(timings), 6),
            "max_seconds": round(max(timings), 6),
            "mean_seconds": round(sum(timings) / len(timings), 6),
        }
        click.echo(json.dumps(payload, indent=2))
    except Exception as e:
        raise click.ClickException(str(e))
        
if __name__ == "__main__":
    main()
