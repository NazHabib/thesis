import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from django.core.management.base import BaseCommand

from apps.recs_csl.services.cold_start.service import run_cold_start_layer


def load_candidates_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_candidates_df(raw_candidates_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_candidates_df.copy()

    if "participant_key" in df.columns:
        df = df.rename(columns={"participant_key": "ParticipantKey"})
    if "cycle" in df.columns:
        df = df.rename(columns={"cycle": "Cycle"})

    candidate_columns = [
        "action_code",
        "action_type",
        "severity",
        "priority",
        "title",
        "explanation",
        "source_rule_ids",
        "tags",
        "evidence",
        "suppressed_by",
    ]

    optional_columns = [
        "contraindications_unknown",
        "conflict_group",
        "conflict_policy",
        "bundle_family",
    ]

    keep_cols = ["ParticipantKey"] + [
        c for c in candidate_columns + optional_columns if c in df.columns
    ]
    df = df[keep_cols].copy()

    for col in ["tags", "evidence", "source_rule_ids", "suppressed_by"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x)
            )

    if "bundle_family" not in df.columns:
        df["bundle_family"] = None

    participant_keys = df["ParticipantKey"].drop_duplicates().reset_index(drop=True)

    grouped_candidates = (
        df.drop(columns=["ParticipantKey"], errors="ignore")
        .groupby(df["ParticipantKey"], sort=False)
        .apply(lambda g: g.to_dict(orient="records"))
        .reset_index(name="Candidates")
    )

    participants_df = participant_keys.to_frame(name="ParticipantKey").merge(
        grouped_candidates,
        on="ParticipantKey",
        how="left",
        validate="1:1",
    )

    return participants_df


class Command(BaseCommand):
    help = "Run cold-start layer on engineered CFM parquet + rules recommendations JSONL."

    def add_arguments(self, parser):
        parser.add_argument("--cfm", type=str, required=True, help="Path to engineered CFM parquet")
        parser.add_argument("--participants", type=str, required=True, help="Path to recommendations JSONL")
        parser.add_argument(
            "--outdir",
            type=str,
            default="apps/recs_csl/output",
            help="Output directory for cold-start results",
        )

    def handle(self, *args, **options):
        cfm_path = options["cfm"]
        participants_path = options["participants"]
        outdir = Path(options["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Loading engineered CFM parquet...")
        cfm_df = pd.read_parquet(cfm_path)

        self.stdout.write("Loading recommendations JSONL...")
        raw_candidates_df = load_candidates_jsonl(participants_path)

        self.stdout.write("Normalizing candidates into participant-level structure...")
        participants_df = normalize_candidates_df(raw_candidates_df)

        self.stdout.write("Running Cold-Start layer...")
        result = run_cold_start_layer(cfm_df, participants_df)

        safety_context_path = outdir / "safety_context.jsonl"
        gated_candidates_path = outdir / "gated_candidates.jsonl"
        manifest_path = outdir / "manifest.json"

        result["safety_context"].to_json(
            safety_context_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        result["gated_candidates"].to_json(
            gated_candidates_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        manifest = {
            **result["manifest"],
            "cfm_path": cfm_path,
            "participants_path": participants_path,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "output_files": {
                "safety_context": str(safety_context_path),
                "gated_candidates": str(gated_candidates_path),
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS("Cold-start layer executed successfully."))
        self.stdout.write(f"Safety context rows: {len(result['safety_context'])}")
        self.stdout.write(f"Gated candidate rows: {len(result['gated_candidates'])}")
        self.stdout.write(f"Saved: {safety_context_path}")
        self.stdout.write(f"Saved: {gated_candidates_path}")
        self.stdout.write(f"Saved: {manifest_path}")
