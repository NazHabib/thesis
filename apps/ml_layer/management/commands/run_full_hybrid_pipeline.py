import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.recs_csl.services.cold_start.service import run_cold_start_layer
from apps.ranking_layer.services.ranking.service import run_ranking_layer
from apps.ml_layer.services.ml_reranking.service import run_ml_reranking_layer
from apps.ml_layer.services.ml_reranking.adapter import prepare_reranked_for_bundling
from apps.bundling_layer.services.bundling.service import run_bundling_layer
from apps.explainability_layer.services.explainability.service import run_explainability_layer


class Command(BaseCommand):
    help = "Run cold-start -> ranking -> ML reranking -> bundling -> explainability"

    def add_arguments(self, parser):
        parser.add_argument(
            "--cfm-parquet",
            type=str,
            default="nhanes_step1_out/NHANES_Master_engineered.parquet",
        )
        parser.add_argument(
            "--rules-jsonl",
            type=str,
            default="rules_safety_engine/output/recommendations.jsonl",
        )
        parser.add_argument(
            "--participant-key",
            type=str,
            default=None,
            help="Run the full pipeline for a single participant only.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Run the full pipeline for the first N participants only.",
        )
        parser.add_argument(
            "--save-intermediate",
            action="store_true",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)

        cfm_path = base_dir / options["cfm_parquet"]
        rules_path = base_dir / options["rules_jsonl"]

        self.stdout.write("Loading engineered participant data...")
        cfm_df = pd.read_parquet(cfm_path)

        self.stdout.write("Loading safe recommendation candidates...")
        rules_df = pd.read_json(rules_path, lines=True)

        participant_key = options.get("participant_key")
        limit = options.get("limit")

        if participant_key:
            self.stdout.write(f"Filtering to participant: {participant_key}")
            cfm_df = cfm_df.loc[
                cfm_df["ParticipantKey"].astype(str) == str(participant_key)
            ].copy()

            rules_key_col = "ParticipantKey" if "ParticipantKey" in rules_df.columns else "participant_key"
            rules_df = rules_df.loc[
                rules_df[rules_key_col].astype(str) == str(participant_key)
            ].copy()

        elif limit is not None:
            self.stdout.write(f"Filtering to first {limit} participants")
            selected_keys = (
                cfm_df["ParticipantKey"]
                .astype(str)
                .drop_duplicates()
                .head(int(limit))
                .tolist()
            )

            cfm_df = cfm_df.loc[
                cfm_df["ParticipantKey"].astype(str).isin(selected_keys)
            ].copy()

            rules_key_col = "ParticipantKey" if "ParticipantKey" in rules_df.columns else "participant_key"
            rules_df = rules_df.loc[
                rules_df[rules_key_col].astype(str).isin(selected_keys)
            ].copy()

        if cfm_df.empty:
            raise ValueError("No participant rows remained after filtering cfm_df.")
        if rules_df.empty:
            raise ValueError("No recommendation rows remained after filtering rules_df.")

        self.stdout.write("Adapting flat rules output to cold-start participant format...")
        participants_df = self._build_participants_candidates_df(rules_df)

        self.stdout.write("Running cold-start layer...")
        cs_out = run_cold_start_layer(cfm_df=cfm_df, participants_df=participants_df)
        safety_context_df = cs_out["safety_context"]
        gated_candidates_df = cs_out["gated_candidates"]

        self.stdout.write("Running ranking layer...")
        ranking_out = run_ranking_layer(
            safety_context_df=safety_context_df,
            gated_candidates_df=gated_candidates_df,
        )
        ranked_df = ranking_out["ranked_recommendations"]

        self.stdout.write("Running ML reranking layer...")
        ml_out = run_ml_reranking_layer(
            safety_context_df=safety_context_df,
            ranked_df=ranked_df,
            participant_features_df=cfm_df,
        )
        reranked_df = ml_out["reranked_recommendations"]

        bundling_input_df = prepare_reranked_for_bundling(reranked_df)

        self.stdout.write("Running bundling layer...")
        bundling_out = run_bundling_layer(
            safety_context_df=safety_context_df,
            ranked_df=bundling_input_df,
        )
        bundles_df = bundling_out["final_recommendations"]

        self.stdout.write("Running explainability layer...")
        explainability_out = run_explainability_layer(
            safety_context_df=safety_context_df,
            final_recommendations_df=bundles_df,
            reranked_df=reranked_df,
            participant_features_df=cfm_df,
        )
        explained_df = explainability_out["explained_recommendations"]

        if options["save_intermediate"]:
            out_dir = base_dir / "apps" / "ml_layer" / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            participants_df.to_json(
                out_dir / "participants_candidates_input.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            gated_candidates_df.to_json(
                out_dir / "gated_candidates_after_coldstart.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            ranked_df.to_json(
                out_dir / "ranked_before_ml.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            reranked_df.to_json(
                out_dir / "reranked_recommendations.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            bundles_df.to_json(
                out_dir / "final_recommendations_after_ml.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            explained_df.to_json(
                out_dir / "explained_recommendations.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )

            manifest = {
                "run_scope": {
                    "participant_key": options.get("participant_key"),
                    "limit": options.get("limit"),
                },
                "cold_start": cs_out.get("manifest", {}),
                "ranking": ranking_out.get("manifest", {}),
                "ml_reranking": ml_out.get("manifest", {}),
                "bundling": bundling_out.get("manifest", {}),
                "explainability": explainability_out.get("manifest", {}),
            }
            with open(out_dir / "full_pipeline_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        self.stdout.write(self.style.SUCCESS("Full hybrid pipeline complete."))
        if options.get("participant_key"):
            self.stdout.write(f"Run scope: single participant = {options['participant_key']}")
        elif options.get("limit") is not None:
            self.stdout.write(f"Run scope: first {options['limit']} participants")

        self.stdout.write(f"Participants input rows: {len(participants_df)}")
        self.stdout.write(f"Safety context rows: {len(safety_context_df)}")
        self.stdout.write(f"Gated candidates rows: {len(gated_candidates_df)}")
        self.stdout.write(f"Ranked rows: {len(ranked_df)}")
        self.stdout.write(f"Reranked rows: {len(reranked_df)}")
        self.stdout.write(f"Final bundles: {len(bundles_df)}")
        self.stdout.write(f"Explained recommendations: {len(explained_df)}")

    @staticmethod
    def _build_participants_candidates_df(rules_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert flat rules-engine recommendations into the structure expected by
        apps.recs_csl.services.cold_start.candidates.explode_candidates():

        columns:
        - ParticipantKey
        - Candidates: list[dict]

        Each candidate dict must contain:
        - action_code
        - action_type
        - severity
        - priority
        - tags
        - evidence
        - source_rule_ids
        - bundle_family
        """
        if rules_df.empty:
            return pd.DataFrame(columns=["ParticipantKey", "Candidates"])

        working = rules_df.copy()

        if "participant_key" in working.columns:
            working = working.rename(columns={"participant_key": "ParticipantKey"})

        required_candidate_fields = [
            "action_code",
            "action_type",
            "severity",
            "priority",
            "tags",
            "evidence",
            "source_rule_ids",
        ]

        missing = [c for c in ["ParticipantKey"] + required_candidate_fields if c not in working.columns]
        if missing:
            raise ValueError(f"Rules input missing columns required for cold-start adapter: {missing}")

        candidate_records_df = working[["ParticipantKey"] + required_candidate_fields].copy()

        candidate_records_df["bundle_family"] = candidate_records_df.apply(
            lambda row: Command._infer_bundle_family(
                action_code=row.get("action_code"),
                action_type=row.get("action_type"),
                tags=row.get("tags"),
            ),
            axis=1,
        )

        participants_df = (
            candidate_records_df
            .groupby("ParticipantKey", as_index=False)
            .agg({
                "action_code": list,
                "action_type": list,
                "severity": list,
                "priority": list,
                "tags": list,
                "evidence": list,
                "source_rule_ids": list,
                "bundle_family": list,
            })
        )

        participants_df["Candidates"] = participants_df.apply(
            lambda row: [
                {
                    "action_code": action_code,
                    "action_type": action_type,
                    "severity": severity,
                    "priority": priority,
                    "tags": tags,
                    "evidence": evidence,
                    "source_rule_ids": source_rule_ids,
                    "bundle_family": bundle_family,
                }
                for action_code, action_type, severity, priority, tags, evidence, source_rule_ids, bundle_family
                in zip(
                    row["action_code"],
                    row["action_type"],
                    row["severity"],
                    row["priority"],
                    row["tags"],
                    row["evidence"],
                    row["source_rule_ids"],
                    row["bundle_family"],
                )
            ],
            axis=1,
        )

        participants_df = participants_df[["ParticipantKey", "Candidates"]].copy()

        return participants_df

    @staticmethod
    def _infer_bundle_family(action_code, action_type, tags) -> str:
        """
        Conservative bundle-family inference for flat rules-engine output.

        This is only an adapter for full-pipeline execution. It should be replaced
        later by a shared canonical family-mapping utility if you decide to expose
        one across layers.
        """
        action_code = str(action_code or "").upper()
        action_type = str(action_type or "").lower()
        tags = tags if isinstance(tags, list) else []
        tags_upper = {str(t).upper() for t in tags}

        if "ESCALATE" in action_code or action_type == "escalate" or "ACUTE" in tags_upper or "SAFETY" in tags_upper:
            return "Monitoring"

        if (
            "DIET" in action_code
            or "DASH" in action_code
            or "MEDITERRANEAN" in action_code
            or "SODIUM" in action_code
            or "FIBER" in action_code
            or "DIET" in tags_upper
            or "SODIUM" in tags_upper
            or "PATTERN" in tags_upper
            or "NHANES_DAY1" in tags_upper
        ):
            return "DietPattern"

        if (
            "MVPA" in action_code
            or "ACTIVITY" in action_code
            or "WALK" in action_code
            or "EXERCISE" in action_code
            or "PA-" in action_code
            or "PHYSICAL_ACTIVITY" in tags_upper
            or "MVPA" in tags_upper
        ):
            return "Movement"

        if (
            "WEIGHT" in action_code
            or "OBESITY" in action_code
            or "BMI" in action_code
            or "WAIST" in action_code
        ):
            return "Weight"

        if (
            "SMOK" in action_code
            or "TOBACCO" in action_code
            or "SMOKING" in tags_upper
        ):
            return "Smoking"

        if (
            "ALCOHOL" in action_code
            or "DRINK" in action_code
            or "ALCOHOL" in tags_upper
        ):
            return "Alcohol"

        if (
            "LIPID" in action_code
            or "CHOLESTEROL" in action_code
            or "HDL" in action_code
            or "TC_" in action_code
            or "LIPIDS" in tags_upper
        ):
            return "Lipids"

        if (
            "MONITOR" in action_code
            or "REVIEW" in action_code
            or "FOLLOWUP" in action_code
            or "FOLLOW_UP" in action_code
            or "BP_" in action_code
            or "A1C" in action_code
            or "FOLLOWUP" in tags_upper
            or "A1C" in tags_upper
        ):
            return "Monitoring"

        if "CARDIOMETABOLIC" in action_code or "INTEGRATED" in action_code:
            return "IntegratedCardiometabolic"

        return "Monitoring"
