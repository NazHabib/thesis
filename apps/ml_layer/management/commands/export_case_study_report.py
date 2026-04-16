import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Export a complete case study report for one participant."

    def add_arguments(self, parser):
        parser.add_argument(
            "--participant-key",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--ranked-before-ml",
            type=str,
            default="apps/ml_layer/output/ranked_before_ml.jsonl",
        )
        parser.add_argument(
            "--reranked",
            type=str,
            default="apps/ml_layer/output/reranked_recommendations.jsonl",
        )
        parser.add_argument(
            "--bundles",
            type=str,
            default="apps/ml_layer/output/final_recommendations_after_ml.jsonl",
        )
        parser.add_argument(
            "--explained",
            type=str,
            default="apps/ml_layer/output/explained_recommendations.jsonl",
        )
        parser.add_argument(
            "--shap-dir",
            type=str,
            default="apps/explainability_layer/output/shap",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="apps/ml_layer/output/case_studies",
        )

    def handle(self, *args, **options):
        base_dir = Path(settings.BASE_DIR)
        participant_key = options["participant_key"]

        ranked_path = base_dir / options["ranked_before_ml"]
        reranked_path = base_dir / options["reranked"]
        bundles_path = base_dir / options["bundles"]
        explained_path = base_dir / options["explained"]
        shap_dir = base_dir / options["shap_dir"]
        out_dir = base_dir / options["output_dir"] / self._safe_name(participant_key)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"Building case study report for: {participant_key}")

        ranked_df = self._read_jsonl_if_exists(ranked_path)
        reranked_df = self._read_jsonl_if_exists(reranked_path)
        bundles_df = self._read_jsonl_if_exists(bundles_path)
        explained_df = self._read_jsonl_if_exists(explained_path)

        ranked_case = self._filter_participant(ranked_df, participant_key)
        reranked_case = self._filter_participant(reranked_df, participant_key)
        bundles_case = self._filter_participant(bundles_df, participant_key)
        explained_case = self._filter_participant(explained_df, participant_key)

        ranked_case = self._sort_ranked_before_ml(ranked_case)
        reranked_case = self._sort_reranked(reranked_case)
        bundles_case = self._sort_bundles(bundles_case)
        explained_case = self._sort_explained(explained_case)

        shap_report = self._load_shap_report(shap_dir=shap_dir, participant_key=participant_key)

        ranked_case.to_csv(out_dir / "ranked_before_ml.csv", index=False)
        reranked_case.to_csv(out_dir / "reranked_after_ml.csv", index=False)
        bundles_case.to_csv(out_dir / "final_bundles.csv", index=False)
        explained_case.to_csv(out_dir / "explained_recommendations.csv", index=False)

        if shap_report:
            with open(out_dir / "shap_report.json", "w", encoding="utf-8") as f:
                json.dump(shap_report, f, indent=2, ensure_ascii=False)

            shap_action_rows = pd.DataFrame(shap_report.get("per_action_explanations", []))
            shap_global_rows = pd.DataFrame(shap_report.get("global_feature_importance", []))
            shap_global_clinical_rows = pd.DataFrame(
                shap_report.get("global_feature_importance_clinical_only", [])
            )

            shap_action_rows.to_csv(out_dir / "shap_per_action.csv", index=False)
            shap_global_rows.to_csv(out_dir / "shap_global_importance.csv", index=False)
            shap_global_clinical_rows.to_csv(
                out_dir / "shap_global_importance_clinical_only.csv",
                index=False,
            )

        summary = self._build_case_summary(
            participant_key=participant_key,
            ranked_case=ranked_case,
            reranked_case=reranked_case,
            bundles_case=bundles_case,
            explained_case=explained_case,
            shap_report=shap_report,
        )

        with open(out_dir / "case_study_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        demo_summary = self._build_demo_summary(
            participant_key=participant_key,
            ranked_case=ranked_case,
            reranked_case=reranked_case,
            bundles_case=bundles_case,
            explained_case=explained_case,
            shap_report=shap_report,
        )

        with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
            json.dump(demo_summary, f, indent=2, ensure_ascii=False)

        before_after_df = self._build_before_vs_after_top_actions_df(
            ranked_case=ranked_case,
            reranked_case=reranked_case,
        )
        before_after_df.to_csv(out_dir / "before_vs_after_top_actions.csv", index=False)

        bundle_summary_df = self._build_bundle_summary_df(bundles_case)
        bundle_summary_df.to_csv(out_dir / "bundle_summary.csv", index=False)

        self.stdout.write(self.style.SUCCESS("Case study report exported."))
        self.stdout.write(f"Saved to: {out_dir}")

    @staticmethod
    def _safe_name(value: str) -> str:
        return str(value).replace("/", "_")

    @staticmethod
    def _read_jsonl_if_exists(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_json(path, lines=True)

    @staticmethod
    def _filter_participant(df: pd.DataFrame, participant_key: str) -> pd.DataFrame:
        if df.empty or "ParticipantKey" not in df.columns:
            return pd.DataFrame()
        return df.loc[df["ParticipantKey"].astype(str) == str(participant_key)].copy()

    @staticmethod
    def _sort_ranked_before_ml(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        sort_cols = [c for c in ["total_score", "confidence_score", "priority", "action_code"] if c in df.columns]
        asc = [False, False, False, True][: len(sort_cols)]
        return df.sort_values(by=sort_cols, ascending=asc, kind="stable").reset_index(drop=True)

    @staticmethod
    def _sort_reranked(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        sort_cols = [c for c in ["final_score", "total_score", "confidence_score", "priority", "action_code"] if c in df.columns]
        asc = [False, False, False, False, True][: len(sort_cols)]
        return df.sort_values(by=sort_cols, ascending=asc, kind="stable").reset_index(drop=True)

    @staticmethod
    def _sort_bundles(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        sort_cols = [c for c in ["bundle_score", "confidence_score", "priority", "bundle_title"] if c in df.columns]
        asc = [False, False, False, True][: len(sort_cols)]
        return df.sort_values(by=sort_cols, ascending=asc, kind="stable").reset_index(drop=True)

    @staticmethod
    def _sort_explained(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        sort_cols = [c for c in ["bundle_score", "confidence_score", "priority", "bundle_title"] if c in df.columns]
        asc = [False, False, False, True][: len(sort_cols)]
        return df.sort_values(by=sort_cols, ascending=asc, kind="stable").reset_index(drop=True)

    @staticmethod
    def _load_shap_report(shap_dir: Path, participant_key: str) -> dict:
        if not shap_dir.exists():
            return {}

        path = shap_dir / f"{participant_key.replace('/', '_')}_shap_report.json"
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_case_summary(
        participant_key: str,
        ranked_case: pd.DataFrame,
        reranked_case: pd.DataFrame,
        bundles_case: pd.DataFrame,
        explained_case: pd.DataFrame,
        shap_report: dict,
    ) -> dict:
        model_version = None
        if not reranked_case.empty and "ml_model_version" in reranked_case.columns:
            versions = reranked_case["ml_model_version"].dropna().astype(str).unique().tolist()
            model_version = versions[0] if versions else None

        summary = {
            "participant_key": participant_key,
            "metadata": {
                "model_version": model_version,
                "fusion_strategy": "participant_minmax_total_score_plus_ml_contrib",
                "shap_model_version": shap_report.get("model_version") if shap_report else None,
            },
            "counts": {
                "ranked_before_ml": int(len(ranked_case)),
                "reranked_after_ml": int(len(reranked_case)),
                "final_bundles": int(len(bundles_case)),
                "explained_recommendations": int(len(explained_case)),
            },
            "top_ranked_before_ml": ranked_case.head(5).to_dict(orient="records"),
            "top_reranked_after_ml": reranked_case.head(5).to_dict(orient="records"),
            "top_final_bundles": bundles_case.head(5).to_dict(orient="records"),
            "top_explained_recommendations": explained_case.head(5).to_dict(orient="records"),
        }

        if not ranked_case.empty and not reranked_case.empty:
            compare_cols = [c for c in [
                "action_code",
                "total_score",
                "final_score",
                "ml_score",
                "lambda_u",
                "ml_contrib",
                "rerank_status",
                "fallback_reason",
            ] if c in reranked_case.columns]

            summary["before_after_comparison"] = {
                "before": ranked_case[[c for c in ["action_code", "total_score", "confidence_score"] if c in ranked_case.columns]]
                .head(10)
                .to_dict(orient="records"),
                "after": reranked_case[compare_cols].head(10).to_dict(orient="records"),
            }

        if shap_report:
            summary["shap_summary"] = {
                "model_version": shap_report.get("model_version"),
                "top_global_features": shap_report.get("global_feature_importance", [])[:10],
                "top_global_features_clinical_only": shap_report.get("global_feature_importance_clinical_only", [])[:10],
                "top_action_explanations": shap_report.get("per_action_explanations", [])[:5],
            }

        return summary
        
    @staticmethod
    def _build_demo_summary(
        participant_key: str,
        ranked_case: pd.DataFrame,
        reranked_case: pd.DataFrame,
        bundles_case: pd.DataFrame,
        explained_case: pd.DataFrame,
        shap_report: dict,
    ) -> dict:
        top_rule_action = None
        if not ranked_case.empty and "action_code" in ranked_case.columns:
            top_rule_action = ranked_case.iloc[0]["action_code"]

        top_hybrid_action = None
        if not reranked_case.empty and "action_code" in reranked_case.columns:
            top_hybrid_action = reranked_case.iloc[0]["action_code"]

        top_bundle = None
        if not bundles_case.empty and "bundle_title" in bundles_case.columns:
            top_bundle = bundles_case.iloc[0]["bundle_title"]

        rank_changed = bool(top_rule_action != top_hybrid_action) if top_rule_action and top_hybrid_action else False

        top3_rule = []
        if not ranked_case.empty and "action_code" in ranked_case.columns:
            top3_rule = ranked_case["action_code"].head(3).astype(str).tolist()

        top3_hybrid = []
        if not reranked_case.empty and "action_code" in reranked_case.columns:
            top3_hybrid = reranked_case["action_code"].head(3).astype(str).tolist()

        top3_changed = set(top3_rule) != set(top3_hybrid) if top3_rule and top3_hybrid else False

        model_version = None
        if not reranked_case.empty and "ml_model_version" in reranked_case.columns:
            model_version = reranked_case["ml_model_version"].dropna().astype(str).head(1).tolist()
            model_version = model_version[0] if model_version else None

        return {
            "participant_key": participant_key,
            "top_rule_action": top_rule_action,
            "top_hybrid_action": top_hybrid_action,
            "top_bundle": top_bundle,
            "rank_changed": rank_changed,
            "top3_changed": top3_changed,
            "n_candidates_before_ml": int(len(ranked_case)),
            "n_candidates_after_ml": int(len(reranked_case)),
            "n_bundles": int(len(bundles_case)),
            "n_explained_recommendations": int(len(explained_case)),
            "model_version": model_version,
            "fusion_strategy": "participant_minmax_total_score_plus_ml_contrib",
            "shap_model_version": shap_report.get("model_version") if shap_report else None,
        }

    @staticmethod
    def _build_before_vs_after_top_actions_df(
        ranked_case: pd.DataFrame,
        reranked_case: pd.DataFrame,
    ) -> pd.DataFrame:
        before = []
        if not ranked_case.empty and "action_code" in ranked_case.columns:
            before = ranked_case["action_code"].head(10).astype(str).tolist()

        after = []
        if not reranked_case.empty and "action_code" in reranked_case.columns:
            after = reranked_case["action_code"].head(10).astype(str).tolist()

        n = max(len(before), len(after))
        rows = []
        for i in range(n):
            before_action = before[i] if i < len(before) else None
            after_action = after[i] if i < len(after) else None
            rows.append({
                "rank": i + 1,
                "before_action": before_action,
                "after_action": after_action,
                "changed": bool(before_action != after_action),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _build_bundle_summary_df(bundles_case: pd.DataFrame) -> pd.DataFrame:
        if bundles_case.empty:
            return pd.DataFrame(columns=[
                "bundle_rank",
                "bundle_title",
                "bundle_family",
                "bundle_score",
                "confidence_label",
                "source_rule_ids",
                "n_actions_in_bundle",
            ])

        out = bundles_case.copy().reset_index(drop=True)
        out["bundle_rank"] = out.index + 1
        out["n_actions_in_bundle"] = out["action_codes"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        cols = [
            c for c in [
                "bundle_rank",
                "bundle_title",
                "bundle_family",
                "bundle_score",
                "confidence_label",
                "source_rule_ids",
                "n_actions_in_bundle",
            ]
            if c in out.columns
        ]

        return out[cols].copy()
