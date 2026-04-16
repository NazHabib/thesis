def build_shap_text_summaries(
    top_positive_factors_clinical_only: list[dict] | None,
    top_negative_factors_clinical_only: list[dict] | None,
) -> dict:
    pos = top_positive_factors_clinical_only or []
    neg = top_negative_factors_clinical_only or []

    pos_labels = [str(item.get("feature_label", item.get("feature", ""))).strip() for item in pos if item]
    neg_labels = [str(item.get("feature_label", item.get("feature", ""))).strip() for item in neg if item]

    pos_labels = [x for x in pos_labels if x]
    neg_labels = [x for x in neg_labels if x]

    if not pos_labels and not neg_labels:
        return {
            "clinician_ml_xai_summary": (
                "No strong clinical or lifestyle SHAP contributors were identified for this action."
            ),
            "patient_ml_xai_summary": (
                "The model did not identify especially strong health or lifestyle factors affecting this recommendation."
            ),
        }

    clinician_parts = []
    patient_parts = []

    if pos_labels:
        clinician_parts.append(
            f"Features increasing relative ML emphasis included: {_join_labels(pos_labels[:3])}."
        )
        patient_parts.append(
            f"The model gave relatively more emphasis because of: {_join_labels(pos_labels[:2])}."
        )

    if neg_labels:
        clinician_parts.append(
            f"Features decreasing relative ML emphasis included: {_join_labels(neg_labels[:3])}."
        )
        patient_parts.append(
            f"Factors reducing relative emphasis included: {_join_labels(neg_labels[:2])}."
        )

    return {
        "clinician_ml_xai_summary": " ".join(clinician_parts),
        "patient_ml_xai_summary": " ".join(patient_parts),
    }


def _join_labels(labels: list[str]) -> str:
    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        return "none"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"
