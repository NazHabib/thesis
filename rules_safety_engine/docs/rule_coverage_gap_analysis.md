# Rule Coverage Gap Analysis

## Overview

The current rulebase demonstrates broad domain coverage across the main safety and lifestyle-support areas relevant to the dissertation scope. Existing rule families cover acute safety escalation, data-quality messaging, hypertension management, diabetes management, obesity, dietary risk, physical activity, smoking, alcohol, lipid-related lifestyle support, and cardiometabolic multimorbidity.

From a structural perspective, the present rulebase should therefore be interpreted as **broadly domain-complete at the cross-sectional level**, with remaining gaps concentrated less in missing clinical families and more in **temporal, longitudinal, and progression-aware reasoning**.

## Areas with Strong Coverage

The following areas appear well represented:

- Safety and out-of-scope escalation
- Acute hypertension and hyperglycaemia escalation
- Data-quality and missing-data communication
- Hypertension lifestyle and review pathways
- Diabetes prevention and diabetes follow-up logic
- Obesity and weight-management support
- Dietary pattern and sodium-related interventions
- Physical activity and sedentary behaviour support
- Smoking and alcohol-related lifestyle messaging
- Lipid-related lifestyle recommendations
- Basic cardiometabolic multimorbidity integration

## Areas with Moderate but Adequate Coverage

The following areas appear represented, but with potential for future refinement:

- Multimorbidity coordination beyond the strongest current combinations
- Lipid support beyond total cholesterol and low HDL
- Smoking/alcohol coverage as part of broader integrated bundles
- Data-quality logic linked to downstream confidence and communication strategies

## Main Coverage Gaps

### 1. Temporal and Longitudinal Logic

The most important remaining gap is the absence of temporal reasoning. Current rules operate primarily on cross-sectional participant state. Future rule expansion should support:

- repeated elevated blood pressure across time
- worsening HbA1c trajectories
- repeated high sodium exposure
- persistence of inactivity or obesity without improvement
- non-improvement despite prior follow-up advice

### 2. Follow-up Timing Granularity

Although some follow-up rules already encode timing concepts (e.g. 1 month or 3–6 months), timing is not yet governed as a broader rule dimension. Future rule coverage should better distinguish:

- urgent review
- short-term review
- medium-term reassessment
- annual preventive follow-up

### 3. Progression-Sensitive Multimorbidity

The current multimorbidity rules are clinically useful, but future expansion could increase precision for cases such as:

- hypertension + obesity without diabetes
- diabetes + dyslipidaemia + inactivity
- obesity + sedentary behaviour + high sodium intake
- layered lifestyle-risk accumulation across multiple domains

### 4. Longitudinal Data-Quality Safety

Current data-quality rules are useful, but future work could explicitly model:

- repeated missingness
- stale measurements
- disagreement across repeated measures
- insufficient longitudinal evidence for stable recommendations

## Interpretation

The present rulebase should not be viewed as underdeveloped. On the contrary, it is already strong in the main domain families required for a safety-first hybrid clinical recommender. The principal next step is not broad domain expansion, but **temporal deepening**, **follow-up refinement**, and **richer multimorbidity coordination**.

## Conclusion

The current rulebase is suitable for a dissertation-stage system and provides strong deterministic coverage across the intended clinical-lifestyle scope. Future rule expansion should prioritize temporal and longitudinal sophistication over simple increases in rule count.
