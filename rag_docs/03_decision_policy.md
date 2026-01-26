# Decision policy and outputs

## What the model predicts
The model outputs:
- `score = P(default = bad)`

So higher score means higher risk.

## Decision rule
This demo applies a single rule:
- **decline if score >= threshold**
- otherwise approve

Threshold is selected on validation data, using a cost function.

## Cost-based thresholding
Threshold is chosen to minimize:
- `cost = fp_cost * FP + fn_cost * FN`

Where:
- FP = predicted bad but actually good (false decline),
- FN = predicted good but actually bad (false approve).

Defaults in this project often reflect that FN is more expensive than FP.

## “Reasons” (explainability)
Reason codes depend on the classifier type:

- LogisticRegression:
  - we can compute local contributions as `x_i * coef_i`
  - these are on the transformed feature space (after preprocessing / one-hot)
  - stable, fast, and good for demos.

- XGBoost:
  - we can request per-feature contributions using `pred_contribs=True`
  - also on transformed feature space
  - works without the `shap` dependency.

- RandomForest / HistGradientBoosting:
  - true local explanations are not available in a lightweight way without extra tooling
  - for demo we typically show:
    - global feature importance artifacts,
    - and keep per-request reasons empty,
    - or return “top global features” as a weak substitute.

In other words:
- if `reasons=[]`, it does not mean the model is broken.
- it usually means “this model type doesn’t support cheap local attributions”.

## Monitoring and drift
We log requests (anonymized/limited to features needed for scoring).
Drift compares:
- reference distribution (training baseline, artifact `drift_reference_train.csv`)
vs
- current distribution (last N requests from `data/requests.jsonl`)

We compute:
- PSI for numeric and categorical (bucketed / top categories),
- KS for numeric (when enough data exists).
These are heuristics, not a replacement for full monitoring, but good for demos.
