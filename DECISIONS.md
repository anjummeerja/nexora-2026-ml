# NEXORA 2026 — Decision Log

## 1. Objective

The objective is to rank gateways by operational fault risk and select at most 15 gateways for field visits each week.

The final system produces a deterministic top-15 ranking for each scoring week.

The optimization target is not ordinary classification accuracy. The important operational objective is to identify gateways likely to require a visit within the fixed 15-visit capacity.

---

## 2. Data Cutoff

The prediction system uses only information available before the beginning of each prediction week.

For a Monday prediction week, the main telemetry features are calculated from the preceding 28 days and do not include telemetry from the prediction week itself.

Future field-visit outcomes are never used as prediction-time features.

---
## 3. Five Key Decisions

### Decision 1 — Define the prediction signal using recent history

I chose to use a preceding 28-day telemetry window for each prediction week.

Alternative considered:
I could have used only the most recent 7 days or included telemetry from the prediction week.

Why I did not:
A 7-day window can be noisy and less representative of normal gateway behaviour. Using telemetry from the prediction week would violate the decision-time cutoff. The 28-day window provides recent history while avoiding future information.

### Decision 2 — Use confirmed field-fault visits as a historical proxy target

I used historical field visits with outcome `Fehler behoben` as a development proxy for a gateway requiring attention.

Alternative considered:
I could have used all field visits as positive labels, or used engineer-review categories as the target.

Why I did not:
A field visit does not necessarily mean that the gateway was faulty, because some visits ended with no fault found or no access. The engineer review was also dated 2026-02-15 and therefore cannot be used for the earlier February prediction weeks without violating the cutoff. The proxy is therefore used only for development validation, not claimed to be the hidden evaluation truth.

### Decision 3 — Use Logistic Regression for the final ranking model

I selected Logistic Regression for the final ranking model.

Alternative considered:
I tested Random Forest and a manually weighted risk score.

Why I did not:
Random Forest produced stronger generic validation metrics in one split but captured fewer confirmed-fault proxy cases in the January top-15 test. The manually weighted risk score also captured fewer cases. Since the operational decision is the top 15 gateways, I prioritised top-15 ranking behaviour and temporal testing rather than generic classification metrics.

### Decision 4 — Treat missing telemetry as a risk signal, not as proof of failure

I retained telemetry coverage as a model feature.

Alternative considered:
I could have removed gateway-weeks with low telemetry coverage or treated low coverage as an automatic fault condition.

Why I did not:
Low coverage may indicate communication problems, but it does not prove that the physical gateway is faulty. Historical analysis showed that gateway-weeks with below 50% telemetry coverage had a higher confirmed-fault proxy rate than gateway-weeks with at least 90% coverage. Therefore, coverage is useful as a predictive signal but should not be a hard rule.

### Decision 5 — Part 2 area: Machine Learning

I selected **Machine Learning** as my Part 2 area.

Alternative considered:
I could have focused Part 2 on Data Engineering, Software Development, DevOps, Data Science, or MLOps.

Why I did not:
The main work I developed beyond the supplied baseline was a leakage-free supervised ranking model, temporal validation, feature analysis, and comparison of multiple ML approaches. Machine Learning therefore matches the work I can explain and defend technically.

For Part 2, I focused on improving the supplied baseline using historical decision-time features and Logistic Regression. The model was tested using later historical weeks than the training period, including 14 gateways that were not present in the training period. This provides a leakage-aware robustness check, although it is not a substitute for the official unseen-month evaluation.

The historical development comparison showed:

| Method | Confirmed-fault proxy cases captured |
|---|---:|
| 3-sigma baseline | 3 / 12 |
| Logistic Regression | 5 / 12 |
| Random Forest | 2 / 12 |
| Relationship-based risk score | 3 / 12 |
| Clean Logistic Regression without gateway/visit-history features | 4 / 12 |

These are development proxy results, not the official hidden evaluation.

The model was also tested using later historical weeks and unseen gateways. The final model retains the strongest historical feature set, while documenting the risk that gateway identity and historical visit features may reflect previous operational decisions rather than only gateway condition.

A future unseen-month evaluation could change the model choice if another approach consistently performs better on top-15 ranking or operational cost.

## 4. Baseline

The supplied 3-sigma baseline was run successfully.

It uses recent telemetry behaviour for:

- offline duration
- disconnection count
- reboot count

The baseline was retained as the reference point for model development.

---

## 5. ML Approach

A Logistic Regression model was selected for the final ranking system.

The model uses historical decision-time features and produces a probability-like risk score.

The gateways are then sorted by this score and the top 15 are selected for each scoring week.

Ties are resolved deterministically using gateway ID.

---

## 6. Historical ML Validation

Several approaches were tested using temporal validation.

The main historical January 2026 test results using historical confirmed-fault visit labels as a proxy were:

| Method | Confirmed-fault proxy cases captured |
|---|---:|
| 3-sigma baseline | 3 / 12 |
| Logistic Regression | 5 / 12 |
| Random Forest | 2 / 12 |
| Relationship-based risk score | 3 / 12 |
| Clean Logistic Regression without gateway/visit-history features | 4 / 12 |

The aligned Logistic Regression gave the strongest top-15 result among the tested approaches.

These results are development evidence only. They are not the official hidden evaluation.

---

## 7. Feature Relationship Analysis

Historical analysis showed several useful associations with confirmed-fault visit labels.

The strongest observed relationships included:

- lower meter read success
- lower telemetry coverage
- higher offline duration
- more disconnections
- longer reboot duration
- higher system/load-related signals

The strongest standardized difference observed was for meter read rate.

These relationships are treated as predictive signals rather than causal explanations.

---

## 8. Missing Telemetry Decision

Missing telemetry was explicitly investigated.

Gateway-weeks with telemetry coverage below 50% had a 4.96% confirmed-fault proxy rate, compared with 0.43% for gateway-weeks with coverage of at least 90%.

Therefore, low telemetry coverage is treated as a risk signal.

It is not treated as proof that a gateway is faulty, because missing telemetry can have multiple operational causes.

---

## 9. Why Random Forest Was Not Selected

A Random Forest model was tested.

Although it produced a strong validation ROC-AUC, its top-15 ranking performance on the January historical test was worse than Logistic Regression.

This reinforced that ranking quality at the 15-visit limit is more relevant to this challenge than generic classification metrics.

---

## 10. Gateway Identity and Historical Visit Features

An experiment removing gateway identity and historical visit-count features produced weaker January top-15 performance.

The final aligned Logistic Regression therefore retains the feature set used in the strongest historical experiment.

However, gateway identity and historical visit information are treated cautiously because historical field visits are observational and may reflect previous operational decisions rather than only gateway condition.

The challenge's hidden evaluation remains the final authority.

---

## 11. Final Prediction

The final model was trained on the historical leakage-free dataset and applied to the official decision-time feature dataset.

The final output contains:

- 8 prediction weeks
- 15 gateways per week
- 120 rows total
- deterministic ranks from 1 to 15
- numeric risk scores
- an operational explanation for each selected gateway

The supplied submission validator reports the final `predictions.csv` as valid.

---

## 12. Limitations

The historical confirmed-fault labels come from field-visit records and are not the hidden evaluation truth.

Unvisited gateways cannot automatically be assumed healthy.

Missing telemetry does not necessarily indicate a physical gateway fault.

The model is trained offline and is not retrained during prediction.

The final ranking is optimized for the fixed 15-visit constraint and should not be interpreted as a calibrated probability of physical failure.

---

## 13. What Would Change Our Decision?

The model choice should be reconsidered if live/unseen-month evaluation shows that another approach consistently produces better top-15 recall or lower operational cost.

A substantial change in telemetry quality, gateway population, or failure behaviour could also justify retraining or revisiting the feature set.