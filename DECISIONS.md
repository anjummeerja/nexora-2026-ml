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

## 3. Baseline

The supplied 3-sigma baseline was run successfully.

It uses recent telemetry behaviour for:

- offline duration
- disconnection count
- reboot count

The baseline was retained as the reference point for model development.

---

## 4. ML Approach

A Logistic Regression model was selected for the final ranking system.

The model uses historical decision-time features and produces a probability-like risk score.

The gateways are then sorted by this score and the top 15 are selected for each scoring week.

Ties are resolved deterministically using gateway ID.

---

## 5. Historical ML Validation

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

## 6. Feature Relationship Analysis

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

## 7. Missing Telemetry Decision

Missing telemetry was explicitly investigated.

Gateway-weeks with telemetry coverage below 50% had a 4.96% confirmed-fault proxy rate, compared with 0.43% for gateway-weeks with coverage of at least 90%.

Therefore, low telemetry coverage is treated as a risk signal.

It is not treated as proof that a gateway is faulty, because missing telemetry can have multiple operational causes.

---

## 8. Why Random Forest Was Not Selected

A Random Forest model was tested.

Although it produced a strong validation ROC-AUC, its top-15 ranking performance on the January historical test was worse than Logistic Regression.

This reinforced that ranking quality at the 15-visit limit is more relevant to this challenge than generic classification metrics.

---

## 9. Gateway Identity and Historical Visit Features

An experiment removing gateway identity and historical visit-count features produced weaker January top-15 performance.

The final aligned Logistic Regression therefore retains the feature set used in the strongest historical experiment.

However, gateway identity and historical visit information are treated cautiously because historical field visits are observational and may reflect previous operational decisions rather than only gateway condition.

The challenge's hidden evaluation remains the final authority.

---

## 10. Final Prediction

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

## 11. Limitations

The historical confirmed-fault labels come from field-visit records and are not the hidden evaluation truth.

Unvisited gateways cannot automatically be assumed healthy.

Missing telemetry does not necessarily indicate a physical gateway fault.

The model is trained offline and is not retrained during prediction.

The final ranking is optimized for the fixed 15-visit constraint and should not be interpreted as a calibrated probability of physical failure.

---

## 12. What Would Change Our Decision?

The model choice should be reconsidered if live/unseen-month evaluation shows that another approach consistently produces better top-15 recall or lower operational cost.

A substantial change in telemetry quality, gateway population, or failure behaviour could also justify retraining or revisiting the feature set.