# AI Usage

## How AI was used

AI assistance was used during development as a supporting tool for:

- understanding the challenge requirements and data dictionary
- brainstorming feature engineering ideas
- reviewing Python code and debugging errors
- explaining ML concepts and validation approaches
- suggesting experiments for comparing ranking approaches
- helping structure documentation and submission materials

All code was run locally and its outputs were checked against the actual challenge data.

AI was not used as a replacement for evaluating the model. Model experiments, validation results, feature relationships, and final predictions were generated and checked locally.

---

## AI-generated suggestions that were tested

Several modelling approaches were explored, including:

- the supplied 3-sigma baseline
- Logistic Regression
- Random Forest
- a manually weighted risk score
- a cleaner Logistic Regression configuration without some potentially memorising features

The approaches were compared using temporal validation and top-15 ranking performance rather than relying only on classification accuracy.

---

## One AI mistake that was caught

An early ML experiment used telemetry features from the same week as the target label.

This could introduce temporal leakage because the prediction system would not actually know the complete telemetry behaviour of the week it was supposed to predict.

The issue was identified during review of the challenge's decision-time requirement.

The feature-generation process was then changed so that prediction-time telemetry features are calculated only from the preceding 28 days, ending before the prediction week begins.

The historical training dataset was rebuilt using the same decision-time logic.

This was an important correction because a model with leaked information could appear stronger during testing while performing worse in the real evaluation.

---

## Human verification

The following checks were performed locally:

1. The supplied baseline was executed successfully.
2. The baseline submission was validated.
3. Multiple ML approaches were tested.
4. Temporal validation was used to avoid training on future weeks.
5. Feature relationships were inspected.
6. Missing telemetry behaviour was explicitly analysed.
7. The final model was trained locally.
8. The final `predictions.csv` was generated locally.
9. The official submission validator was run successfully.

Final validator result:

`predictions.csv: OK`

The final submission contains 15 ranked gateways for each of the 8 required weeks.