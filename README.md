# NEXORA 2026 – ML Innovation Challenge

## Project Overview

This project implements a machine-learning-based gateway ranking system for the NEXORA 2026 LPDG Innovation Challenge.

The objective is to rank the 15 gateways that are most likely to require a field visit each week, using only information available before the prediction week begins.

## Approach

The solution uses:

* Decision-time telemetry features from the preceding 28 days
* Connectivity and disconnection signals
* Reboot and offline-duration signals
* Meter-read success information
* Gateway metadata
* Historical field-visit information available before the prediction cutoff
* Logistic Regression with balanced class weights
* Deterministic ranking of gateways by predicted fault risk

The feature engineering is designed to avoid using information from the prediction week itself.

## Baseline Improvement

The official 3-Sigma baseline was evaluated first.

A leakage-free Logistic Regression ranker was then developed and evaluated using forward historical validation.

In historical proxy evaluation:

* 3-Sigma baseline: 3/12 fault cases captured
* Logistic Regression: 5/12 fault cases captured
* Baseline proxy cost: EUR 28,200
* ML proxy cost: EUR 27,000

These are historical proxy results and are not the official hidden challenge score.

## Output

The final solution generates:

`predictions.csv`

The file contains 120 rows covering 8 prediction weeks, with 15 ranked gateways per week.

Columns:

* `week_start`
* `rank`
* `gateway_id`
* `score`
* `reason`

## Running the Solution

The challenge data must be placed in the required local `data/` directory.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python run.py
```

The pipeline builds the required decision-time features, trains the model, generates predictions, and validates the output schema.

## Important

The challenge dataset is private and is intentionally excluded from this repository.

The `data/` directory, challenge ZIP files, and challenge materials are ignored by Git.

## Limitations

The model predicts risk based on historical and telemetry signals. It does not guarantee that a gateway is faulty.

It cannot:

* observe hidden ground truth
* physically inspect a gateway
* guarantee cost savings
* recover missing telemetry
* use future information that was unavailable at decision time
* replace operational judgement

See `DECISIONS.md`, `AI-USAGE.md`, and `WHAT_IT_CANNOT_DO.md` for additional details.
