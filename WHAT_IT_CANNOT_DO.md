# What It Cannot Do

## 1. It cannot know the hidden evaluation truth

The model is trained using historical operational data and field-visit information available during development.

The final evaluation uses a separate hidden ground truth, so the system cannot know the official answer in advance.

---

## 2. It cannot guarantee that a selected gateway is faulty

A high model score means that the gateway has higher predicted operational risk relative to the other gateways.

It does not prove that the gateway has a physical fault.

---

## 3. It cannot assume missing telemetry means failure

Missing or incomplete telemetry can have several causes.

The model therefore treats telemetry coverage as a risk signal rather than automatically classifying missing data as a fault.

---

## 4. It cannot use future information

For each prediction week, the feature calculation is based on information available before that week begins.

The model cannot use future telemetry or future field-visit outcomes to improve the prediction.

---

## 5. It cannot inspect gateways physically

The system only analyses available digital data.

It cannot replace an engineer's physical inspection of a gateway.

---

## 6. It cannot guarantee the lowest possible operational cost

The model is designed to improve gateway ranking under the 15-visit constraint.

Actual cost depends on the hidden fault episodes and their timing.

Therefore, the model cannot guarantee a specific final cost before evaluation.

---

## 7. It cannot automatically adapt itself during prediction

The model is trained offline.

It does not continuously retrain itself while generating predictions.

A changed environment, new gateway population, or changed failure behaviour may require retraining.

---

## 8. It cannot assume historical field visits are unbiased

Historical visits represent previous operational decisions.

An unvisited gateway cannot automatically be considered healthy, and a visited gateway is not necessarily representative of the whole gateway population.

---

## 9. It cannot replace operational judgement

The ranking is intended to support field-visit prioritisation.

Operations teams should consider the ranking together with current operational information and engineering knowledge.