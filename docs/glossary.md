# Glossary

| Term | Meaning here |
|---|---|
| Defect / label | `label == 1` in the CSV; `-1` is a normal wafer |
| Prevalence | Share of defects in a set of rows (6.64% overall, 3.81% in the test partition) |
| Legacy PS-L wafer | A wafer whose `sensor_0` equals exactly 0.0. A sentinel, not a measurement; none of the 146 such wafers is defective |
| Indicator / flag columns | The four extra 0/1 features: legacy wafer, and `sensor_1` codes 100, 200, 300 on legacy wafers |
| Chronological split | Rows ordered by timestamp, then cut: earliest 70% train, next 15% validation, last 15% test |
| Validation set | Used to pick the checkpoint and the alert threshold |
| Test set | Held out from training and selection. Already inspected, so results are exploratory |
| Alert rate | Share of rows the model flags. The notebook targets 10% on validation |
| Threshold | Score at or above which a wafer is flagged |
| Average precision (AP) | Area under the precision-recall curve; a random model scores about the prevalence |
| ROC-AUC | Chance that a random defect scores higher than a random normal wafer; 0.5 is chance |
| Lift | Precision of the alerts divided by the prevalence |
| Bootstrap CI | Interval from resampling the test rows 2,000 times; wide here because there are only 9 test defects |
| Parity check | The notebook reloads its saved files through the app's code and requires identical features, scores and alerts |
