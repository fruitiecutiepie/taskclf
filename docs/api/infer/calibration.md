# infer.calibration

Probability calibration hooks for post-model adjustment, including
per-user calibration via `CalibratorStore`.

## Overview

After the model produces raw class probabilities, the calibration layer
optionally adjusts them before the reject decision is made:

```
raw model probs → calibrate → reject decision
```

Calibration improves the reliability of predicted probabilities without
changing the underlying model.  Three calibrator implementations are
provided, all satisfying the `Calibrator` protocol:

| Class | Description |
|-------|-------------|
| `IdentityCalibrator` | No-op pass-through (default when no calibrator is configured) |
| `TemperatureCalibrator` | Single-parameter temperature scaling: `T > 1` softens, `T < 1` sharpens |
| `IsotonicCalibrator` | Per-class isotonic regression via `sklearn.isotonic.IsotonicRegression` |

All calibrators must preserve input shape and ensure the output sums
to 1.0 along the class axis.

## Calibrator protocol

Any object with a `calibrate(core_probs: np.ndarray) -> np.ndarray`
method satisfies the protocol.  The method accepts arrays of shape
`(n_classes,)` (single row) or `(n_rows, n_classes)` (batch).

## CalibratorStore

`CalibratorStore` manages per-user calibration with a global fallback.
It holds a global calibrator (fitted on all users' validation data) and
optional per-user calibrators for users that meet the personalization
eligibility thresholds (see [`train calibrate`](../train/calibrate.md)).

At inference time, `get_calibrator(user_id)` returns the per-user
calibrator if one exists, otherwise the global calibrator.
`calibrate_batch(core_probs, user_ids)` applies the correct calibrator
row-by-row.

### Directory layout

When persisted via `save_calibrator_store`, the store is written as:

```
<store_dir>/
    store.json          # metadata: method, user_count, user_ids
    global.json         # global calibrator (any type)
    users/
        <user_id>.json  # per-user calibrator (one file per user)
```

## Serialization

### Single calibrator

```python
from pathlib import Path
from taskclf.infer.calibration import (
    TemperatureCalibrator, save_calibrator, load_calibrator,
)

cal = TemperatureCalibrator(temperature=1.35)
save_calibrator(cal, Path("artifacts/calibrator.json"))

loaded = load_calibrator(Path("artifacts/calibrator.json"))
```

### Calibrator store

```python
from pathlib import Path
from taskclf.infer.calibration import (
    CalibratorStore, TemperatureCalibrator,
    save_calibrator_store, load_calibrator_store,
)

store = CalibratorStore(
    global_calibrator=TemperatureCalibrator(1.2),
    user_calibrators={"alice": TemperatureCalibrator(1.05)},
    method="temperature",
)
save_calibrator_store(store, Path("artifacts/calibrators"))

loaded_store = load_calibrator_store(Path("artifacts/calibrators"))
print(loaded_store.user_ids)  # ['alice']
```

## Eligibility

Per-user calibration is only fitted when the user meets the eligibility
thresholds defined in `core.defaults` (minimum labeled windows, days,
and distinct labels).  See the
[`train calibrate` CLI command](../train/calibrate.md) and
[personalization guide](../../guide/personalization.md) for details.

::: taskclf.infer.calibration
