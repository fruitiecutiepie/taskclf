# features/dynamics

Temporal dynamics features: rolling means over 5/15-minute windows and inter-bucket deltas for interaction metrics.
`DynamicsTracker` is implemented as a slotted dataclass with the same
constructor arguments (`rolling_5`, `rolling_15`).

::: taskclf.features.dynamics
