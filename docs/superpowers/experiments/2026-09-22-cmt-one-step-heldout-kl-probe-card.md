# CMT One-Step Held-Out KL Probe Experiment Card

Question: Does one CMT-weighted OPD optimizer step reduce held-out conditional reverse KL more than a uniform-OPD step starting from the same state?

Hypothesis: CMT weighting increases paired one-step KL improvement because it allocates more update mass to locally and sequentially teachable positions.

Baseline: Uniform OPD using the pinned OPD recipe at code commit `68439f8`, with weight one on every valid token in the same PPO group.

Variant: Replace only uniform token weights with the production CMT allocation; model, optimizer state, PPO group, OPD reference, learning rate, clipping, and held-out reference are paired and identical.

Primary metric: Mean paired gap `delta_cmt - delta_uniform`, where `delta = KL_before - KL_after` and KL is global token-mean conditional `KL(p_student,S || q_teacher,S)` on pre-step Student Top-16. Success requires a positive mean paired gap whose paired bootstrap 95% confidence interval excludes zero.

Guardrails: Exact shadow-state restoration; no change to the real optimizer-step count or callbacks; finite KL/loss/gradients; final weights remain bounded; probe runtime and peak host/GPU memory are logged.

Data / split: Training remains CompetitionMath train. The diagnostic uses a deterministic 64-example subset of CompetitionMath test, with fixed initial-student response prefixes. Leakage check: probe values never affect training, checkpoint selection, early stopping, or hyperparameters during the locked run.

Seeds: One fixed probe subset/generation seed (`20260922`) per training run. A later confirmatory study must repeat complete runs with at least three training seeds; individual optimizer steps are paired observations, not independent seed replicates.

Budget: No artificial compute cap. Run R1-R4 first; start a short real-data pilot only after exact restoration and resume logging pass. Promote to the full configured CMT run only if the pilot produces complete finite pairs without training-state drift.

Start rung: R1 static/config/unit verification, followed by an R2 tiny-model forward/backward and snapshot-restore test.

Exploratory-only: Forward KL, teacher cross-entropy, per-step scatter, runtime, support strata, and CompetitionMath accuracy may motivate later experiments but cannot confirm this card.

