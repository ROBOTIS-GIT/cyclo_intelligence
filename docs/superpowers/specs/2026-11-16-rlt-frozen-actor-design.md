# RLT frozen warmup Actor

The user approved training the Actor against Q while retaining successful human
behavior and anchoring failed states to a frozen warmup Actor. Freezing the
trainable Actor was rejected because it prevents policy improvement. Diagnostic
action clipping stopped numeric growth but drove actions to its boundaries.

Capture a detached, eval-mode Actor once on the first explicit RL phase, after
Critic warmup. Successful rows retain the existing human-action penalty. Failed
rows penalize the deterministic Actor mean against the frozen Actor evaluated
with the complete reference input, including when the trainable Actor receives
reference dropout. Keep the existing coefficient, active-joint selection, Q
objective, optimizer, and target-critic cadence.

Persist the frozen Actor in learner state; never refresh it on resume. Old
warmup bundles remain valid. An already-trained RL checkpoint without the frozen
Actor cannot start the new objective: it must restart from Critic warmup. Legacy
RL artifacts remain readable for inference and inspection.

Expose target magnitude, policy Q magnitude, reference deviation, and the two
Actor objective components in training logs. For explicit RL with the known
terminal-success reward contract, reject Q/target magnitudes over 10 (ten times
the theoretical return bound). For independently normalized MLP actions, reject
deviations over 10 normalized units from the selected behavior target. These
are coarse divergence alarms, not physical joint limits or clipping. Failed runs
must not publish a completed bundle.
The same checks apply before async actor publication/checkpointing. Policy
magnitudes include the post-optimizer deterministic policy as well as the sampled
training policy; split critic/actor updates require episode-success provenance.

Acceptance: meaningful objective/gradient tests, immutable anchor and exact
resume tests, compatibility and corruption tests, CLI logging/alarm tests, and
10,000-step real-data replay verification without replacing user checkpoints.

Implementation decisions: work in the existing live-mounted checkout to retain
the user's uncommitted normalization work. Existing bounds are statistical, not
physical limits; do not add arbitrary [-1, 1] action clipping.
