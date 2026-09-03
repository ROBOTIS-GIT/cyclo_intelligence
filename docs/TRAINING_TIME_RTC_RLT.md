# Training-Time RTC Loss Contract for GR00T N1.7 + RLT

Status: partially implemented; runtime transport, scheduling, the logical
`16x19` capability gate, and RLT reference-offset plumbing exist. Internal
`40x132`/four-step manifest checks, model-side per-action-token flow
conditioning, the postfix-only TT-RTC loss, and TT-RTC sampling are not yet
implemented. No current checkpoint is deployment-qualified for TT-RTC.

This document defines the training and runtime contract for adding
Training-Time Real-Time Chunking (TT-RTC) to the CYCLO GR00T N1.7 action head
and then passing the time-aligned continuation to the RLT action MLP. The fixed
contract is split between the checkpoint's padded internal tensor and the
CYCLO embodiment's logical action tensor:

| Symbol | Meaning | Required value |
|---|---|---:|
| `H_model` | GR00T internal padded action horizon | 40 steps |
| `A_model` | GR00T internal padded action dimension | 132 |
| `H_valid` | CYCLO logical/reference horizon | 16 steps |
| `A_valid` | CYCLO logical action dimension | 19 |
| `D_max` | Maximum supported handoff delay | 6 steps |
| `C` | RLT action chunk length | 10 steps |
| `f` | Action clock | 15 Hz |
| `N_flow` | Current checkpoint denoising steps | 4 |

The values `H_model=40`, `A_model=132`, and `N_flow=4` come from the current
`showroom_groot/config.json`; the CYCLO processor pads the logical `16x19`
action into that model tensor. One action step is approximately 66.7 ms, so
`D_max = 6` represents a maximum planned end-to-end handoff delay of 400 ms.
The critical shape invariant is

```text
H_valid - D_max >= C       16 - 6 = 10
```

GR00T generates a padded raw tensor `[B,40,132]`. The logical normalized
reference is first cropped as `raw[:, :16, :19]`. For a request with delay
`d`, the RLT MLP must then receive exactly `reference[:, d:d+10, :]`; it must
not always receive `reference[:, :10, :]`.

## Source boundary: PI method versus CYCLO design

The following distinction is normative. Do not describe every item in this
document as a result from the PI paper.

| Topic | PI Training-Time RTC | CYCLO GR00T + RLT contract |
|---|---|---|
| Base policy | Demonstrated by PI by fine-tuning a `pi0.6` flow policy | NVIDIA GR00T N1.7 action head |
| Training objective | Clean committed action prefix, ordinary flow corruption on the postfix, postfix-only loss | Same training principle, translated to GR00T's native flow convention |
| Delay | Random prefix length during fine-tuning | Integer `d` sampled from `0..6` inclusive at 15 Hz |
| Horizon | PI experiments use their policy's own horizon and rate | Internal `40x132`; logical `H_valid=16`, `C=10`, `A_valid=19`, 15 Hz |
| Downstream policy | No CYCLO RLT contract | RLT consumes `reference[d:d+10]` |
| Value learning | Not the CYCLO Stage 2 critic design | Credit and bootstrap are aligned to continuation execution time |
| Parameters | PI reports no additional learnable parameters for the conditioning mechanism | Existing weights can be reused, but GR00T tensor-shape support must change and its action head must be fine-tuned |

PI's official RTC page states in its December 8, 2025 update that Training-Time
RTC was used for the `pi*0.6` espresso-making demonstration. The earlier
`pi*0.6` release article and model card do not establish TT-RTC as a default
property of every `pi0.6`/`pi*0.6` checkpoint or every reported evaluation.
For CYCLO, a checkpoint is therefore TT-RTC-capable only when the metadata and
qualification checks in this document say so.

The current implementation boundary is:

- The [GR00T N1.7 action head](../cyclo_brain/policy/groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)
  uses one scalar flow time per batch item, corrupts all action tokens, and
  applies loss over the full valid action mask. It has no
  `get_action_tt_rtc` model entry point.
- Its existing inference option called `RTC` seeds an overlap from the previous
  chunk and applies frozen/ramped velocity updates. It was not trained with
  clean prefix conditioning and is not PI's Training-Time RTC. It also is not
  PI's inference-time pseudoinverse/VJP RTC algorithm.
- Request fields, engine transport, the common-runtime scheduler, the strict
  logical-contract capability gate, and deadline/stale-result rejection are
  implemented. They intentionally fail closed while the model entry point is
  absent. The gate must still be extended to verify the internal `40x132`
  tensor and four denoising steps. The current TensorRT path is also rejected
  for TT-RTC.
- The [RLT inference adapter](../cyclo_brain/policy/groot/runtime/rlt_adapter.py)
  and shadow policy accept `reference_offset_steps` and can select
  `reference[:, d:d+10]`. Ordinary inference defaults the offset to zero; a
  TT-RTC caller must pass the validated request delay. This slicing support
  alone does not make the GR00T reference prefix-conditioned.
- The current [RLT Stage 2](../cyclo_brain/algorithm/rl/rlt/stage2.py) critic
  does not carry the delayed-handoff context or variable execution duration
  defined below.

## Time and indexing contract

All indices are action-clock indices, not wall-clock sample numbers:

- `t`: time at which the observation for a VLA request is captured.
- `d`: handoff delay selected for that request, where `0 <= d <= 6`.
- `e = t + d`: time at which the newly produced RLT continuation starts.
- `P_t^d`: logical commands already committed for `[t, e)`, with shape
  `[d, 19]`. The policy processor normalizes and pads them to `[d,132]` before
  inserting them into the internal model tensor.
- `R_t^raw`: the padded GR00T result with shape `[40,132]`.
- `R_t`: the TT-RTC GR00T reference, with shape `[16, 19]`, indexed over
  `[t, t+16)`, obtained as `R_t^raw[:16,:19]`.
- `U_t`: the final RLT continuation, with shape `[10, 19]`, executed over
  `[e, e+10)`.

The alignment is:

```text
absolute time       t                         e=t+d                 e+10
                    |-------------------------|----------------------|
GR00T R_t index      0 ...                 d-1 d ...             d+9
meaning              committed prefix P_t^d   predicted continuation
RLT reference                                    R_t[d:d+10]
RLT output                                       U_t[0:10]
```

The request transport carries the first `d` committed commands in the physical
logical action domain. The GR00T policy-side processor must apply the checkpoint
normalization exactly once, pad each row from 19 to 132 components, and place
the result into the first `d` positions of `R_t^raw`. Consequently, the first
`d` elements of logical normalized `R_t` represent exactly those committed
commands. They are not measured joint states and not an uncommitted prediction
from an older model call.

`d` is a planned handoff delay selected before the model call from a qualified
end-to-end latency class. A conservative conversion is

```text
d = ceil(latency_budget_seconds * 15)
```

where the budget includes observation preprocessing, GR00T inference, RLT MLP,
decoding, IPC, and enqueue time. This formula is a CYCLO scheduling policy, not
a PI paper result. A result that misses its selected handoff deadline cannot be
made correct by retroactively changing `d` or slicing at a different offset.
If the required value is greater than six, the runtime must reject/fallback;
silently clamping it to six violates temporal alignment.

The first request has no previously committed prefix. It must use `d=0` and a
safe blocking/bootstrap path, or obtain a prefix from an explicitly defined
startup controller. TT-RTC does not create actions that can execute before the
first inference completes.

## GR00T flow-matching objective

This section uses the convention already present in GR00T N1.7:

```text
x_i(tau) = (1 - tau) * epsilon_i + tau * a_i
u_i      = a_i - epsilon_i
```

Thus `tau=0` is the noise endpoint and `tau=1` is the clean action endpoint.
The TT-RTC paper's displayed Eq. (2) prints the opposite velocity sign, but its
executable Algorithm 1 targets `action - noise` and integrates forward with
`x = x + dt * velocity`, matching current GR00T. The implementation must follow
that executable/native convention; copying a displayed equation without
checking its endpoint and velocity sign is not valid.

For each training example, independently sample

```text
d ~ UniformInteger({0, 1, 2, 3, 4, 5, 6}).
```

Let `i in [0,H_model)` and `j in [0,A_model)`. Let `p_i = 1[i < d]` be the
prefix-time mask and let `m_ij` be the processor-produced validity mask on the
padded `[40,132]` tensor. For CYCLO data it selects the logical `16x19` region
and excludes temporal and component padding. Sample the normal GR00T flow time
`tau` and noise `epsilon`, then construct per-action-token times and inputs as

```text
tau_i = 1                         if i < d
        tau                       otherwise

x_i   = a_i                       if i < d
        (1 - tau) epsilon_i
          + tau a_i               otherwise.
```

The required loss is postfix-only:

```text
                 sum_{i,j} m_ij (1 - p_i)
                   * ||v_theta(x, tau_token, o)_ij - (a_i - epsilon_i)_j||^2
L_TT-RTC = -----------------------------------------------------------------
                         sum_{i,j} m_ij (1 - p_i)
```

No direct residual term from a prefix output may enter the numerator or
denominator. The prefix still participates as an input to self-attention, so
postfix loss is allowed—and intended—to train the network to use prefix
information. "No prefix loss" does not mean detaching the prefix token from the
model.

At `d=0`, this reduces to the ordinary full-valid-mask GR00T flow objective. At
`d=6`, exactly ten **valid logical** temporal positions remain available for
the RLT continuation. Padded internal positions `16..39` remain loss-masked and
must not be counted as extra supported continuation steps.

## Optimization stages and source provenance

TT-RTC flow fine-tuning, RL-token representation learning, and RLT online RL
have different targets and masks. Do not call all of them "the TT-RTC loss."
The sequential, frozen boundaries below describe the current CYCLO training
implementation. They are not a universal requirement of the RLT paper: RLT
Stage 1 explicitly permits jointly optimizing its reconstruction loss with a
supervised VLA loss, `L_ro + alpha * L_vla`. CYCLO currently chooses
`alpha=0`, freezes GR00T during RL-token training, and runs TT-RTC action-head
fine-tuning as a separate stage.

### 1. CYCLO GR00T TT-RTC fine-tuning

Use the postfix-only flow loss defined above. The VLM may remain frozen, but
the GR00T action expert that consumes token-wise flow time must be trainable.
This CYCLO stage produces the TT-RTC-capable **VLA Action** route. It is based
on PI's Training-Time Action Conditioning paper, not on the RLT losses.

### 2. RLT Stage 1: official objective and current CYCLO reduction

Official RLT writes the frozen VLA tokens as `z_bar_i = stop_gradient(z_i)`,
the RL-token encoder as `g_phi`, the autoregressive decoder as `d_phi`, and its
output head as `h_phi`. Its Eq. (2) is

```text
L_ro(phi) = E_D [sum_{i=1}^M
    ||h_phi(d_phi([z_rl, z_bar_1:i-1]))_i - z_bar_i||_2^2].
```

Current CYCLO first compacts the frozen token sequence according to its
configured `token_selection` policy. Let `F_sel(o)` denote that compacted
sequence, `m_bi` its post-selection `target_valid` mask, and `F_hat` the
teacher-forced decoder output. Its default `paper_per_sample_sum` implements
the masked, batch-averaged form of Eq. (2):

```text
L_ro = mean_b sum_i m_bi
         * ||F_hat_bi - stop_gradient(F_sel(o)_bi)||_2^2.
```

CYCLO also exposes a non-default `valid_token_mean` reduction:

```text
L_ro_valid_mean =
    sum_{b,i} m_bi * ||F_hat_bi - stop_gradient(F_sel(o)_bi)||_2^2
    / sum_{b,i} m_bi.
```

The raw GR00T `token_valid` mask must not be substituted after token selection;
the implementation uses the compacted `target_valid` mask. The reconstruction
decoder is discarded after Stage 1, and the frozen RL-token encoder is retained
for Stage 2 and inference.

### 3. RLT Stage 2: official equations, current code, and TT extension

These three layers must remain distinguishable.

#### Official RLT, Eqs. (3)-(5)

Official RLT defines `x=(z_rl,s^p)` and a VLA reference chunk `a_tilde`. Its
critic is off-policy: the critic action comes from replay, not necessarily from
the current actor. Eq. (3) uses

```text
L_Q = E_{(x,a_exec,x_next)~B}
        [(Q_hat - Q_psi(x,a_exec))^2]

Q_hat = sum_{j=0}^{C-1} gamma^j reward[j]
        + gamma^C E_{a_next~pi_phi}[Q_psi_target(x_next,a_next)].
```

Eq. (4) defines the Gaussian actor

```text
pi_phi(a | x,a_tilde)
    = Normal(mu_phi(x,a_tilde), sigma_fixed^2 I),
```

and Eq. (5) defines

```text
L_pi = E_{x~B, a~pi_phi, a_tilde~pi_vla}
         [-Q_psi(x,a) + beta * ||a-a_tilde||_2^2].
```

The second term is the RLT paper's VLA-reference policy constraint, not an
arbitrary imitation term added from generic TD3+BC. During an optional human
intervention, the official RLT algorithm replaces the stored reference with
the human action, so in that case the constraint is intentionally anchored to
the intervention. A CYCLO dataset that does not support interventions must say
so rather than silently changing the meaning of `reference_actions`.

#### Current CYCLO Stage 2

Current CYCLO implements the paper objective with twin critics, a minimum over
target critics, one Monte Carlo actor sample, delayed actor updates, a fixed
Gaussian standard deviation, and Polyak critic-target updates. It has no target
actor. Let `K` be a Bernoulli keep mask with one scalar shared across each
batch row's whole reference chunk. The exact current update is

```text
x_code       = (z_rl, proprio)
a_exec       = batch.executed_actions
a_phi        = mu_phi(x_code, K * batch.reference_actions)
               + sigma_fixed * epsilon
a_next       = mu_phi(x_code_next, batch.next_reference_actions)
               + sigma_fixed * epsilon_next

y_code       = batch.reward
               + batch.bootstrap_discount
                 * min(Q1_target(x_code_next,a_next),
                       Q2_target(x_code_next,a_next))

L_critic     = MSE(Q1(x_code,a_exec), y_code)
               + MSE(Q2(x_code,a_exec), y_code)

L_actor      = -mean(Q1(x_code,a_phi))
               + policy_constraint_weight
                 * mean_batch(||a_phi-batch.reference_actions||_2^2).
```

Reference dropout changes only the actor input; the constraint target remains
the original unmasked reference. `policy_constraint_weight` is the CYCLO name
for the paper's `beta`. Replay action `a_exec`, current actor sample `a_phi`,
and next actor sample `a_next` are distinct variables.

#### Target CYCLO TT-RTC extension

For a TT-RTC transition, `batch.reference_actions` must be constructed as

```text
R_shift = R_t[:, d:d+10, :]          # never [:, :10, :] when d > 0.
```

The actor receives this shifted ten-step reference. Credit begins at execution
time `e=t+d`, as specified below. The target critic context additionally
contains the committed prefix, delay, and shifted reference. That augmented
critic is a CYCLO TT-RTC design; it is neither official RLT Eq. (3) nor the
current `Q(z_rl,proprio,action)` implementation. Stage-2 batch/spec/network and
checkpoint formats must change before equations using the augmented context
can execute.

### Required model plumbing

The [current action encoder](../cyclo_brain/policy/groot/Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py)
rejects a `[B,H]` timestep tensor and broadcasts a single `[B]` time over all
actions. The DiT AdaLN path similarly assumes one embedding per batch item.
Training-Time RTC therefore requires all of the following:

1. The action encoder accepts quantized per-token timestep codes with shape
   `[B,40]`, matching `H_model`, not the cropped logical horizon.
2. The DiT accepts quantized timestep codes with shape `[B,41]`: one state
   token followed by forty internal action tokens. Its timestep encoder then
   produces embeddings with shape `[B,41,E]`, where `E` is the DiT embedding
   width, and AdaLN consumes them token-wise.
3. The state token uses the example's ordinary sampled flow time. Prefix action
   tokens use the clean endpoint; postfix action tokens use the sampled time.
4. AdaLN and output conditioning apply scale/shift token-wise instead of
   inserting a batch-wide singleton dimension.
5. The scalar-time path remains supported and numerically equivalent for old
   callers and the `d=0` regression test.

The existing sinusoidal/time MLP weights can be reused, so these shape changes
do not inherently add learnable parameters. They do change forward semantics.
A legacy action-head checkpoint must be fine-tuned with this objective; merely
loading it through the new tensor path does not make it prefix-conditioned.
The VLM backbone may remain frozen, but the action encoder, action DiT/expert,
and decoder used by the flow loss must be trainable according to the chosen
GR00T fine-tuning configuration. Because the input timestep shape and AdaLN
broadcast contract change, the current scalar-time TensorRT engine cannot be
reused; a TT-RTC engine must be exported and qualified separately after the
model implementation exists.

### Training pseudocode

This pseudocode is a contract, not a drop-in patch. `quantize_time` must
preserve the exact clean endpoint code used at inference.

```python
def training_time_rtc_loss(backbone_output, batch):
    # Processor-padded model tensors, not decoded logical actions.
    a = batch.action                    # [B, H_model=40, A_model=132]
    valid = batch.action_mask.bool()    # [B, 40, 132]
    B, H_model, A_model = a.shape
    assert (H_model, A_model) == (40, 132)
    assert valid.shape == a.shape
    assert not valid[:, 16:, :].any()   # logical horizon is 16
    assert not valid[:, :, 19:].any()   # logical action width is 19

    # Independent delay and ordinary flow-time sample for each example.
    d = torch.randint(0, 7, (B,), device=a.device)       # inclusive 0..6
    tau = action_head.sample_time(B, a.device, a.dtype) # [B]
    eps = torch.randn_like(a)

    index = torch.arange(H_model, device=a.device)[None, :]
    prefix = index < d[:, None]                          # [B, 40]

    tau_action = tau[:, None].expand(B, H_model).clone()
    tau_action[prefix] = 1.0                             # clean endpoint

    x = ((1.0 - tau_action[..., None]) * eps
         + tau_action[..., None] * a)
    # Copy all 132 components. Padding is zero and excluded by `valid`.
    x = torch.where(prefix[..., None], a, x)
    target_velocity = a - eps                            # GR00T convention

    action_time_codes = quantize_time(tau_action)        # integer [B, 40]
    state_time_codes = quantize_time(tau)[:, None]       # integer [B, 1]
    dit_time_codes = torch.cat(
        [state_time_codes, action_time_codes], dim=1
    )                                                    # integer [B, 41]

    action_features = action_encoder(
        x, action_time_codes, batch.embodiment_id
    )
    hidden = torch.cat([state_features, action_features], dim=1)
    pred = action_dit_and_decoder(
        hidden, backbone_output, dit_time_codes,
        batch.embodiment_id,
    )[:, -H_model:, :A_model]

    postfix_valid = valid & ~prefix[..., None]
    denominator = postfix_valid.sum()
    if denominator == 0:
        raise ValueError("TT-RTC batch has no valid postfix target")

    residual = (pred - target_velocity).square()
    loss = (residual * postfix_valid).sum() / denominator
    return loss, {"delay_steps": d, "postfix_mask": postfix_valid}
```

Samples must contain a contiguous 16-step logical target after `t` and must not
cross episode boundaries. The processor pads it to `40x132`; padding remains
excluded by `action_mask`. For `d>0`, the first `d` clean logical actions,
padded to 132 components, are the ground-truth training analogue of commands
already committed at deployment.

## Asynchronous inference contract

Training-Time RTC removes the need for inference-time gradient/VJP guidance; it
does not remove GR00T inference. GR00T is still called on every reference refill.
Continuity comes from executing an already committed prefix while the new VLA
and RLT computation is in flight.

At every flow integration step, the prefix must be clamped to the committed
commands and presented with clean-endpoint time. Updating only the postfix once
at initialization is insufficient.

The helper names in the following contract pseudocode are illustrative; in
particular, `normalize_and_pad_action_prefix` is a required processor operation,
not an API that exists in the current codebase.

```python
@torch.inference_mode()
def sample_ttrtc(observation_t, committed_prefix_physical, d):
    assert 0 <= d <= 6
    assert committed_prefix_physical.shape == (d, 19)

    # Apply the checkpoint action transform exactly once, then pad 19 -> 132.
    committed_prefix_model = processor.normalize_and_pad_action_prefix(
        observation_t, committed_prefix_physical,
        output_dim=132,
    )                                                       # [d,132]

    x = torch.randn(1, 40, 132, device=device, dtype=model_dtype)
    prefix = torch.arange(40, device=device) < d
    x[:, prefix] = committed_prefix_model

    assert len(flow_schedule) == 4  # current showroom checkpoint contract
    for tau, dt in flow_schedule:
        tau_action = torch.full((1, 40), tau, device=device)
        tau_action[:, prefix] = 1.0
        x[:, prefix] = committed_prefix_model

        velocity = action_head(
            observation_t,
            x,
            action_time_codes=quantize_time(tau_action),
            state_time_codes=quantize_time(torch.tensor([tau], device=device)),
        )
        x[:, ~prefix] += dt * velocity[:, ~prefix]
        x[:, prefix] = committed_prefix_model

    return x                                                # raw [1,40,132]


def produce_next_chunk(request):
    t = request.observation_action_index
    d = request.preselected_delay_steps
    if not 0 <= d <= 6:
        return fail_closed("delay is outside trained support")

    prefix = committed_command_queue.slice(t, t + d)          # physical [d,19]
    raw = sample_ttrtc(request.observation, prefix, d)         # [1,40,132]
    reference = raw[:, :16, :19]                              # [1,16,19]
    rlt_reference = reference[:, d:d + 10, :19]              # [1,10,19]
    assert rlt_reference.shape == (1, 10, 19)

    # The current actor signature consumes the already shifted reference.
    continuation = rlt_actor(
        request.z_t, request.proprio_t, rlt_reference,
    )                                                        # [1,10,19]

    e = t + d
    if not result_is_ready_before_action_index(e):
        return fail_closed("missed TT-RTC handoff deadline")
    committed_command_queue.enqueue_at(e, continuation)
```

The current actor relies on the shifted reference to carry the continuation
proposal and does not accept `d` or the committed prefix directly. The dataset
and transition manifest must still record both fields. The augmented target
critic context below requires them explicitly and is not yet implemented.

Only one result may own a handoff slot. Sequence IDs and request timestamps must
discard stale or reordered responses. On a deadline miss, the safety policy may
continue a separately guaranteed queued prefix or enter a safe stop; it must
not publish the late chunk under a falsely relabeled time index.

## RLT actor and critic temporal alignment

The RLT decision must be represented at the information time actually
available during inference. Official RLT and current CYCLO use
`x_code=(z_t,q_t)`. The target TT-RTC critic extends it to

```text
X_t^TT = (z_t, q_t, P_t^d, d, R_t[d:d+10]).
```

where `z_t` and proprioception `q_t` come from the request observation at `t`.
Do not train the actor or critic with `z_e`/`q_e` captured at execution start
`e=t+d`: those observations are in the future when the inference request is
made and would create train/deployment leakage.

The chosen action argument to the critic is only the final RLT continuation
`U_t` that actually executes over `[e,e+10)`. Neither the committed prefix nor
the full 16-step GR00T reference may be mislabeled as the chosen action.

For the first implementation, enforce one-in-flight fixed-boundary handoffs:

```text
e_next = e + C = e + 10.
```

The target TT-RTC transition is then

```text
(X_t^TT, U_t, G_t, gamma^10, X_t_next^TT)
```

with

```text
G_t = sum_{j=0}^{9} gamma^j * reward[e + j]

U_next ~ pi_phi(. | z_t_next, q_t_next, R_shift_next)

y = G_t + gamma^10
    * min(Q_target_1(X_t_next^TT, U_next),
          Q_target_2(X_t_next^TT, U_next))
```

There is no target actor in official RLT or the current CYCLO learner;
`U_next` is a sample from the current actor and the two Q networks are the
target networks. The bootstrap term is zero for a terminal or invalid next
transition. The next context is anchored at the next request time `t_next`, not
necessarily at `e_next`; it includes the next committed prefix and delay that
bridge `t_next -> e_next`.

This has four consequences for the training dataset and networks:

1. Rewards from the committed prefix `[t,e)` are not credited to `U_t`.
2. `executed_actions` contains exactly the ten final RLT commands from
   `[e,e+10)`, in command order, after all runtime transforms that define the
   training action domain.
3. The target TT-RTC critic must condition on the delayed decision context,
   including at least `d` and the committed prefix (or a versioned, validated
   encoding of them), in addition to request-time features and `U_t`. The
   current critic signature `(z_rl, proprio, actions)` is not sufficient for
   this contract.
4. `next_*` fields are assembled from the next request context and its own
   `P`, `d`, and shifted reference—not by shifting arrays by an assumed row.

If later scheduling permits an earlier handoff, define
`Delta = e_next - e`, truncate both executed actions and reward accumulation at
`e_next`, carry an explicit executed mask/duration, and use `gamma^Delta`.
The current fixed `[10,19]` Stage 2 batch has no such duration mask. Until that
schema and critic are implemented, overlapping/early replacement of an RLT
chunk is disallowed for training data and deployment qualification. Padding a
partially executed chunk and treating padding as executed action is invalid.

## Required checkpoint and dataset metadata

Each TT-RTC GR00T checkpoint must ship with a machine-readable manifest. Each
RLT Stage 2 checkpoint must bind to that GR00T manifest and to the temporal
alignment schema. The exact serialization may be JSON or YAML, but the logical
fields below are required.

```yaml
schema: cyclo.training-time-rtc/v1

training_time_rtc:
  trained: true
  source_method: pi-training-time-action-conditioning
  implementation: cyclo-gr00t-n1.7-ttrtc
  action_horizon: 16             # logical H_valid; current gate field
  action_dimension: 19           # logical A_valid; current gate field
  model_action_horizon: 40       # padded H_model from GR00T config
  model_action_dimension: 132    # padded A_model from GR00T config
  num_inference_timesteps: 4
  action_hz: 15.0
  max_delay_steps: 6
  delay_sampling:
    type: uniform_integer
    min_inclusive: 0
    max_inclusive: 6
  prefix_input: ground_truth_clean_action
  loss_region: postfix_only
  per_action_timestep: true
  state_timestep: sampled_flow_time
  flow_convention:
    noise_endpoint: 0.0
    clean_endpoint: 1.0
    velocity_target: action_minus_noise
    time_quantizer: <name-and-version>
    clean_endpoint_code: <exact-integer-or-float-code>
  base_groot_checkpoint:
    identifier: <checkpoint-id>
    sha256: <digest>
  processor_fingerprint: <digest>
  action_schema_fingerprint: <digest>
  dataset_fingerprint: <digest>
  code_revision: <git-revision>

rlt:
  chunk_length: 10
  reference_horizon: 16
  reference_slice: "[d:d+10]"
  reference_source_checkpoint_sha256: <tt-rtc-groot-digest>
  stage1_fingerprint: <digest>
  stage2_fingerprint: <digest>
  temporal_alignment_schema: request-observation_execution-credit/v1
  handoff_policy: fixed_boundary_one_in_flight
  handoff_duration_steps: 10
  critic_context: [z_at_request, proprio_at_request, delay_steps,
                   committed_prefix, shifted_reference]
  critic_context_status: target_not_implemented
  reward_window: "[execution_start:execution_start+10)"
  bootstrap_discount: gamma_pow_10

qualification:
  status: unqualified
  hardware: <device-and-host-description>
  software_image: <immutable-image-id>
  latency_scope: preprocess+groot+rlt+decode+ipc+enqueue
  latency_ms: {p50: null, p95: null, p99: null, maximum: null}
  selected_delay_steps: null
  measured_at: null
```

The loader must fail closed when `trained` is absent/false, when any logical or
model horizon, dimension, denoising-step count, rate, flow convention,
processor/action schema, or reference fingerprint differs, or when
`H_valid - max_delay_steps < chunk_length`. It must cross-check the declared
`40x132` model tensor and four denoising steps against the GR00T config rather
than trusting the TT-RTC manifest alone. A legacy GR00T or RLT checkpoint must
not receive inferred defaults that make it appear TT-RTC-qualified.

Training records also need, per decision, `request_action_index`,
`execution_start_action_index`, `delay_steps`, the exact committed prefix,
shifted reference, final executed continuation, reward interval, terminal flag,
next request/execution indices, and the applicable fingerprints. Wall-clock
timestamps should be retained for auditing, but integer action indices are the
source of truth for slicing and critic credit.

## Validation gates

All gates below must pass before enabling asynchronous robot execution.

### Loss and model-shape tests

- `d=0` with identical RNG state is numerically equivalent to the existing
  GR00T objective and scalar-time forward path on the padded `[B,40,132]`
  tensor.
- For every `d`, prefix inputs equal clean ground-truth actions exactly and
  prefix token times equal the clean endpoint code. Logical 19-dimensional actions are
  normalized once and padded to 132 components before insertion.
- Direct loss residual and denominator contribution are zero on every prefix
  component; postfix attention may still propagate gradients through prefix
  features.
- `d=6` leaves exactly ten valid logical temporal positions; internal positions
  `16..39` and components `19..131` remain loss-masked.
- Batched examples may have different `d`; action time has shape `[B,40]` and
  DiT token time has shape `[B,41]` without accidental broadcasting.
- A batch with no valid postfix components raises an error rather than
  returning a zero/NaN loss.
- Samples never cross episode/termination boundaries, provide the required 16
  logical future positions, and are padded to `40x132` with an exact validity
  mask.
- RLT Stage 1 regression tests distinguish `paper_per_sample_sum` from
  `valid_token_mean` and use the post-selection `target_valid` mask.
- RLT Stage 2 regression tests keep replay `executed_actions`, current actor
  samples, and next actor samples distinct; reference dropout affects only the
  actor input, and no target actor state is expected.
- Old scalar-time checkpoints still run through the compatibility path, but
  are rejected by the TT-RTC deployment loader unless fine-tuned and marked.

### Reference and action-domain tests

- Synthetic ramp actions prove that `reference[d:d+10]` maps to absolute times
  `[t+d,t+d+10)` for every `d=0..6` after cropping raw `[40,132]` output to
  logical `[16,19]`; include explicit `d=0` and `d=6` tests.
- Prefix, GR00T reference, RLT output, replay actions, and critic actions share
  the same 19-D ordering, normalization, codec, and embodiment ID.
- Runtime compares the prefix with commands actually committed in the command
  queue, not measured state or the unsent tail of a previous prediction.
- Checkpoint/processor/action-schema fingerprints round-trip through save and
  load and reject every deliberate mismatch.
- Capability loading cross-checks logical `16x19`, model `40x132`, and four
  denoising steps against both processor and GR00T model configs.

### Critic and replay tests

- A toy timestamped episode verifies that action `U_t` receives rewards only
  from `[e,e+10)`, never `[t,e)`.
- The fixed-boundary implementation always records
  `execution_start_next - execution_start = 10` and
  `bootstrap_discount = gamma**10`.
- Terminal transitions use zero bootstrap. Truncated or partially executed
  chunks are rejected until a duration/mask schema is implemented.
- `next_z`, `next_proprio`, next prefix, next `d`, and next shifted reference
  all come from the same next request ID.
- Reordered, duplicate, late, and stale inference results cannot enter replay
  under a different request or execution index.

### Runtime and latency tests

- `d` is selected before inference, is in trained support `0..6`, and the exact
  same value controls prefix length, GR00T conditioning, RLT slicing, and
  handoff time.
- End-to-end latency measurement covers the complete path through RLT enqueue,
  not only GPU kernel or GR00T forward time.
- Production hardware meets the selected handoff deadline at the approved tail
  percentile with an explicit safety margin. If it requires `d>6`, deployment
  fails qualification.
- The prefix is re-clamped at every denoising step, and only the postfix is
  integrated.
- Startup, deadline miss, queue underrun, stale response, pause/resume, and
  emergency-stop behavior are exercised on a non-moving/simulated robot before
  hardware rollout.

## What TT-RTC does and does not solve

TT-RTC lets the robot continue executing previously committed commands while a
fresh VLA call is running, and trains the new reference to continue after those
commands. It also avoids inference-time RTC's gradient/VJP guidance overhead.
It does not make GR00T or the RLT MLP instantaneous, eliminate refill calls, or
guarantee that arbitrary hardware meets the deadline. The 400 ms support window
is a model/data contract; measured end-to-end latency remains a deployment gate.

For context only, PI's real-world experiment details report 8,000 task
fine-tuning steps with batch size 512, uniformly sampled delays from 0 through
10 on a 50 Hz robot, and a remotely served single NVIDIA H100 with five
denoising steps. They measured mean end-to-end latency of 108 ms for
Training-Time RTC (`d` approximately 5) and 135 ms for inference-time RTC (`d`
approximately 7). The paper identifies the GPU but does not provide the server
CPU, RAM, networking hardware, or detailed serving-stack specification. Those
numbers characterize PI's setup, not CYCLO's camera pipeline, network, GR00T
checkpoint, RLT MLP, or production host. CYCLO must record its own
qualification measurements in the manifest above.

## Primary references

- Physical Intelligence, [Real-Time Chunking with Large Models](https://www.pi.website/research/real_time_chunking)
  (includes the December 8, 2025 Training-Time RTC update).
- Black et al., [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964).
- Xu et al., [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://www.pi.website/download/rlt.pdf).
- Physical Intelligence, [Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt).
- Physical Intelligence, [`pi*0.6`: a VLA That Learns From Experience](https://www.pi.website/blog/pistar06).
- Physical Intelligence,
  [reference TT-RTC simulation implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py).
