# Reference Tinker recipe

This recipe connects Feather Train's Verifiers rollouts to Tinker Cookbook. It
supplies the current sampling policy and converts recorded trajectories into
Cookbook training samples. World execution and grading remain in Feather Train.
See the [README](README.md) for setup and the example configuration.

## Configuration

The recipe overlaps rollouts and training with async streaming, one minibatch, and
one optimizer substep per iteration. Rollout concurrency is independent of training
batch size.

| Setting | Meaning |
| --- | --- |
| `group_size` | Sibling rollouts per task. |
| `groups_per_batch` | Rollout groups per training update. |
| `max_inflight_rollout_groups` | Concurrent groups; multiply by `group_size` for the rollout limit. |
| `max_steps_off_policy` | Maximum sampling-policy lag. |
| `max_steps` | Total training iterations, including those completed before resume. |
| `max_tokens` | Maximum tokens per response, also bounded by remaining context space. |
| `harness.compaction.context_window_tokens` | Context budget used for compaction. |

Defaults include CISPO with clipping thresholds 0.0/5.0, learning rate 1e-5, and
LoRA rank 64. See [TrainConfig](src/feather_train/recipes/tinker/train.py) and
[TrainingConfig](src/feather_train/recipes/tinker/config.py) for the full schema.

## Retries and resume

Verifiers owns rollout retries. Exhausted nonfatal rollouts are excluded from
training and the group penalty mean; successful siblings remain trainable. An
all-failed group or fatal harness error stops training.

To resume, set `behavior_if_exists="resume"` with the same dataset and log path.
Cookbook restores model/optimizer checkpoints, but not in-flight rollout queues.
Resolved launch settings are saved in `feather-configs.jsonl`.

With Cookbook 0.5.7, shutdown ordering or discarded stale groups can leave a run
short of its requested updates. Cookbook owns end-of-data handling and final
checkpointing; use optimizer metrics to determine how many updates completed.

## Training samples

The recipe converts each sampled response in a Verifiers trace into a Cookbook
transition with its recorded prompt, completion tokens, and logprobs. Cookbook
packs cumulative, prefix-compatible turns into one datum per context window and
uses a loss mask to select generated tokens. Compaction creates separate windows
for the summary request and continuation.

To adapt this integration, see the three [framework hooks](architecture.md#integrating-another-training-framework).
