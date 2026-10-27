# Stage-by-stage acceptance checklist

This is an execution/evidence checklist, not a second approval process. Run authorized stages in order and record actual results.

| Stage | Output | Evidence before moving on |
|---|---|---|
| Dataset import | `data/wan_pilot_v1/manifest_wan.jsonl` | 40 unique clips, expected model revision and split labels |
| Audit | `runs/wan_same_prompt_v2/audit.json` | 32 train, 4 val, 4 test; checksums pass; acceptance assumption explicit |
| Pixel/VAE preparation | `prepared/metadata.json`, `index.jsonl` | `complete=true`, 40 records; forward/reverse pixels independently encoded |
| One teacher target | `kd_cache/metadata.json` | At least one training draw, finite teacher target |
| GPU execution gate | `gpu_smoke.json` | Nonzero update, finite gradient, reload agreement and acceptable actual VRAM/time |
| Full teacher cache | `kd_cache/metadata.json` | `complete=true`, 288 default entries |
| Student training | `training_complete.json`, `training_log.jsonl` | 400 actual optimizer updates; validation tracked; model selection uses validation only |
| Model | `checkpoints/step_*/adapter.safetensors` | Committed metadata/checksums and a `best.json` pointer |
| Held-out losses | `evaluation/test/*_heldout_loss.json` | Unadapted and student, same fixed held-out draws |
| Qualitative outputs | `evaluation/test/comparison_manifest.jsonl` | Four fixed test contexts × three methods; inspect inverse action completion |
| Automatic metrics | `metrics_summary.json` | Report counts and caveats; do not equate warp/CLIP with direction success |
| Manual metrics | `human_review.csv`, `human_summary.json` | All ratings filled from visual inspection; success rate reported with sample counts |
| Interface | Notebook section 14 | Trained checkpoint loads, prompt and seed inputs produce/display direct student video |
| Submission | `export/wan_reverse_trained_adapter.zip`, `REPORT.md` | Source and pinned base identity, adapter, real metrics, videos and completed discussion |

If a stage fails, preserve the output and report the full error. Do not assume a source ZIP alone contains a trained model. If the student does not complete inverse actions, record that result and investigate data, conditioning, temporal sampling, objective and optimization rather than claiming completion from a loss curve.

## Report structure

1. Objective and video-time versus noise-time distinction.
2. Teacher choice: prior smaller/faster teacher tests, their visible action failures, and the successful Wan generation setting. Separate user observations from measured files.
3. Dataset generation, detailed prompt design, seeds, reversal, labeling, fixed splits and review assumptions.
4. Causal VAE preprocessing and the T4 resolution/frame compromise.
5. Proposed paired latent-transport KD derivation, exact flow objective, temporal auxiliary and conditioning dropout.
6. Student LoRA scope, frozen parameters, precision, gradient checkpointing and hardware smoke measurements.
7. Major experiments and ablations, costs and checkpoint selection.
8. Held-out quantitative results with limitations and qualitative examples including failures.
9. Interface implementation, dependencies, input flow and use instructions.
10. Reproducibility, submission artifacts, remaining limitations and future work.

The automatically generated `REPORT.md` starts this structure with real saved values. Finish the experimental interpretation and add images/videos yourself; do not fabricate missing metrics or claim a general result from four test videos.
