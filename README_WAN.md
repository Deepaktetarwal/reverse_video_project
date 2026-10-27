# Correction: same prompt, reversed output (v2)

For the input **"A red ball falls off a table"**, the frozen teacher generates the fall and the trained student should directly generate the reverse event (the ball rising onto the table). **Both receive the same original prompt. No reversed prompt is required at inference.** Reverse captions remain evaluation labels only.

The earlier v1 training code conditioned the student on reverse captions. That was a task-design error relative to this requirement. This v2 fixes the training, sampling, evaluation and interface. It rejects v1 checkpoints. Do not resume a reverse-caption checkpoint for this task.

## Reuse work already completed

Extract this v2 source ZIP over the project code. Open **notebooks/Wan_Reverse_SamePrompt_Colab.ipynb**. It defaults to a new run folder, **runs/wan_same_prompt_v2/**, and includes an explicit migration cell that copies existing `prepared` and `kd_cache` files from **runs/wan_t4/**. The 40 videos, VAE endpoints and frozen-teacher predictions are unchanged and reusable, including completed entries from a partial cache. Stop any active old stage cleanly or let it finish before copying. Old training checkpoints and smoke results are not copied.

Run notebook setup sections 1–3, the cache migration cell, then the test/audit and GPU smoke stages. The prepare/cache stages verify and skip valid copied work. Rerun the GPU smoke because the train signature now includes the conditioning task. Finish any remaining cache entries and train a **fresh adapter** in the new run. No video regeneration is needed.

You can use the already installed `/content/wan_reverse_work/env` in the same Colab VM; setup reuses it. Do not run the old notebook's training or demo cells after updating the source.

# Complete Wan reverse-time pilot code: Google Colab

This code uses the **already imported Wan dataset shown in your screenshot**. It replaces the earlier Zeroscope training implementation. The generation notebooks in `wan_dataset_40` and the 40 teacher candidates remain separate inputs; they do not need to be regenerated.

This is a research implementation and a runnable workflow, not a guarantee that 32 training clips produce good reverse-time generation. Trained weights cannot be supplied before the code is run against your dataset on a GPU. This package creates and exports those weights when training completes.

## Put the files here

Extract `Wan_Reverse_SamePrompt_V2.zip` **directly into your existing `reverse_video_project` folder**. The archive has no extra enclosing project folder. Do not create a nested `reverse_video_project/reverse_video_project` folder.

| Path inside reverse_video_project | Purpose |
|---|---|
| `data/wan_pilot_v1/` | Your existing imported 40-candidate dataset; keep it |
| `incoming_wan_bundles/` | Your original transfer ZIPs; keep them as backups |
| `wan_dataset_40/` | Your existing generation/import package |
| `wan_reverse/` | New Wan training, caching, inference and evaluation modules |
| `configs/wan_t4.json` | Default T4 pilot settings |
| `notebooks/Wan_Reverse_SamePrompt_Colab.ipynb` | New step-by-step execution and demo interface |
| `requirements-wan.txt` | Pinned dependencies |
| `tests/` | Real tiny-Wan/LoRA/VAE CPU tests |
| `docs/` | Mathematics, stage checklist and report guidance |
| `runs/wan_same_prompt_v2/` | Created by the notebook: caches, checkpoints, videos, reports |

The notebook expects Drive path `/content/drive/MyDrive/reverse_video_project`. Change `SOURCE` in its first cell only if your real folder is elsewhere.

## Current model and training format

Base/frozen teacher: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, pinned to `0fad780a534b6463e45facd96134c9f345acfa5b`.

Student: the same pretrained architecture plus FP32 rank-8 LoRA adapters on attention query/key/value/output projections. No base weights are optimized. This is adaptation/distillation, not a compressed model or a few-step sampling model.

**T4 preset: 256 pixels high × 448 wide, 17 frames, display at 8 fps, 50 inference steps.** Training samples cover the entire original 81-frame action by uniform selection of 17 frames, then resize. The original stored 480×832/81-frame videos remain intact. This reduction is needed to give a T4 backward pass a realistic chance of fitting; full-model memory fit still needs the actual smoke test. It is a substantial change of training format and may reduce output quality. We do not claim the student retains the original 480p quality.

The chosen 17-frame display at 8 fps lasts 2.125 seconds and accelerates the sampled original action. It does not preserve the source video's native playback timing. Teacher reference, unadapted and student evaluation clips all use the same reduced comparison format.

Do **not** change training resolution or frame count mid-run. To try a larger setup, use a new `RUN_NAME`, edit the config before preparing latents, and rerun the real GPU smoke test. A possible higher-memory experiment is 320×576/33 frames; it is not validated on T4. The original 480p/81-frame setup is not promised to fit student training on a 16 GB GPU.

## Run in Colab, in order

1. Open `notebooks/Wan_Reverse_SamePrompt_Colab.ipynb`. Select **T4 GPU**. The notebook installs a separate Python 3.11 environment, regardless of the kernel's Python version. Use the provided dependency pins, not the old Zeroscope requirements.
2. Mount Drive and copy source code to local `/content`. The dataset stays in Drive. Set `RUN_NAME='wan_same_prompt_v2'` unless starting a separate experiment. Saved run configurations are reused, not silently overwritten.
3. Install dependencies and run the CPU tests. These use real tiny Wan/VAE models and do not download the full teacher. Review the config display before proceeding.
4. Run the dataset audit. The default explicitly honors your statement that all 40 imported candidates are correct (`accept_all_user_declared=true`), reads `manifest_wan.jsonl`, and writes that assumption to `audit.json`. It does not silently turn human review into an independent judgment. If you later want filtering, set this flag false and use the accepted manifest with an appropriate expected count.
5. Prepare the full dataset. Encode forward and reversed **pixels** separately with the causal Wan VAE. This is resumable per clip. Expect 40 prepared records.
6. Cache **one** training teacher target and run `gpu_smoke`. This performs a real forward/backward/optimizer step and a saved-adapter roundtrip. It writes actual peak VRAM and step timing. Stop on an out-of-memory, non-finite gradient or failed reload. The tiny CPU tests are not a substitute for this full-model GPU check.
7. Only after the smoke test passes, finish the full frozen-teacher cache. Expected entries: 256 train, 16 validation, 16 test. The teacher exits before student training starts.
8. Train. Default: 400 optimizer updates, batch size 1, accumulate 4, LR 1e-4, checkpoints and validation every 50 updates. Tensors are staged on local SSD to avoid repeatedly reading many GB from Drive. Checkpoints/logs remain on Drive. Training runtime is **unknown until measured**; use the smoke projection only as a rough compute estimate, excluding validation and I/O.
9. Inspect the saved log curve and best validation checkpoint. Best selection uses validation flow MSE; test data never enters optimizer updates or checkpoint selection.
10. Run fixed held-out loss evaluation for unadapted and student models, then generate all four test comparisons. These include the stored reversed teacher reference, unadapted direct original-prompt generation, and trained student direct generation. Sampling and VAE decode use separate processes.
11. Run automatic metrics and watch the videos. Fill `evaluation/test/human_review.csv`, then run the human summary. Leave no blank ratings in the final report. Low flow/warp loss alone is not proof of reverse-action success.
12. Run the report/export cells. Exported LoRA weights plus a pinned base revision reconstruct the finetuned model. Save the adapter ZIP, source code, configuration, logs, held-out videos and completed report for submission.
13. Use the final prompt/seed widgets to test the trained mechanism. Cached prompts avoid reloading UMT5. Arbitrary new prompts use a frozen quantized text encoder in a separate process; model download can be large and first use slow.

Do not run full training before the smoke test. Do not claim the experiment succeeds because a checkpoint exists. Check that the student visibly completes inverse events on the fixed held-out prompts.

## Method summary

For separately encoded paired endpoints zf and zr, use the same Gaussian noise ε and noise time s:

- xf=(1-s)zf+sε, xr=(1-s)zr+sε.
- Exact reverse flow target: ε-zr.
- Proposed transported frozen-teacher target: vT(xf,s,forward_caption)+zf-zr.
- Student conditions on the same original forward-action caption as the teacher.
- Default loss: exact flow MSE + 0.1 teacher-target MSE + 0.05 latent temporal-reconstruction MSE.

This transport follows from differentiating xr=xf+(1-s)(zr-zf). It is a pair-dependent modeling choice, not a global exact time-reversal of the teacher. The teacher's field is generally marginal rather than the exact sample's conditional flow, so KD can introduce bias. Run the optional flow-only ablation to measure whether KD helps. Full derivation and limitations are in `docs/MATHEMATICS_WAN.md`.

Training uses frozen text embeddings from your generation package. Student sampling directly denoises original-prompt-conditioned reverse-video latents and the decoder preserves their order. Only dataset preparation and the teacher comparison construct frame reversals.

## Resume without losing completed work

After a disconnected/replaced Colab runtime, rerun mount/copy, environment setup and config cells. Model weights and local staging may need downloading/copying again; completed prepared tensors, KD entries and committed training checkpoints are in Drive.

- `prepare` skips checksum-verified completed clips.
- `cache` skips checksum-verified completed draws.
- `train --resume auto` restores the most recent committed checkpoint, optimizer, scaler, RNG and sampler state. Work since the last committed checkpoint is repeated.
- `sample` skips an existing sample only when its identity and latent hash match.
- Changing data, dimensions, cache seeds or loss settings invalidates the corresponding signatures; start a new run rather than mixing artifacts.
- Do not run multiple training processes writing the same `run_dir`.

## Evaluation and reporting

Automatic outputs: held-out flow/KD/temporal losses, CLIP frame/text cosine, Farneback optical-flow magnitude and backward-flow warp L1, plus CSV/JSON summaries and a comparison gallery.

Manual outputs: inverse-action success (0/1), visual quality (1–5), temporal quality (1–5), and notes per video and method. The action success rate is the essential direction-sensitive performance metric. With four test clips it changes by 25 percentage points per clip; show counts, not only percentages.

CLIP is largely insensitive to frame order. Static clips can score well on warp error. Flow includes camera/lighting/occlusion effects. Do not treat these proxies as proof of physical correctness. Four test videos are insufficient for a credible FVD comparison, so no four-sample FVD is fabricated.

`report.py` builds a stage-wise Markdown report with existing artifact values and marks missing results as not run. Complete the discussion, add representative qualitative outputs, document failures and include the real hardware result. The report cannot decide whether a visually incorrect video is acceptable.

## Optional controlled ablation

A second run should use a new `run_dir`, the same data/preprocessing/cache seeds, `kd_weight=0` and `temporal_weight=0`. The notebook includes an optional cell to create that configuration. Identical prepared and teacher-cache directories can be copied to the new run because their signatures exclude loss weights; the new GPU smoke and training must still be run. Compare the same fixed test prompts and inference seeds. Do not overwrite the primary experiment or tune the test set.

## Validation status

The author environment has CPU only. Automated tests run against actual small `WanTransformer3DModel`, `WanPipeline` and `AutoencoderKLWan` instances at pinned Diffusers 0.35.1. They test checkpointed LoRA backward/update, frozen base weights, adapter roundtrip, tiny latent sampling, causal VAE reversal, normalization, transport algebra, safe checkpoint loading, preprocessing and motion proxies.

**The full 1.3B T4 backward pass, dataset-specific training and final output quality have not been executed here.** Your Colab GPU smoke and resulting evaluations provide that evidence. No trained model or performance numbers are invented in the package.
