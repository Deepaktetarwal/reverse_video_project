# All-81-frame Wan experiment, 600 optimizer updates

Open **notebooks/Wan_81Frames_600Steps_Colab.ipynb** from the supplied source package.

## Your two Drive trees

The destination Drive already has `data/wan_pilot_v1/` and the generation/import manifests. That is enough to start this fresh experiment. You do NOT need to copy `runs/wan_v3_full`, `runs/wan_v3_overfit` or checkpoint 600 from the original Drive.

1. On the destination Drive, extract **Wan_81Frames_600Steps_Code.zip** directly into `My Drive/reverse_video_project`. Replace matching source files. Do not delete existing data or run folders. There is no extra enclosing directory in this ZIP.
2. Open the NEW notebook in its `notebooks/` folder using Colab. Mount the intended destination Drive. Authorization is account-specific; putting a notebook in a Drive does not silently mount or transfer data from another account.
3. Choose T4 GPU and execute notebook sections in order. The old 17-frame notebooks should not be used to create or train this experiment.
4. The new run saves everything in `runs/wan_v3_81f_600/`. It starts fresh rank-16 adapters. Old 17-frame optimizer/checkpoint state is not resumed.

All required source modules, tests, dependency pins and config template are in this ZIP. Model weights download from Hugging Face as needed; the ZIP does not include a trained model or pretrained base weights. No teacher-video regeneration is required.

## Exact settings

| Setting | Value |
|---|---|
| Frozen teacher/base | Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| Revision | 0fad780a534b6463e45facd96134c9f345acfa5b |
| Stored source clips | Existing 40, each exactly 81 frames |
| Split | Fixed 32 train / 4 validation / 4 test |
| Training/generation frame count | **81, all source indices 0–80** |
| Spatial resolution | **256 high × 448 wide**, resized from the original source |
| Playback | 16 fps, approximately 5.06 seconds |
| Trainable network | Rank-16, alpha-16 attention LoRA; frozen base/VAE/text encoder |
| Precision | FP16 base computation, FP32 LoRA masters/AdamW state |
| LR / weight decay | 2e-4 / 0.01 |
| Batch / accumulation | 1 / 2 |
| Maximum optimizer updates | **600 total**, not 600 additional updates |
| Checkpoint interval | 25 updates, plus block endpoints/best validations |
| Validation interval / draws | 50 updates / 8 fixed draws per validation clip |
| Sampling | CFG=1, 50 UniPC steps, flow shift 8 |
| Teacher-velocity KD | 0 in this experiment |
| Objective | Fresh flow matching + 0.1 normalized latent-motion + 0.1 endpoint-order hinge |

These settings retain the earlier V3 learning design while testing denser temporal representation. All 81 frames does NOT mean original 480×832 spatial-resolution training. This experiment is teacher-sample distillation: frozen teacher videos reversed as targets. Do not describe it as teacher-velocity/feature KD. The earlier optional transported-KD algorithm remains a separate experiment and its 17-frame cache is incompatible with these latents.

## Cell order and outputs

1. Mount Drive/copy code to `/content`.
2. Create or reuse isolated Python 3.11, install pins and run component tests.
3. Create/reuse new config and audit the destination dataset/checksums.
4. Encode ONE training clip at 81 frames. Run a real full-model CUDA forward/backward/update/reload smoke. It writes `gpu_smoke_v3.json`, including measured peak VRAM and a rough compute estimate.
5. Only after the smoke succeeds, finish all 40 separately encoded forward/reversed VAE endpoints. All 81 pixel frames remain represented; the causal VAE compresses them to 21 latent time positions.
6. Inspect ink and balloon forward/reverse VAE reconstructions. Incorrect targets are a reason to stop before training.
7. Train to 600 updates, automatically resuming this NEW run if disconnected. The first attempt starts at update 1. Optional blocks are 200, 400 and 600 with the same total ceiling.
8. Plot actual flow and adapter-effect diagnostics.
9. Generate four validation prompts × seeds 2026/2027 at checkpoint 600 and CFG=1. Preserve failures.
10. Generate four test comparisons (teacher reversed, unadapted, student), compute automatic proxies and display all 12 test videos.
11. Fill `evaluation/test/human_review.csv`, then summarize real ratings, generate a stage report and export the trained adapter.
12. Run the demo widgets. Original prompt + seed → direct model output. Optional unadapted comparison. Cached prompts reuse embeddings; new text is passed unchanged to the frozen text encoder.

The final checkpoint is explicitly pinned to `checkpoints/step_000600`, not the moving latest pointer or the best-flow pointer. Lower reconstruction loss is not equivalent to better reverse-action generation.

## GPU feasibility and failure handling

The previous 17-frame T4 smoke does not establish 81-frame fit. At unchanged spatial size, latent time length increases from 5 to 21 positions. Attention cost and activation memory can grow substantially; runtime cannot be predicted reliably from frame-count ratio alone. The notebook measures one real 81-frame training update BEFORE the full preparation/training job. Its runtime projection excludes validation, model loads, transfers and I/O.

If the smoke fails with OOM, do not run later cells. No silent frame/resolution reduction is implemented. Keeping all 81 frames with a smaller explicit spatial preset or using a higher-memory GPU is a separate experiment. A new run name/config, new latents and new smoke would be required. Memory fit does not imply a free Colab session will last long enough for all 600 updates. Checkpointing makes interrupted work resumable.

Increasing frames is an experiment, not a guarantee that balloons will deflate or object geometry will remain consistent. Retain the earlier 17-frame results as a baseline.

## Resume and transfer

On a replacement runtime, rerun sections 1–3. Repeated audit/model downloads can take time; saved tensors/checkpoints are retained on Drive. Continue the unfinished preparation/training cell. Existing committed work is verified and reused. To use evaluation/demo after reconnecting, restore `rows` and `FINAL_CHECKPOINT` using the last notebook section; no training repeat is necessary after step 600 is committed.

The new run is independent of the old runs. If moving this new experiment later, copy its entire folder, including `prepared`, saved configs, pointer JSONs, checkpoints and other outputs, plus the same dataset and code. Keep the standard `MyDrive/reverse_video_project` path. Allow Drive syncing to complete first. No automatic cross-account synchronization is performed.

## Evaluation and report

Use actual completed inverse action (0/1), visual quality (1–5), temporal quality (1–5) and notes for all 12 test comparison rows. Do not count only selected ink demos. CLIP frame-average alignment and optical-flow warp are proxies; neither proves temporal reversal. The order hinge can exploit appearance/boundary differences rather than action direction. Test contexts were inspected in earlier trials, so acknowledge that history.

The 81-frame teacher reference/unadapted/student comparison uses the same frame count, resolution and fps. For cross-experiment comparison with 17 frames at 8 fps, record the different temporal density and playback duration. Do not interpret raw optical-flow magnitudes at different frame spacing as a controlled motion-quality improvement.

`REPORT_V3.md` collects the new run's actual config/artifacts; add your discussion, generation history, all major trials, measured compute and representative successes/failures. Keep missing results marked as missing. See `docs/REFINEMENT_MATH_V3.md` for the objective derivation and limitations.
