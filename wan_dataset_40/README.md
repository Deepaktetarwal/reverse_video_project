# Wan teacher pilot: four portable shards, 40 candidates

The package contains four self-contained Colab notebooks, a fixed detailed prompt plan, transfer verification tools, and CPU-only integrity tests. Use authorized GPU resources; it does not implement account rotation. Colab prohibits multiple accounts used to work around access/resource restrictions: https://research.google.com/colaboratory/faq.html#usage-limits .

## What you will generate

Teacher: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, pinned to commit `0fad780a534b6463e45facd96134c9f345acfa5b`.

Sampling: **480 high × 832 wide, 81 frames, 16 fps, 50 steps, UniPC shift 8, guidance 6**. Transformer FP16; VAE FP32; frozen text encoder NF4 with FP32 compute. The inference and memory settings follow the earlier successful T4 benchmark. The newly expanded prompts have not yet been generated or quality-validated.

| Notebook | Candidates | Train / val / test | First action to inspect |
|---|---:|---|---|
| `notebooks/Wan_50Steps_Shard_01.ipynb` | 10 | 8 / 1 / 1 | Falling |
| `notebooks/Wan_50Steps_Shard_02.ipynb` | 10 | 8 / 1 / 1 | Pouring |
| `notebooks/Wan_50Steps_Shard_03.ipynb` | 10 | 8 / 1 / 1 | Ink dispersion |
| `notebooks/Wan_50Steps_Shard_04.ipynb` | 10 | 8 / 1 / 1 | Balloon inflation |
| Total | 40 | 32 / 4 / 4 | Four event families |

There are ten contexts per event. Eight per event are train, one val and one test, assigned before generation. These are held-out contexts within four known event families, not held-out action categories. Four val/test samples each are too small for confident distribution metrics such as FVD; report that limitation. This is a pilot, not a sufficient final training dataset or a validated performance claim.

Forward prompts contain 125–146 words and describe the scene, object identity, full framing, starting state, visible direction of change and ending state. Reverse captions describe the inverse event with a matching static scene. Prompts cannot guarantee correct motion. The teacher always receives the **forward prompt**. Training targets are the decoded frames in reversed order; reverse captions are also encoded and saved for future student conditioning.

At the previously measured 45.3 minutes/video, plan roughly **30 GPU-hours total**, or 7.5 hours/shard, excluding setup, encoding, transfer and rejected/replacement clips. These times are estimates, not guarantees. If only 20 candidates pass review, you have 20 usable examples, not 40. Do not relabel failed actions or automatically include them to reach a quota.

## Execute one notebook

1. Upload/open the assigned notebook in Colab. Select a **T4 GPU** runtime. Do not run multiple writers for one shard folder.
2. Run sections 1–4 in order. A separate Python 3.11 environment is installed automatically; a newer Colab kernel is fine. GPU packages are isolated from the notebook kernel. Keep about 35 GiB free local disk for first setup; cached restarts need less, but incomplete caches may still require downloads.
3. Section 5 encodes the next unfinished candidate. Section 6 generates it at full quality.
4. Inspect the sample in section 7. Each shard starts with a different event. Reject a sample with incomplete/wrong motion even if it looks sharp.
5. If appropriate, set `APPROVE_REMAINING=True` in section 8 and run it. The default permits up to nine additional candidates. Set `NEW_CLIPS_THIS_RUN=1` or `2` for smaller batches. Rerunning section 8 continues from completed clips rather than duplicating them.
6. Review all completed candidates in section 9. Edit the shard's `review.csv`: set `keep`, `action_complete`, `object_consistent`, and `reverse_clear` to 1 only for acceptable clips. Blank means not accepted. Add rejection reasons.
7. Section 10 exports the shard ZIP. You can export incomplete shards; its summary states how many candidates are actually complete. Re-export after completion or changed review decisions.

If a runtime is lost, rerun setup sections 1–4, then encode/generate the next unfinished work. Completed metadata and checksums in Drive identify finished clips. The model cache and installed environment are temporary under `/content`. Interrupted generation resumes only at the **clip level**; the unfinished clip restarts. No unattended keep-alive or quota-bypass mechanisms are included.

## Files saved per shard

Drive root: `MyDrive/reverse_video_wan_shards_v1/shard_01/` (or 02–04).

| Relative path | Purpose |
|---|---|
| `dataset_config.json` | Frozen sampling settings, teacher revision and assigned prompt plan |
| `forward/<id>.mp4` | Teacher forward video |
| `reverse/<id>.mp4` | Exact frame-order reversal for inspection, encoded separately as MP4 |
| `frames/<id>.npz` | Losslessly stored uint8 decoded forward frames, key `frames`, layout THWC |
| `latents/<id>.safetensors` | Final normalized forward diffusion latent, key `forward_latent` |
| `embeddings/<id>.safetensors` | `forward_prompt_embeds` and `reverse_prompt_embeds` |
| `embeddings/negative.safetensors` | Shared `negative_prompt_embeds` |
| `metadata/<id>.json` | Caption labels, split, seed, teacher, settings, runtime and asset hashes |
| `manifest.jsonl` | Verified completed candidates only |
| `review.csv` and `review.html` | Human quality decisions and offline viewing |
| `bundle_checksums.json` | Transfer inventory |

The metadata JSON is written last as a commit marker. A candidate is skipped on rerun only after its files and checksums verify. Corruption causes an explicit error; it is not silently accepted or auto-regenerated at extra GPU cost.

The final diffusion latent is saved for provenance/debugging. **Do not flip its temporal axis to build a reverse Wan training target.** Wan's VAE compresses time, so encoding reversed pixel frames is generally not the same as reversing forward latents. For future training, take the saved pixel array, reverse its frame order, and re-encode using the Wan VAE with an explicitly designed flow-matching/distillation objective.

## Transfer to reverse_video_project

Each export is named `wan_pilot_v1_shard_01.zip` through `04.zip`, saved under the Drive root's `transfer_bundles/`. Transfer these four ZIPs to the central machine/Drive. Keep the ZIPs intact: copying only MP4s loses labels and embeddings.

Extract this code package on the central machine. Open a terminal in the extracted `wan_dataset_40` folder. The importer needs Python 3.10+ only; it does not require PyTorch or a GPU.

```bash
python tools/import_wan_shards.py --project "/path/to/reverse_video_project" --bundles "/path/to/wan_pilot_v1_shard_01.zip" "/path/to/wan_pilot_v1_shard_02.zip" "/path/to/wan_pilot_v1_shard_03.zip" "/path/to/wan_pilot_v1_shard_04.zip" --require-complete
```

Replace the quoted paths with real Windows or Linux paths. To import a partial shard, pass just that ZIP and omit `--require-complete`. To import in Colab on the main Drive, put this code package and transfer ZIPs there, mount Drive, and run the same command using the corresponding `/content/drive/MyDrive/...` paths.

The importer verifies archive paths, file inventory, checksums, exact prompt labels, teacher revision and settings before merging candidates. Conflicting existing immutable files are not overwritten. Local review decisions are preserved by default; `--replace-review` explicitly imports changed decisions from a bundle.

Destination: **`reverse_video_project/data/wan_pilot_v1/`**.

| Destination file | Meaning |
|---|---|
| `shards/shard_01/` through `shard_04/` | All assets with distinct IDs |
| `manifest_wan.jsonl` | All completed candidates with correct relative asset paths |
| `accepted_manifest_wan.jsonl` | Only explicitly accepted candidates |
| `dataset_summary.json` | Counts, accepted counts and split totals |
| `review_all.html` | Combined offline review gallery |

Open `review_all.html` locally. Edit each shard's `review.csv`, then rebuild accepted counts:

```bash
python tools/import_wan_shards.py --project "/path/to/reverse_video_project" --refresh-only
```

A row is accepted only when all four decision columns (`keep`, `action_complete`, `object_consistent`, `reverse_clear`) are affirmative. A keep decision alone is insufficient. Never select or tune prompts based on test results and then present those test samples as untouched evaluation.

## Important training boundary

**The earlier reverse_video_project is a Zeroscope implementation. These files do not turn it into a Wan trainer.** They are deliberately stored in a separate namespace with a different manifest name. Do not run the existing `reverse_video.cache` or `reverse_video.train` against these assets and do not change only `model_id`.

A Wan migration still needs Wan-specific model loading, temporal VAE encoding of reversed pixels, flow targets/KD design, adapter selection, backward-pass memory test, training/evaluation and UI integration. The new manifest is a clean input to that migration, not a claim that it is already complete. The importer does not modify source code or old Zeroscope datasets.

## Validation

Run:

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic file bytes. They check 40 unique IDs/seeds, 10 per shard, the 32/4/4 split, detailed prompts, four notebook sources, immutable config guards, full/partial imports, idempotency, review preservation, checksum failures and unsafe ZIP paths. They do not prove generated action correctness or GPU execution.

No full four-shard inference or newly detailed prompt sweep was executed by the author. The previous user-run 50-step benchmark is the available hardware/quality evidence. Inspect each generated action before accepting it.
