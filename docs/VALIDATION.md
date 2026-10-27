# Validation record

Author environment: CPU-only PyTorch 2.5.1, Diffusers 0.35.1, Transformers 4.49.0. No full pretrained Wan weights, user dataset or CUDA device were used for the tests below.

Executed successfully:

1. Paired flow transport identity and numerical derivative agree with the exact reverse conditional velocity.
2. The temporal auxiliary and other losses vanish for an exact flow predictor.
3. A real tiny `WanTransformer3DModel` with non-reentrant activation checkpointing supports custom LoRA backward/update; base weights remain unchanged and adapter reload preserves predictions.
4. A real tiny `WanPipeline` with a flow UniPC scheduler produces finite latent video tensors.
5. A real small `AutoencoderKLWan` encodes/decodes correctly with normalization and confirms that encoding reversed pixels differs from flipping encoded forward latents.
6. Uniform frame selection preserves both original temporal endpoints and expected output dimensions.
7. Prompt-group split leakage and path traversal are rejected.
8. The optical-flow proxy detects translation; the static clip demonstrates why low warp error is not proof of quality.
9. Training checkpoints load with `weights_only=True`; sampler state and metadata roundtrip correctly.

Result for v2: **13 tests passed** (including regression checks for same-prompt routing, cache reuse/signature changes, and rejection of legacy checkpoints, and safe cache migration). All 15 command modules expose their argument help without loading GPU models. All Python source files parse. The 34-cell Colab notebook passes nbformat schema validation and each executable cell parses as Python.

Not established here:

- Full pretrained 1.3B CUDA backward fit on a Colab T4.
- Real training runtime or model convergence on the imported 40 videos.
- Visible direct reverse-action success or quality relative to unadapted/teacher outputs.
- New-prompt generalization, larger resolutions/frame counts, or reliable distribution metrics.

Run the included actual GPU smoke stage and held-out evaluation to obtain those results. The generated report must keep missing results marked as missing. The exported trained model is created only after a real training run, not supplied pre-trained in the source archive.

The user has already run a v1 full-model T4 backward smoke successfully. That measured execution fit for v1; v2 still requires a new smoke record and fresh training because its conditioning task has changed.
