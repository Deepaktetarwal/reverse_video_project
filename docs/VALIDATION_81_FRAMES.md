# Validation record for the all-81-frame source package

Author runtime: CPU-only Python 3.12, PyTorch 2.5.1+cpu, Diffusers 0.35.1, Transformers 4.49.0, NumPy 1.26.4. Target Colab uses isolated Python 3.11 and the supplied dependency pins.

**26 tests passed** using `PYTHONPATH=wan_81_overlay python -m pytest -q wan_81_overlay/tests_refine`.

The 21 V3/compatibility tests include original-prompt routing, flow/hinge/motion math, real small Wan LoRA checkpointed backward/update, frozen-base behavior, serialization/resume including fresh-noise RNG and actual split training across save/resume. Five 81-frame checks add:

1. Every source index 0–80 is retained in temporal order; non-81-length inputs are rejected for this experiment.
2. Preprocessing signatures distinguish 17 and 81 frames, preventing reuse of old latent artifacts.
3. Partial preparation is allowed only for a smoke check, with train-split validation; actual training/evaluation requires complete preparation.
4. A real tiny WanPipeline samples 81 output-frame latents (21 temporal positions); a real rank-16 checkpointed backward pass on that shape has finite gradients and leaves the base frozen.
5. A real small AutoencoderKLWan encodes 81 frames to 21 latent positions and decodes back to 81 finite pixel frames.

The 26-cell notebook passes nbformat schema validation; all executable cells and Python sources parse. All 19 shipped CLI entrypoints pass argument-help loading without pretrained weights.

Not tested here: full pretrained 1.3B CUDA memory fit, actual 600-step T4 runtime, the user's complete dataset, quality improvement from 81 frames, or successful reverse actions. The included full-model smoke and evaluations must establish those empirically. No empirical GPU result or trained weight is supplied in the source ZIP.
