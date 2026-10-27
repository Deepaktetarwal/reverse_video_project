# Validation of the V3 source package

Author environment: CPU-only Python 3.12, PyTorch 2.5.1+cpu, Diffusers 0.35.1, Transformers 4.49.0, NumPy 1.26.4, tokenizers 0.21.0 and Pillow 10.4.0. The Colab notebook uses isolated Python 3.11 with the supplied dependency pins.

Executed: `PYTHONPATH=wan_refine_overlay python -m pytest -q wan_refine_overlay/tests_refine`

**21 tests passed.** Eight V3 checks cover exact reverse-velocity optima, penalties/gradient direction for forward reconstruction, numerical safety for nearly identical/static targets, fresh-noise RNG replay, real tiny-Wan rank-16 checkpointed backward/update with frozen base weights, checkpoint/optimizer/RNG restoration and signature rejection, external prepared-path handling, and an actual three-update trainer entrypoint split across save/resume. Thirteen compatibility checks cover the earlier real tiny-Wan/VAE/pipeline/serialization, flow algebra, motion proxy, split/path guards, original-prompt routing and cache migration.

The 34-cell notebook passes nbformat schema validation. All executable cells and Python sources parse. CLI argument/help checks pass for all 19 shipped command modules without downloading model weights.

No full pretrained 1.3B model, user training dataset or CUDA device was used in these checks. They establish component execution and intended data routing, not GPU fit, runtime, convergence or reverse-action success. The notebook includes the necessary real rank-16 T4 smoke and staged qualitative diagnostics. It deliberately asks the experimenter to inspect one-clip motion before launching the full run.

The uploaded V2 ink videos were inspected across their frame sequences. The teacher reference visibly withdraws/clears ink; the student visibly introduces/spreads ink. The screenshot's validation flow MSE stays near 0.20. This supports reporting V2 as unsuccessful, but it does not identify a unique cause or prove V3 will fix it.
