# The 81-frame experiment

This is a fresh temporal-representation experiment using the V3 architecture and objective. The accompanying `REFINEMENT_MATH_V3.md` explains the original 17-frame pilot. In THIS notebook, replace its temporal preprocessing P by spatial resize only: every source frame 0–80 is retained. This experiment does not change or fine-tune the pretrained VAE/text encoder.

For stored teacher video X with 81 frames and original forward-action text c:

zf = N(mode(E(resize(X))))
zr = N(mode(E(reverse_frames(resize(X))))).

The normalized causal Wan VAE compresses 81 pixel frames to 21 latent time positions. The source reversal occurs before encoding. Latent reversal is not substituted for pixel reversal.

Fresh Gaussian eps and sigma give xr=(1-sigma)zr+sigma*eps and exact conditional flow target eps-zr. The same original prompt embedding is supplied unchanged. The student is the frozen Wan1.3B transformer plus rank-16 LoRA attention projections. There is no new large backbone or extra trainable temporal network.

The objective remains:

L = Lflow + 0.1 Lmotion + 0.1 Lorder.

Lorder compares the predicted clean latent xr-sigma*v_theta to the paired reverse and forward endpoints with the normalized margin described in the V3 mathematics document. It is an experimental reconstruction-order preference, not a physical-motion guarantee. Default kd_weight=0 means teacher-sample distillation, not teacher-velocity/feature matching. The old 17-frame KD cache is not used.

The comparison aims to test whether denser temporal supervision improves generated action completion and consistency. It does not establish frame count as the cause of previous failures. Changing tensor shape also changes how a seeded random stream is consumed, so this is not an exact paired-noise optimization trajectory comparison.

The new display format is 81 frames at 16 fps. The earlier 17-frame clips displayed at 8 fps cover the same source action with different temporal density and playback duration. Report that distinction in any cross-run quality or optical-flow comparison. Within the new final test, teacher reference, unadapted model and student use the same 81-frame/256×448/16fps comparison format.

Fit and runtime require the full pretrained 1.3B GPU smoke. Passing CPU tiny-model tests does not establish T4 fit or reverse-video quality. If the smoke succeeds, 600 optimizer updates with accumulation 2 correspond to 1200 independently sampled training microbatches; they are not 600 epochs. Validation/test examples never enter those optimizer updates.
