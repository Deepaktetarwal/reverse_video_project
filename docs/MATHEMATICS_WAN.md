# Same-prompt task definition (v2)

The original prompt c describes a forward action, e.g. "a ball falls from a table". Both teacher and student receive c. The teacher distribution is forward-time video; the student target distribution is the frame-reversed teacher video conditioned on that same c. The interface does not rewrite c into a reverse-action description. A separate reverse caption is an evaluation annotation only and never enters the v2 student network.

Training pairs are `(c, R(X_teacher(c)))`. At inference the user supplies c, and the adapted student should directly produce the reversed evolution. This behavior must be learned and verified empirically; the code change alone does not establish that it has been learned.

# Mathematical method and its limits

This project uses **video-frame time** and **diffusion noise time**. Reversing video frames changes the event order. Denoising runs from high noise to low noise for both forward-video and reverse-video models. These are different operations.

## Pixel reversal and Wan's temporal VAE

For a teacher video X=(X0,...,X(F-1)), let R(X) reverse frame order. Uniformly select frames including both endpoints and resize them with P. With frozen causal VAE encoder E and per-channel normalization N, compute:

`zf = N(mode(E(P(X))))`

`zr = N(mode(E(R(P(X)))))`.

The posterior mode is used for reproducible cached endpoints. Wan's VAE compresses time and treats temporal boundaries causally. In general `E(R(X)) != R(E(X))`. Consequently, the code does not reverse the saved forward diffusion latent to obtain a target. The stored old 480p forward latent is not used as the reverse training endpoint.

The VAE's channel-wise normalization is `N(z)=(z-mean)/std`; decoding uses `N^-1(z)=z*std+mean`. The same pretrained normalization is applied to each separately encoded endpoint.

## Conditional flow matching

Draw Gaussian noise `ε ~ N(0,I)` and noise level `s` uniformly from `[0.02,0.98]`. In this implementation, `s=0` is clean and `s=1` is noise. Define a linear path:

`xr(s)=(1-s)zr+sε`.

Its exact conditional velocity is:

`d xr/ds = ε-zr`.

The original-prompt-conditioned reverse-time student predicts `vθ(xr(s),1000s,c)`. The exact flow loss is:

`Lflow = mean(||vθ-(ε-zr)||²)`.

The transformer sees the pretrained Wan timestep convention `1000*s`; inference uses the pretrained UniPC flow scheduler with flow shift 8. The training sigma distribution is an explicit simple design choice, not a claim to reproduce Wan's original large-scale training distribution exactly.

## Paired teacher transport distillation

A frozen teacher knows the original caption c and predicts a forward velocity. Use the **same Gaussian noise** for a second path:

`xf(s)=(1-s)zf+sε`.

For the paired endpoint difference `δ=zr-zf`:

`xr(s)=xf(s)+(1-s)δ`.

Differentiate with respect to noise time:

`d xr/ds = d xf/ds - δ`.

Thus the proposed transported teacher target is:

`vKD = stopgrad(vT(xf(s),1000s,c) + zf - zr)`.

If the teacher output equals the exact conditional forward velocity `ε-zf`, then `vKD=ε-zr`, which equals the reverse conditional target exactly. With a real teacher, the remaining teacher prediction residual is inherited by the reverse target. This is why the default KD weight is 0.1 alongside a weight-1 exact flow loss.

This transport is **pair-dependent translation in encoded latent space**. It is not an exact global conjugation of the pretrained teacher by pixel reversal, nor proof that the teacher estimates each sample's conditional endpoint velocity. A pretrained diffusion/flow teacher typically estimates a marginal conditional field. The transported residual can help or hurt; the KD-weight-zero ablation is needed to assess its effect. The method should be described as a proposed pilot distillation objective, not as an established published reverse-video algorithm.

The teacher is loaded only for an offline cache. Each training sample stores `(ε,s,vKD,conditioning-drop flag)` and a pointer to `(zr,c)`. The default has 32×8=256 training noise draws and 4×4=16 draws for each evaluation split. Frozen teacher outputs for val/test are generated from those held-out videos but never used for training or optimizer selection based on test outcomes. Validation flow loss alone selects the best checkpoint.

## Temporal auxiliary loss

Given predicted velocity, estimate the clean reverse endpoint:

`z0_hat = xr(s)-s*vθ`.

Let D be first difference along compressed latent-frame time. The auxiliary loss is:

`Ltemp = mean(||D(z0_hat)-D(zr)||²)`.

The total default objective is:

`L = 1.0 Lflow + 0.1 LKD + 0.05 Ltemp`.

This is a latent-space reconstruction regularizer, not a calibrated optical-flow/physical-consistency metric. It does not explicitly enforce reversed gravity, conservation of mass or exact pixel trajectories.

With probability 0.1 during training-cache construction, both teacher forward conditioning and student conditioning are replaced with the dataset's saved negative-prompt embedding. Other samples use the same original forward-action caption for both teacher and student. Inference uses that saved negative embedding for classifier-free guidance on cached examples. For arbitrary new UI text, the encoder uses the empty string as the negative prompt; this is a small conditioning difference that should be recorded when comparing custom demonstrations.

## Student parameterization

The student is the frozen Wan2.1 1.3B backbone plus trainable rank-8 LoRA matrices on self/cross-attention query, key, value and output linear projections:

`Wstudent(x)=Wbase*x + (alpha/rank) B A x`.

A is initialized with Kaiming uniform values; B is initialized to zero. Initially the adapter contributes zero. Base weights stay frozen. Adapter parameters and AdamW state use FP32; base forward computation uses FP16 in the T4 preset with loss scaling. Activation checkpointing trades additional compute for reduced backward memory. The student is not a smaller-parameter standalone network: inference still needs the base model and adapter. No inference-step speedup or compression is claimed.

Training uses batch size 1 and four accumulated microbatches per optimizer update. The default is 400 optimizer updates. This is a starting experiment, not an empirically established optimal duration. There are only 32 training clips; loss can decrease by memorization.

## Direct reverse-time inference

For the original forward-action prompt, the frozen text encoder returns c. The student is trained to interpret this input as a request for the time-reversed event. Sample Gaussian latent noise and denoise with the student and pretrained flow scheduler. Decode with the frozen Wan VAE. Student output frames are left in generated order. `sample.py` and `decode.py` do not reverse frames. Only `prepare.py` (training targets) and `eval_generate.py` (teacher reference) construct a reversal.

For held-out comparisons, use exactly the same inference seed and settings for unadapted direct generation and student direct generation. The existing teacher reference is a separately generated original video, then resized and uniformly temporally subsampled. Its noise trajectory is not matched to the smaller student run; pixelwise PSNR/SSIM is therefore not used as a generative-quality score.

## Evaluation interpretation

Report held-out flow MSE, CLIP text-image cosine averaged over frames, backward-flow warp L1, optical-flow magnitude, manual inverse-action success rate, visual quality and temporal quality ratings. The short action-first metric captions avoid truncating the key action out of long UMT5 captions. CLIP is a content/alignment proxy and cannot reliably identify time reversal. Optical flow/warp metrics must be interpreted together: static clips can have very low warp error. Human action success is essential.

Only four held-out videos are available. FVD computed on four clips would be unstable and is intentionally not presented as a credible population metric. The report must state sample counts and failure cases. Evaluate the same fixed test prompts/seeds without tuning them on results.

Recommended experiments, subject to compute: unadapted generator; flow-only fine-tuning (`kd_weight=0`, `temporal_weight=0`); full default objective; optionally KD+flow without the temporal auxiliary. Use a **new run_dir** for each changed loss setup; share the same preprocessing settings, fixed data split and evaluation seeds. A lower held-out flow loss does not by itself prove visible inverse motion.

## V2 evaluation semantics

Both unadapted and student models receive the same original forward-action prompt. The unadapted model is expected to retain its ordinary forward interpretation; the trained student should generate the inverse evolution. CLIP is evaluated against the inverse-event description as a target annotation, not against the literal forward input. Reverse descriptions are never fed to the v2 student. The teacher reference remains the stored forward teacher clip played in reverse.
