# Mathematics, diagnosis and limitations

## 1. The task distribution

Let c be the original forward-action text, X~p_T(X|c) a frozen teacher sample, and R the pixel-frame reversal operator. The desired student distribution is the pushforward

p_S(Y|c) = R# p_T(X|c).

For c="ink spreads in water", the teacher shows spreading and the student should show contraction/withdrawal. The task intentionally changes how the student interprets c. There is no mathematical requirement to change c to a reverse caption. Conversely, merely retaining c does not force reverse motion: the network must learn the changed distribution.

Use the frozen Wan temporal VAE E and pretrained normalization N:

zf = N(E(P(X))), zr = N(E(R(P(X)))).

P selects 17 frames across the full original action and resizes to 256×448. E uses posterior mode. The causal VAE requires **separate pixel encoding**: flipping zf is not equivalent to encoding reversed pixels. Existing prepared targets already obey this rule. The new target-check stage decodes zf and zr as stored, without any output flip, to verify the target retains visible motion.

## 2. The prior objective

For shared Gaussian noise eps and sigma s in (0,1), define

xf=(1-s)zf+s eps, xr=(1-s)zr+s eps.

The exact conditional reverse-endpoint velocity is u_r=eps-zr. The V2 flow term was mean square error between v_theta(xr,1000s,c) and u_r. This is an ordinary conditional flow-matching construction, consistent with the general framework in Lipman et al., *Flow Matching for Generative Modeling*, https://arxiv.org/abs/2210.02747 .

In the ideal population setting, its minimizer is the conditional expectation E[eps-zr | xr,c,s]. With sufficient data/capacity and successful optimization/integration, this is the marginal vector field for the reversed-endpoint distribution. Finite LoRA training on 32 clips is not that ideal setting. The mathematical target is correct; convergence and generalization are empirical questions.

The V2 transported KD term used

v_KD = v_T(xf,1000s,c) + zf-zr.

This follows from differentiating xr=xf+(1-s)(zr-zf) along a **paired** path. If v_T=(eps-zf)+r, then v_KD=(eps-zr)+r: the teacher residual is inherited. The pretrained teacher estimates a marginal conditional field, not the exact conditional path for every selected sample. This pair-dependent translation is a proposed approximation, not an established global reverse-video distillation theorem. It can bias training. V3 switches it off initially to test the teacher-sample objective separately.

The old temporal auxiliary was

||D(xr-s v_theta)-D(zr)||² = s² ||D((eps-zr)-v_theta)||²,

where D is first difference in latent frame time. It reweights reconstruction errors; it does not explicitly compare forward and reverse event order. A lower value cannot prove the generated action has reversed. The nearly flat uploaded validation curve and visibly forward student video show that V2 did not achieve its intended behavior, but they do not prove a particular implementation bug or isolate a unique cause.

## 3. V3 fresh flow matching

Every microbatch draws a fresh eps~N(0,I) and s~Uniform(sigma_min,sigma_max), with the old bounds 0.02 and 0.98. It uses the original frozen prompt embedding c. Conditional dropout uses the saved negative embedding, with probability 0 for the one-clip diagnostic and the old probability 0.1 for full training.

L_flow = mean((v_theta(xr,1000s,c) - (eps-zr))²).

No scheduler, noise-time sign or frame order is secretly reversed. Sampling still integrates the learned flow from high noise to low noise using the pinned Wan UniPC scheduler. Video-frame time and diffusion noise time are different quantities.

## 4. Proposed explicit endpoint-order surrogate

Define the predicted clean endpoint:

z_hat = xr - s v_theta.

e_r = mean((z_hat-zr)²), e_f = mean((z_hat-zf)²), d = mean((zr-zf)²).

Use the hinge

L_order = I * max(0, m*d + e_r - e_f) / max(d,tau).

Default m=0.2, tau=0.01. I is 1 only when 0.2<=s<=0.85 and d>=1e-4. This avoids imposing the preference on almost identical pairs and very extreme noise levels.

If z_hat=zr, then e_r=0 and e_f=d, so the hinge is zero for m<1. If z_hat=zf, the hinge is (m+1)*d/max(d,tau), which is positive for a distinguishable pair. The margin is scaled by d before numerical flooring; this ensures a correct reverse endpoint is not falsely penalized for a small d.

This is a proposed reconstruction-ranking auxiliary, **not Diffusion-DPO**, a likelihood-ratio objective or a physical time-arrow guarantee. Forward/reverse causal latents can differ in appearance and temporal boundary features. The hinge can exploit those differences without learning the intended action. Free-generation inspection remains essential.

## 5. Normalized latent motion term

L_motion = mean((D(z_hat)-D(zr))²) / max(mean(D(zr)²),0.05).

The denominator increases the relative influence of temporal changes when static appearance dominates the ordinary MSE, while its floor prevents nearly static clips from creating an enormous normalization factor. This remains reconstruction supervision, not optical-flow ground truth or a force-law constraint.

Default V3 total:

L = L_flow + 0.1 L_motion + 0.1 L_order.

These weights are trial choices. The same stochastic text/noise inputs used for the flow term produce the auxiliary estimates; no VAE gradients are needed during training.

## 6. Optional teacher-velocity KD ablation

A separate matched full run can add the old immutable transported KD cache. Every K=4 optimizer updates, one additional cached microbatch contributes

K * lambda_KD * mean((v_theta(xr_cached,s_cached,c)-v_KD_cached)²),

with lambda_KD=0.05. Its expected average contribution per update is lambda_KD under the sampled cached distribution, though its variance and interaction with optimization differ from an every-step term. This branch retains the approximation limitations above. It is not the source of fresh-noise targets. Compare it with flow-only using validation; do not assume KD helps.

## 7. Architecture, stages and metrics

The frozen Wan2.1 1.3B transformer plus rank-16 LoRA remains the architecture. Frozen text encoder/VAE are reused. FP32 LoRA masters and AdamW state, FP16 base computation, gradient scaling and activation checkpointing retain the earlier T4 design. Rank 16 and higher learning rates require a new real GPU smoke test.

One-clip overfitting uses one train ID, accumulation 1, LR 3e-4, 100 then 200 updates and CFG=1. It is a learnability diagnostic, not a test result. Full training starts fresh, uses all 32 train IDs, accumulation 2, LR 2e-4 and a trial ceiling of 1200 updates. Checkpoints every 25 updates reduce work lost on disconnect. Validation draws use independent fixed seeds. Resume restores the fresh-noise RNG as well as optimizer/scaler/sampler state.

The adapter effect diagnostic is RMS(v_adapter-v_base)/max(RMS(v_base),1e-6) on the same noised target and prompt. It verifies effect size, not correctness. The reverse-endpoint-closer fraction counts active draws with e_r<e_f; it is a reconstruction-order proxy, not free-generation success. It may be high even if the model generates forward videos from pure noise.

Use actual human inverse-action success on generated videos as the essential direction-sensitive metric. CLIP frame-average alignment is largely insensitive to ordering; static videos can have low warp errors. Report counts, failure examples, compute/settings and the previous V2 failure. The fixed four test contexts were already viewed in V2, so do not overstate their independence or generalization strength.

If the reverse VAE reconstruction is visibly wrong, investigate the representation/targets first. If the single-clip model fails, diagnose its logged adapter effect, gradients, sigma-dependent reconstruction and latent motion before a larger run. If it succeeds only on that clip but fails held-out validation, investigate data diversity/generalization; increasing steps alone can worsen memorization.
