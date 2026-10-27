"""GPU worker. Each stage is a process; do not launch two writers for one shard."""
import argparse
import json
import os
import time
from pathlib import Path

from wan_shards_common import read_json, atomic_json, sha256, refresh, completed, verify_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['encode', 'generate', 'status'])
    p.add_argument('--root', required=True)
    p.add_argument('--limit', type=int, default=1, help='Maximum newly completed clips in this call; <=0 means all remaining')
    args = p.parse_args()
    root = Path(args.root)
    c = read_json(root / 'dataset_config.json')
    if args.stage == 'status':
        rows = refresh(root, c)
        print(f"Verified complete: {len(rows)}/10", flush=True)
        return
    import torch
    from safetensors.torch import load_file, save_file
    from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler
    assert torch.cuda.is_available(), 'A GPU is required. Select a T4 runtime.'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Dataset:', c['fingerprint'], flush=True)
    print('Sampling:', c['generation'], flush=True)
    done = {r['id'] for r in completed(root, c)}
    specs = [r for r in c['prompts'] if r['id'] not in done]
    if args.limit > 0:
        specs = specs[:args.limit]
    if not specs:
        print('No new candidates requested; all verified completed clips are skipped.', flush=True)
        refresh(root, c)
        return
    kw = dict(revision=c['revision'], low_cpu_mem_usage=True, use_safetensors=True)
    embdir = root / 'embeddings'
    embdir.mkdir(exist_ok=True)
    negative_path = embdir / 'negative.safetensors'
    def save_tensors(path, values):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, str(tmp))
        os.replace(tmp, path)
    if args.stage == 'encode':
        pending = [r for r in specs if not (embdir / f"{r['id']}.safetensors").exists()]
        if not pending and negative_path.exists():
            print('Reusing all needed prompt embeddings.', flush=True)
            return
        from transformers import AutoTokenizer, UMT5EncoderModel, BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        print('Loading quantized UMT5. Model downloads may take time.', flush=True)
        encoder = UMT5EncoderModel.from_pretrained(c['model'], subfolder='text_encoder', quantization_config=quant, torch_dtype=torch.float32, device_map={'': 0}, **kw).eval()
        encoder.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(c['model'], subfolder='tokenizer', revision=c['revision'])
        pipe = WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=None, transformer=None)
        def encode(text):
            with torch.inference_mode():
                pos, _ = pipe.encode_prompt(prompt=text, do_classifier_free_guidance=False, max_sequence_length=512, device=torch.device('cuda'), dtype=torch.float32)
            if not torch.isfinite(pos).all():
                raise RuntimeError('Non-finite UMT5 embedding')
            return pos
        if not negative_path.exists():
            negatives = {r['negative_prompt'] for r in c['prompts']}
            assert len(negatives) == 1
            save_tensors(negative_path, {'negative_prompt_embeds': encode(next(iter(negatives)))})
        for r in pending:
            print('Encoding forward and reverse captions:', r['id'], flush=True)
            # The reverse caption is stored for future student conditioning, not used to generate teacher targets.
            forward = encode(r['prompt'])
            reverse = encode(r['reverse_caption'])
            save_tensors(embdir / f"{r['id']}.safetensors", {'forward_prompt_embeds': forward, 'reverse_prompt_embeds': reverse})
            del forward, reverse
        print('Encoding complete. Process exit releases the text encoder.', flush=True)
        return
    for r in specs:
        assert (embdir / f"{r['id']}.safetensors").exists(), f"Encode {r['id']} first"
    assert negative_path.exists()
    print('Loading Wan transformer and FP32 VAE; text encoder is not loaded.', flush=True)
    load_start = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(c['model'], subfolder='vae', torch_dtype=torch.float32, **kw)
    pipe = WanPipeline.from_pretrained(c['model'], vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.float16, **kw)
    pipe.transformer.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=c['generation']['flow_shift'])
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
    print('Model load seconds:', round(time.perf_counter() - load_start, 1), flush=True)
    negative = load_file(str(negative_path))['negative_prompt_embeds']
    import numpy as np
    from diffusers.utils import export_to_video
    g = c['generation']
    for r in specs:
        print('\nGenerating', r['id'], '|', r['split'], '| seed', r['seed'], flush=True)
        print(r['prompt'], flush=True)
        torch.manual_seed(r['seed'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        captured = {}
        def callback(pipeline, i, timestep, values):
            lat = values['latents']
            if not torch.isfinite(lat).all():
                raise RuntimeError('Non-finite denoising latents; stop and diagnose numerical precision before accepting this sample.')
            if i + 1 == g['steps']:
                captured['forward_latent'] = lat.detach().cpu().contiguous()
            if (i + 1) % 5 == 0 or i == 0:
                print(f"{r['id']} step {i+1}/{g['steps']}; elapsed {(time.perf_counter()-begin)/60:.1f} min", flush=True)
            return values
        embpath = embdir / f"{r['id']}.safetensors"
        emb = load_file(str(embpath))
        with torch.inference_mode():
            frames = pipe(prompt_embeds=emb['forward_prompt_embeds'].to('cuda', torch.float16), negative_prompt_embeds=negative.to('cuda', torch.float16), height=g['height'], width=g['width'], num_frames=g['frames'], num_inference_steps=g['steps'], guidance_scale=g['guidance'], generator=torch.Generator('cpu').manual_seed(r['seed']), callback_on_step_end=callback, output_type='np').frames[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated() / 2**30
        if frames.shape != (g['frames'], g['height'], g['width'], 3) or not np.isfinite(frames).all():
            raise RuntimeError('Unexpected/non-finite decoded frames')
        uint8_frames = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
        paths = dict(forward=f"forward/{r['id']}.mp4", reverse=f"reverse/{r['id']}.mp4", frames_uint8=f"frames/{r['id']}.npz", forward_latent=f"latents/{r['id']}.safetensors", embedding=f"embeddings/{r['id']}.safetensors", negative_embedding='embeddings/negative.safetensors')
        for rel in paths.values():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
        for key, pixels in [('forward', uint8_frames), ('reverse', uint8_frames[::-1].copy())]:
            dest = root / paths[key]
            tmp = dest.with_name(dest.stem + '.partial.mp4')
            export_to_video(pixels.astype(np.float32) / 255.0, str(tmp), fps=g['fps'])
            os.replace(tmp, dest)
        dest = root / paths['frames_uint8']
        tmp = dest.with_name(dest.name + '.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, frames=uint8_frames)
        os.replace(tmp, dest)
        save_tensors(root / paths['forward_latent'], captured)
        assets = {k: dict(path=v, sha256=sha256(root / v), bytes=(root / v).stat().st_size) for k, v in paths.items()}
        record = dict(**r, schema=c['schema'], status='completed_candidate', model=c['model'], revision=c['revision'], generation=g, dataset_fingerprint=c['fingerprint'], assets=assets, generation_and_decode_seconds=seconds, peak_gpu_gib=peak, gpu=torch.cuda.get_device_name(0), raw_frames_layout='THWC uint8', forward_latent_note='Normalized forward Wan diffusion latent; do not flip this tensor to construct reverse targets. Reverse decoded frames and re-encode with Wan VAE.', review_status='unreviewed', text_encoder_note='Frozen UMT5, NF4 4-bit weights and FP32 compute; both captions encoded separately.')
        # This JSON is the commit marker. Unfinished candidates have no metadata record and are regenerated.
        atomic_json(root / 'metadata' / f"{r['id']}.json", record)
        refresh(root, c)
        print(f"SAVED {r['id']} | {seconds/60:.1f} min | {peak:.2f} GiB | not yet quality-approved", flush=True)
        del frames, uint8_frames, captured, emb
        torch.cuda.empty_cache()
    print('This batch is complete. Export and review before training.', flush=True)

if __name__ == '__main__':
    main()
