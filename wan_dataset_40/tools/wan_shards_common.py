"""Pure Python inventory, checksums, packaging and review utilities."""
import csv
import hashlib
import html
import json
import os
import zipfile
from pathlib import Path

SCHEMA = 'wan-reverse-pilot-v1'
MODEL = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
REVISION = '0fad780a534b6463e45facd96134c9f345acfa5b'
GENERATION = dict(height=480, width=832, frames=81, fps=16, steps=50,
                  guidance=6.0, flow_shift=8.0, transformer_precision='fp16',
                  vae_precision='fp32', text_encoder='nf4-fp32-512')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def shard_plan(plan, shard):
    events = ['fall', 'pour', 'ink', 'inflate']
    return sorted([r for r in plan if r['shard'] == shard], key=lambda r: ((events.index(r['event']) - (shard - 1)) % 4, r['context_index']))

def initialize(root, shard, plan):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = shard_plan(plan, shard)
    assert len(rows) == 10 and len({r['id'] for r in rows}) == 10
    config = dict(schema=SCHEMA, model=MODEL, revision=REVISION,
                  generation=GENERATION, shard=shard, prompts=rows)
    config['fingerprint'] = digest(config)
    existing = root / 'dataset_config.json'
    if existing.exists() and read_json(existing) != config:
        raise ValueError('Existing shard configuration differs. Use a new output root; do not mix runs.')
    atomic_json(existing, config)
    return config

def verify_record(root, record, config=None, check_hash=True):
    root = Path(root).resolve()
    if config:
        expected = {r['id']: r for r in config['prompts']}
        base = expected.get(record.get('id'))
        if not base:
            raise ValueError('Unknown candidate ID')
        for k, v in base.items():
            if record.get(k) != v:
                raise ValueError(f'Candidate label mismatch: {k}')
        if record.get('dataset_fingerprint') != config['fingerprint']:
            raise ValueError('Dataset fingerprint mismatch')
        if record.get('model') != config['model'] or record.get('revision') != config['revision']:
            raise ValueError('Teacher mismatch')
        if record.get('generation') != config['generation']:
            raise ValueError('Sampling configuration mismatch')
    required = {'forward', 'reverse', 'frames_uint8', 'forward_latent', 'embedding', 'negative_embedding'}
    if not required.issubset(record.get('assets', {})):
        raise ValueError('Required assets missing from record')
    for asset in record['assets'].values():
        rel = Path(asset['path'])
        path = (root / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Unsafe asset path')
        if not path.is_file() or path.stat().st_size != asset['bytes']:
            raise ValueError(f'Missing/truncated asset: {rel}')
        if check_hash and sha256(path) != asset['sha256']:
            raise ValueError(f'Checksum mismatch: {rel}')
    return record

def completed(root, config, check_hash=True):
    rows = []
    for spec in config['prompts']:
        p = Path(root) / 'metadata' / f"{spec['id']}.json"
        if p.exists():
            rows.append(verify_record(root, read_json(p), config, check_hash))
    return rows

def refresh(root, config):
    root = Path(root)
    rows = completed(root, config)
    tmp = root / 'manifest.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    os.replace(tmp, root / 'manifest.jsonl')
    review = root / 'review.csv'
    old = {}
    if review.exists():
        with review.open(newline='', encoding='utf-8') as f:
            old = {r['id']: r for r in csv.DictReader(f)}
    fields = ['id', 'keep', 'action_complete', 'object_consistent', 'reverse_clear', 'reason']
    tmp = root / 'review.csv.tmp'
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(old.get(row['id'], dict(id=row['id'], keep='', action_complete='', object_consistent='', reverse_clear='', reason='UNREVIEWED')))
    os.replace(tmp, review)
    cards = []
    for r in rows:
        cards.append(f"<article><h2>{html.escape(r['id'])} — {r['split']}</h2><p>{html.escape(r['prompt'])}</p><p><b>Reverse:</b> {html.escape(r['reverse_caption'])}</p><p><b>Accept only if:</b> {html.escape(r['acceptance'])}</p><video controls preload='none' width='420' src='{r['assets']['forward']['path']}'></video> <video controls preload='none' width='420' src='{r['assets']['reverse']['path']}'></video></article>")
    (root / 'review.html').write_text("<!doctype html><meta charset='utf-8'><title>Teacher candidate review</title><h1>Forward and reversed teacher candidates</h1><p>Edit review.csv; blank keep means NOT accepted. A reversed clip is an inspection baseline.</p>" + '\n'.join(cards), encoding='utf-8')
    return rows

def export_shard(root, destination):
    root = Path(root)
    config = read_json(root / 'dataset_config.json')
    rows = refresh(root, config)
    if not rows:
        raise ValueError('No completed candidates to export')
    files = {'dataset_config.json', 'manifest.jsonl', 'review.csv', 'review.html'}
    for r in rows:
        files.add(f"metadata/{r['id']}.json")
        files.update(a['path'] for a in r['assets'].values())
    checksums = {p: dict(sha256=sha256(root / p), bytes=(root / p).stat().st_size) for p in sorted(files)}
    summary = dict(schema=SCHEMA, shard=config['shard'], fingerprint=config['fingerprint'], completed=len(rows), expected=10, files=checksums)
    atomic_json(root / 'bundle_checksums.json', summary)
    files.add('bundle_checksums.json')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + '.tmp')
    # NPZ and MP4 are already compressed. DEFLATE mainly shrinks padded embeddings.
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel in sorted(files):
            z.write(root / rel, f"shard_{config['shard']:02d}/{rel}")
    os.replace(tmp, destination)
    return dict(path=str(destination), completed=len(rows), expected=10, bytes=destination.stat().st_size)
