"""CPU-only integrity tests with synthetic bytes, not generated videos."""
import ast
import csv
import json
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from wan_shards_common import initialize, atomic_json, sha256, refresh, export_shard, completed
from import_wan_shards import import_bundle, rebuild, checked_extract

PLAN = json.loads((ROOT / 'prompts_40.json').read_text())


def fake_clip(root, config, row):
    paths = dict(forward=f"forward/{row['id']}.mp4", reverse=f"reverse/{row['id']}.mp4",
                 frames_uint8=f"frames/{row['id']}.npz", forward_latent=f"latents/{row['id']}.safetensors",
                 embedding=f"embeddings/{row['id']}.safetensors", negative_embedding='embeddings/negative.safetensors')
    assets = {}
    for key, rel in paths.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((key if key == 'negative_embedding' else row['id'] + key).encode())
        assets[key] = dict(path=rel, bytes=path.stat().st_size, sha256=sha256(path))
    record = dict(**row, schema=config['schema'], dataset_fingerprint=config['fingerprint'],
                  model=config['model'], revision=config['revision'], generation=config['generation'], assets=assets)
    atomic_json(root / 'metadata' / f"{row['id']}.json", record)


class PackagingTests(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(len(PLAN), 40)
        self.assertEqual(len({r['id'] for r in PLAN}), 40)
        self.assertEqual(len({r['seed'] for r in PLAN}), 40)
        self.assertEqual(Counter(r['split'] for r in PLAN), {'train': 32, 'val': 4, 'test': 4})
        self.assertEqual(Counter(r['shard'] for r in PLAN), {1: 10, 2: 10, 3: 10, 4: 10})
        self.assertEqual(Counter(r['event'] for r in PLAN), {'fall': 10, 'pour': 10, 'ink': 10, 'inflate': 10})
        for r in PLAN:
            self.assertGreaterEqual(len(r['prompt'].split()), 120)
            self.assertGreater(len(r['reverse_caption'].split()), 60)
            self.assertNotEqual(r['prompt'], r['reverse_caption'])
            self.assertEqual(r['split'], 'train' if r['context_index'] < 8 else ('val' if r['context_index'] == 8 else 'test'))

    def test_all_shards_import_and_review(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); dest = td / 'central'
            for shard in range(1, 5):
                root = td / f'source_{shard}'
                c = initialize(root, shard, PLAN)
                for r in c['prompts']:
                    fake_clip(root, c, r)
                self.assertEqual(len(refresh(root, c)), 10)
                bundle = td / f'{shard}.zip'
                export_shard(root, bundle)
                self.assertEqual(import_bundle(bundle, dest, PLAN), 10)
                # Reimports must be idempotent.
                import_bundle(bundle, dest, PLAN)
            summary = rebuild(dest, PLAN)
            self.assertEqual(summary['completed_candidates'], 40)
            self.assertEqual(summary['accepted_candidates'], 0)
            self.assertEqual(summary['completed_splits'], {'train': 32, 'val': 4, 'test': 4})
            review = dest / 'shards/shard_01/review.csv'
            with review.open(newline='') as f:
                rows = list(csv.DictReader(f))
            rows[0].update(keep='1', action_complete='1', object_consistent='1', reverse_clear='1', reason='Reviewed fixture')
            with review.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            self.assertEqual(rebuild(dest, PLAN)['accepted_candidates'], 1)
            import_bundle(td / '1.zip', dest, PLAN)
            self.assertEqual(rebuild(dest, PLAN)['accepted_candidates'], 1)
            # Merely setting keep is insufficient.
            rows[1]['keep'] = '1'
            with review.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            self.assertEqual(rebuild(dest, PLAN)['accepted_candidates'], 1)

    def test_partial_then_complete_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); root = td / 'source'; dest = td / 'dest'
            c = initialize(root, 2, PLAN)
            fake_clip(root, c, c['prompts'][0]); refresh(root, c)
            export_shard(root, td / 'partial.zip')
            import_bundle(td / 'partial.zip', dest, PLAN)
            self.assertEqual(rebuild(dest, PLAN)['completed_candidates'], 1)
            for r in c['prompts'][1:]: fake_clip(root, c, r)
            export_shard(root, td / 'complete.zip')
            import_bundle(td / 'complete.zip', dest, PLAN)
            self.assertEqual(rebuild(dest, PLAN)['completed_candidates'], 10)
            (root / f"forward/{c['prompts'][0]['id']}.mp4").unlink()
            with self.assertRaises(ValueError): completed(root, c)

    def test_tamper_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); root = td / 'source'
            c = initialize(root, 1, PLAN)
            fake_clip(root, c, c['prompts'][0])
            p = root / f"forward/{c['prompts'][0]['id']}.mp4"
            p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): export_shard(root, td / 'bad.zip')
            z = td / 'traversal.zip'
            with zipfile.ZipFile(z, 'w') as archive:
                archive.writestr('../outside.txt', 'bad')
            with self.assertRaises(ValueError): checked_extract(z, td / 'extract')

    def test_config_guard_and_notebook_sources(self):
        with tempfile.TemporaryDirectory() as td:
            initialize(td, 1, PLAN)
            different = json.loads(json.dumps(PLAN))
            different[0]['seed'] += 1
            with self.assertRaises(ValueError): initialize(td, 1, different)
        for shard in range(1, 5):
            path = ROOT / 'notebooks' / f'Wan_50Steps_Shard_{shard:02d}.ipynb'
            n = json.loads(path.read_text())
            self.assertEqual(n['nbformat'], 4)
            self.assertEqual(len({c['id'] for c in n['cells']}), len(n['cells']))
            for cell in n['cells']:
                if cell['cell_type'] != 'code': continue
                source = ''.join(cell['source']); tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        for filename in ('wan_shards_worker.py', 'wan_shards_common.py', 'import_wan_shards.py'):
                            content = (ROOT / 'tools' / filename).read_text()
                            if node.value == content: break
                if "(WORK/'wan_shards_worker.py').write_text" in source:
                    for filename in ('wan_shards_worker.py', 'wan_shards_common.py', 'import_wan_shards.py'):
                        self.assertIn(repr((ROOT / 'tools' / filename).read_text()), source)
            first_code = next(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code')
            self.assertIn(f'SHARD={shard}', first_code)

if __name__ == '__main__':
    unittest.main(verbosity=2)
