"""Import transfer ZIPs into a separate Wan dataset namespace. No ML packages required."""
import argparse
import csv
import html
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from wan_shards_common import (SCHEMA, MODEL, REVISION, GENERATION, atomic_json,
                               read_json, sha256, initialize, completed, refresh, shard_plan)


def checked_extract(bundle, destination):
    with zipfile.ZipFile(bundle) as z:
        if sum(i.file_size for i in z.infolist()) > 12 * 2**30:
            raise ValueError('Archive exceeds 12 GiB extraction limit')
        seen = set()
        tops = set()
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Unsafe archive path')
            if info.filename in seen or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Duplicate entry or symlink in archive')
            seen.add(info.filename)
            if p.parts:
                tops.add(p.parts[0])
        if len(tops) != 1 or next(iter(tops)) not in {f'shard_{i:02d}' for i in range(1, 5)}:
            raise ValueError('Expected exactly one numbered shard per archive')
        z.extractall(destination)
    return Path(destination) / next(iter(tops))


def verify_bundle(source, plan):
    summary = read_json(source / 'bundle_checksums.json')
    if summary['schema'] != SCHEMA:
        raise ValueError('Unknown archive schema')
    for rel, expected in summary['files'].items():
        path = (source / rel).resolve()
        if not path.is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError('Invalid inventory path')
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Archive integrity failure: {rel}')
    c = read_json(source / 'dataset_config.json')
    if c['shard'] != summary['shard'] or source.name != f"shard_{c['shard']:02d}":
        raise ValueError('Shard identity mismatch')
    if c['model'] != MODEL or c['revision'] != REVISION or c['generation'] != GENERATION:
        raise ValueError('Incompatible teacher or sampling configuration')
    # Rebuild exact expected config from the distributed master plan.
    expected_rows = shard_plan(plan, c['shard'])
    if c['prompts'] != expected_rows or len(expected_rows) != 10:
        raise ValueError('Prompt plan mismatch')
    from wan_shards_common import digest
    without = {k: v for k, v in c.items() if k != 'fingerprint'}
    if digest(without) != c['fingerprint'] or summary['fingerprint'] != c['fingerprint']:
        raise ValueError('Config fingerprint mismatch')
    rows = completed(source, c)
    manifest = [json.loads(s) for s in (source / 'manifest.jsonl').read_text().splitlines() if s.strip()]
    if manifest != rows or len(rows) != summary['completed']:
        raise ValueError('Manifest/completed-record mismatch')
    expected_files = {'dataset_config.json', 'manifest.jsonl', 'review.csv', 'review.html'}
    for r in rows:
        expected_files.add(f"metadata/{r['id']}.json")
        expected_files.update(a['path'] for a in r['assets'].values())
    if set(summary['files']) != expected_files:
        raise ValueError('Incomplete or unexpected inventory')
    return c, rows


def copy_immutable(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f'Existing file conflicts; nothing is overwritten: {destination}')
        return
    tmp = destination.with_name(destination.name + '.importing')
    shutil.copy2(source, tmp)
    tmp.replace(destination)


def import_bundle(bundle, target, plan, replace_review=False):
    with tempfile.TemporaryDirectory() as d:
        source = checked_extract(bundle, d)
        config, rows = verify_bundle(source, plan)
        destination = Path(target) / 'shards' / source.name
        initialize(destination, config['shard'], plan)
        # Verify all existing immutable paths before starting copies.
        paths = {'dataset_config.json'}
        for r in rows:
            paths.add(f"metadata/{r['id']}.json")
            paths.update(a['path'] for a in r['assets'].values())
        for rel in paths:
            dest = destination / rel
            if dest.exists() and sha256(source / rel) != sha256(dest):
                raise ValueError(f'Conflicting immutable artifact: {dest}')
        for rel in sorted(paths):
            copy_immutable(source / rel, destination / rel)
        local_review = destination / 'review.csv'
        previous = {}
        if local_review.exists():
            with local_review.open(newline='', encoding='utf-8') as f:
                previous = {r['id']: r for r in csv.DictReader(f)}
        with (source / 'review.csv').open(newline='', encoding='utf-8') as f:
            incoming = list(csv.DictReader(f))
        valid_ids = {r['id'] for r in rows}
        if len({r['id'] for r in incoming}) != len(incoming) or {r['id'] for r in incoming} != valid_ids:
            raise ValueError('Review IDs do not match completed candidates')
        for r in incoming:
            if replace_review or r['id'] not in previous:
                previous[r['id']] = r
        fields = ['id', 'keep', 'action_complete', 'object_consistent', 'reverse_clear', 'reason']
        with local_review.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            w.writerows(previous.values())
        refresh(destination, config)
        return len(rows)


def rebuild(target, plan):
    target = Path(target)
    merged, accepted, cards = [], [], []
    expected = {r['id']: r for r in plan}
    yes = lambda value: str(value).strip().lower() in ('1', 'true', 'yes')
    for shard in range(1, 5):
        root = target / 'shards' / f'shard_{shard:02d}'
        if not (root / 'dataset_config.json').exists():
            continue
        c = read_json(root / 'dataset_config.json')
        initialize(root, shard, plan)
        rows = refresh(root, c)
        with (root / 'review.csv').open(newline='', encoding='utf-8') as f:
            decisions = {r['id']: r for r in csv.DictReader(f)}
        for row in rows:
            if row['id'] not in expected or any(row[k] != expected[row['id']][k] for k in expected[row['id']]):
                raise ValueError('Unexpected candidate labels')
            r = dict(row)
            r['assets'] = {k: dict(a, path=f"shards/shard_{shard:02d}/{a['path']}") for k, a in row['assets'].items()}
            decision = decisions.get(r['id'], {})
            r['review'] = decision
            passes = all(yes(decision.get(k, '')) for k in ('keep', 'action_complete', 'object_consistent', 'reverse_clear'))
            r['review_status'] = 'accepted' if passes else 'not_accepted'
            merged.append(r)
            if passes:
                accepted.append(r)
            review_rel = f'shards/shard_{shard:02d}/review.csv'
            cards.append(f"<article><h2>{html.escape(r['id'])} — {r['split']} — {r['review_status']}</h2><p>{html.escape(r['prompt'])}</p><p>Accept only if: {html.escape(r['acceptance'])}</p><video controls preload='none' width='420' src='{r['assets']['forward']['path']}'></video> <video controls preload='none' width='420' src='{r['assets']['reverse']['path']}'></video><p>Edit {review_rel}</p></article>")
    order = {r['id']: i for i, r in enumerate(plan)}
    if len({r['id'] for r in merged}) != len(merged):
        raise ValueError('Duplicate candidates')
    for filename, rows in [('manifest_wan.jsonl', merged), ('accepted_manifest_wan.jsonl', accepted)]:
        rows.sort(key=lambda r: order[r['id']])
        tmp = target / (filename + '.tmp')
        tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        tmp.replace(target / filename)
    summary = dict(schema=SCHEMA, model=MODEL, revision=REVISION, generation=GENERATION,
                   expected_candidates=40, completed_candidates=len(merged), accepted_candidates=len(accepted),
                   completed_splits={s: sum(r['split'] == s for r in merged) for s in ('train', 'val', 'test')},
                   accepted_splits={s: sum(r['split'] == s for r in accepted) for s in ('train', 'val', 'test')},
                   training_compatibility='Wan-specific dataset. Existing Zeroscope cache/train commands are NOT compatible.')
    atomic_json(target / 'dataset_summary.json', summary)
    (target / 'review_all.html').write_text("<!doctype html><meta charset='utf-8'><title>Wan candidate review</title><h1>Wan dataset candidates</h1><p>Edit each shard review.csv, then run import_wan_shards.py --refresh-only.</p>" + '\n'.join(cards), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, help='Path to reverse_video_project')
    p.add_argument('--bundles', nargs='*', default=[])
    p.add_argument('--plan', default=str(Path(__file__).resolve().parent.parent / 'prompts_40.json'))
    p.add_argument('--refresh-only', action='store_true')
    p.add_argument('--replace-review', action='store_true', help='Explicitly replace local review decisions with incoming ones')
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    if not args.bundles and not args.refresh_only:
        p.error('Supply --bundles or --refresh-only')
    plan = read_json(args.plan)
    target = Path(args.project) / 'data' / 'wan_pilot_v1'
    target.mkdir(parents=True, exist_ok=True)
    for bundle in args.bundles:
        print('Importing:', bundle)
        import_bundle(bundle, target, plan, args.replace_review)
    summary = rebuild(target, plan)
    print(json.dumps(summary, indent=2))
    print('Dataset:', target)
    if args.require_complete and summary['completed_candidates'] != 40:
        raise SystemExit('Import is partial: fewer than 40 completed candidates. Existing imports remain saved.')

if __name__ == '__main__':
    main()
