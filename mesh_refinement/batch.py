"""Manifest-driven dataset evaluation; source ordinals are never renumbered."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .core import refine_mesh, paper_fusion_config
from .fusion import validate_skinning_fusion_config
from .metrics import evaluate_mesh, evaluate_seam_opening, _validate_pose_options


GLOBAL_METRICS = ('skin_l1', 'de', 'fpr', 'cd', 'nme', 'so')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_manifest(path, input_dir=None):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get('records'), list) or not document['records']:
        raise ValueError('manifest must contain a nonempty records list')
    root = (Path(input_dir) if input_dir is not None else path.parent).resolve()
    records, ids, paths = [], set(), set()
    for row in document['records']:
        key = row['sample_id']
        if (not isinstance(key, str) or not key or key in ('.', '..') or
                any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in key)):
            raise ValueError('unsafe sample_id')
        relative = Path(row['file'])
        target = (root / relative).resolve()
        if relative.is_absolute() or root not in target.parents or target.suffix != '.npz':
            raise ValueError('record file must be an NPZ inside input-dir')
        ordinal = row.get('sample_index', row.get('source_index'))
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError('each record requires a nonnegative sample_index or source_index')
        if key in ids or target in paths:
            raise ValueError('duplicate sample_id or input file')
        if not target.is_file():
            raise FileNotFoundError(target)
        ids.add(key); paths.add(target)
        records.append(dict(row, sample_id=key, sample_index=ordinal, input_path=str(target),
                            index_basis='sample_index' if 'sample_index' in row else 'source_index'))
    if 'population_size' in document and document['population_size'] != len(records):
        raise ValueError('manifest population_size disagrees with records')
    return document, records


def _evaluate_one(task):
    from .cli import _load_npz
    row, output, settings, config, expected_weights = task
    key, path = row['sample_id'], Path(row['input_path'])
    data = _load_npz(path, ('vertices', 'faces', 'skin', 'gt_skin'))
    metadata = json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
    if metadata.get('sample_id', key) != key:
        raise ValueError('manifest sample_id disagrees with NPZ metadata: ' + key)
    if expected_weights is not None and metadata.get('weights') != expected_weights:
        raise ValueError('manifest weight mode disagrees with NPZ metadata: ' + key)
    present = ('matrix_local' in data, 'parents' in data)
    if any(present) and not all(present):
        raise ValueError('partial skeleton: ' + key)
    skeleton = {k: data[k] for k in ('matrix_local', 'parents')} if all(present) else None
    vertices, faces = data['vertices'], data['faces']
    refined = refine_mesh(vertices, faces, data['skin'], fusion_config=config)['evaluation']
    original = dict(vertices=vertices, faces=faces, skin=data['skin'],
                    keep_indices=np.arange(len(vertices)), merged_old_to_new=np.arange(len(vertices)),
                    kept_source_face_indices=np.arange(len(faces)))
    options = dict(settings, sample_index=row['sample_index'], skeleton=skeleton)
    methods = {name: evaluate_mesh(vertices, faces, data['gt_skin'], state, **options)
               for name, state in (('original', original), ('refined', refined))}
    seam = evaluate_seam_opening(vertices, faces, data['gt_skin'],
        {'original': original, 'refined': refined}, **options)
    for name in methods:
        methods[name].update(so=seam['methods'][name]['so'],
            so_pair_scope='common_original_and_refined', so_common_pair_count=seam['common_pair_count'])
    arrays = {k: x for k, x in refined.items() if isinstance(x, np.ndarray)}
    arrays['gt_skin'] = data['gt_skin'][refined['keep_indices']]
    if skeleton is not None:
        arrays.update(skeleton)
    arrays['metadata_json'] = np.asarray(json.dumps(dict(release_version=__version__, phase='evaluation',
        fusion=asdict(config), sample_id=key, sample_index=row['sample_index']), allow_nan=False))
    target = Path(output) / 'refined' / (key + '.npz')
    with target.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result = dict(sample_id=key, sample_index=row['sample_index'], index_basis=row['index_basis'],
        input_file=row['file'], input_sha256=file_hash(path), refined_sha256=file_hash(target),
        methods=methods, seam=seam)
    write_json(Path(output) / 'records' / (key + '.json'), result)
    return result


def aggregate_records(records):
    """Global/SO macro means; OFPR pair pooling; Local original-vertex pooling."""
    local=(('local_skin_l1','local_skin_valid_vertex_count'),('local_de','local_de_valid_vertex_count'))
    methods={name:{k:dict(total=0.,count=0,meshes=0) for k in (*GLOBAL_METRICS,'ofpr','local_skin_l1','local_de')}
             for name in ('original','refined')}
    meshes=0
    for record in records:
        meshes+=1
        for name,acc in methods.items():
            scores=record['methods'][name]
            for key in GLOBAL_METRICS:
                if scores[key] is not None:
                    acc[key]['total']+=scores[key];acc[key]['count']+=1
            if scores['ofpr_preserved_pair_count'] is not None:
                acc['ofpr']['total']+=scores['ofpr_preserved_pair_count']
                acc['ofpr']['count']+=scores['ofpr_source_pair_count']
            for key,count_key in local:
                if scores[key] is not None:
                    acc[key]['total']+=scores[key]*scores[count_key]
                    acc[key]['count']+=scores[count_key];acc[key]['meshes']+=1
    if not meshes:raise ValueError('cannot aggregate an empty run')
    result=dict(mesh_count=meshes,methods={})
    for name,acc in methods.items():
        metrics={key:dict(mean=acc[key]['total']/acc[key]['count'] if acc[key]['count'] else None,
                          valid_mesh_count=acc[key]['count']) for key in GLOBAL_METRICS}
        metrics['ofpr']=dict(mean=acc['ofpr']['total']/acc['ofpr']['count'] if acc['ofpr']['count'] else None,
            source_pair_count=acc['ofpr']['count'],preserved_pair_count=int(acc['ofpr']['total']))
        for key,_ in local:
            metrics[key]=dict(mean=acc[key]['total']/acc[key]['count'] if acc[key]['count'] else None,
                valid_vertex_count=acc[key]['count'],valid_mesh_count=acc[key]['meshes'])
        result['methods'][name]=metrics
    return result


def _write_csv(path, records):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        fields = ['sample_id', 'sample_index', 'method', *GLOBAL_METRICS,
                  'ofpr', 'local_skin_l1', 'local_de']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in records:
            for name, scores in row['methods'].items():
                writer.writerow(dict(sample_id=row['sample_id'], sample_index=row['sample_index'],
                    method=name, **{k: scores[k] for k in fields[3:]}))


def run_dataset(manifest, output_dir, *, input_dir=None, fusion_config=None,
                workers=1, seed=321, pose_count=64, pose_angle_deg=30.0):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output already exists: '+str(output))
    document, rows = read_manifest(manifest, input_dir)
    config = validate_skinning_fusion_config(fusion_config or paper_fusion_config())
    _validate_pose_options(pose_count, pose_angle_deg, seed, 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    output.mkdir(parents=True, exist_ok=False)
    (output/'refined').mkdir(); (output/'records').mkdir()
    settings = dict(seed=seed, pose_count=pose_count, pose_angle_deg=pose_angle_deg)
    run = dict(release_version=__version__, dataset=document.get('dataset'), split=document.get('split'),
        weights=document.get('weights'), planned_mesh_count=len(rows), fusion=asdict(config),
        pose=settings, manifest_sha256=file_hash(manifest), python=platform.python_version(), numpy=np.__version__,
        module_sha256={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        records=[{k: r[k] for k in ('sample_id','sample_index','index_basis','file')} for r in rows])
    write_json(output/'run.json', run)
    tasks = [(row, str(output), settings, config, document.get('weights')) for row in rows]
    results = set()
    try:
        if workers == 1:
            for task in tasks:
                result = _evaluate_one(task); results.add(result['sample_id'])
                print('Evaluated {}/{}: {}'.format(len(results),len(rows),result['sample_id']),flush=True)
        else:
            # Keep only a small bounded work queue; raw datasets/shards are never reloaded here.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending, source = set(), iter(tasks)
                for _ in range(min(len(tasks),2*workers)):
                    pending.add(pool.submit(_evaluate_one,next(source)))
                while pending:
                    done, pending = wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result(); results.add(result['sample_id'])
                        print('Evaluated {}/{}: {}'.format(len(results),len(rows),result['sample_id']),flush=True)
                        task = next(source,None)
                        if task is not None: pending.add(pool.submit(_evaluate_one,task))
        def ordered():
            for row in rows:
                yield json.loads((output/'records'/(row['sample_id']+'.json')).read_text())
        summary = aggregate_records(ordered())
        summary.update(dataset=run['dataset'], split=run['split'], weights=run['weights'])
        write_json(output/'summary.json',summary)
        _write_csv(output/'metrics.csv',ordered())
        write_json(output/'complete.json',dict(status='complete',mesh_count=len(rows),
            run_sha256=file_hash(output/'run.json'), summary_sha256=file_hash(output/'summary.json'),
            record_sha256={r['sample_id']:file_hash(output/'records'/(r['sample_id']+'.json')) for r in rows}))
        return summary
    except Exception as error:
        write_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,message=str(error)))
        raise


def aggregate_runs(run_dirs, output):
    """Combine completed splits without averaging their already-averaged means."""
    records, seen, signature, dataset = [], set(), None, None
    for directory in map(Path,run_dirs):
        complete = json.loads((directory/'complete.json').read_text())
        run = json.loads((directory/'run.json').read_text())
        if complete['status'] != 'complete' or complete['run_sha256'] != file_hash(directory/'run.json'):
            raise ValueError('run is incomplete or its configuration changed')
        current = (run['fusion'],run['pose'],run['module_sha256'],run['weights'],run['dataset'])
        if signature is not None and signature != current:
            raise ValueError('cannot combine different methods, poses, code versions, datasets or weight modes')
        signature, dataset = current,run['dataset']
        for row in run['records']:
            key=row['sample_id']; record_path=directory/'records'/(key+'.json')
            if key in seen: raise ValueError('overlapping sample_id across runs')
            if complete['record_sha256'].get(key) != file_hash(record_path):
                raise ValueError('record changed after completion')
            seen.add(key); records.append(record_path)
    result = aggregate_records(json.loads(p.read_text()) for p in records); result['dataset']=dataset
    write_json(output,result)
    return result
