"""Correct ensemble aggregation, frozen postprocessing, and paired temporal reports."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit.dataset import Dataset
from audit.io import write_json, save_npz, sha256
from audit.metrics import PredictionMean, panel, paired_panel, postprocess
from audit.protocol import jobs


def summarize(args):
    out = Path(args.out)
    manifests = list(out.glob('run_worker_*.json'))
    if not manifests:
        raise ValueError('No run manifest')
    meta = json.loads(manifests[0].read_text(encoding='utf-8'))
    if any(json.loads(p.read_text(encoding='utf-8'))['run_signature'] != meta['run_signature'] for p in manifests):
        raise ValueError('Mixed worker recipes')
    config = meta['config']
    data = Dataset(args.data)
    if data.manifest['fingerprint'] != meta['dataset_fingerprint']:
        raise ValueError('Different dataset')
    expected = jobs(config)
    missing = [j.key for j in expected if not (out/'jobs'/j.key/'done.json').exists()]
    if missing:
        result = {'complete':False,'eligible_for_review':False,'missing':missing,
                  'note':'No comparison or submission until every planned model is complete.'}
        write_json(out/'summary.json',result)
        print(json.dumps(result,indent=2))
        return result
    results, vectors, global_rows = {}, {}, None
    n = len(data.y)
    for arm in config['arms']:
        all_raw,all_post = np.full(n,np.nan),np.full(n,np.nan)
        test_mean = PredictionMean(len(data.load('test_ids.npy'))) if config['mode']=='final' else None
        bounds_by_origin, counts_by_origin, panels_by_origin = {}, {}, {}
        seed_vectors = {str(s):np.full(n,np.nan) for s in config['seeds']}
        for origin in sorted({j.origin for j in expected}):
            es_mean,score_mean = PredictionMean(n),PredictionMean(n)
            seed_means = {str(s):PredictionMean(n) for s in config['seeds']}
            group = [j for j in expected if j.arm==arm and j.origin==origin]
            for j in group:
                folder = out/'jobs'/j.key
                done = json.loads((folder/'done.json').read_text(encoding='utf-8'))
                if done['run_signature'] != meta['run_signature']:
                    raise ValueError('Incompatible model predictions')
                if sha256(folder/'predictions.npz') != done['artifacts']['predictions.npz']:
                    raise ValueError('Corrupted predictions')
                with np.load(folder/'predictions.npz') as p:
                    _,es_mask,score_mask = j.masks(data.months)
                    if not np.array_equal(p['es_rows'],np.flatnonzero(es_mask)) or not np.array_equal(p['score_rows'],np.flatnonzero(score_mask)):
                        raise ValueError('Prediction rows differ from protocol')
                    es_mean.add(j.key,p['es_rows'],p['es_pred'])
                    score_mean.add(j.key,p['score_rows'],p['score_pred'])
                    seed_means[str(j.seed)].add(j.key,p['score_rows'],p['score_pred'])
                    if test_mean is not None:
                        if not np.array_equal(p['test_rows'],np.arange(len(test_mean.total))):
                            raise ValueError('Test mapping differs from IDs')
                        test_mean.add(j.key,p['test_rows'],p['test_pred'])
            seen = es_mean.count > 0
            q = config['clip_quantiles']
            bounds = np.quantile(es_mean.mean()[seen],q).tolist() if q is not None else None
            rows = np.flatnonzero(score_mean.count > 0)
            expected_count = len(group) if config['mode']=='confirm' else len(config['seeds'])
            if not np.all(score_mean.count[rows] == expected_count):
                raise ValueError('Incomplete ensemble coverage')
            if np.isfinite(all_raw[rows]).any():
                raise ValueError('Outer windows overlap')
            raw = score_mean.mean()[rows]
            all_raw[rows] = raw
            all_post[rows] = postprocess(raw,bounds,data.load('train_nodata.npy')[rows])
            bounds_by_origin[str(origin)] = bounds
            counts_by_origin[str(origin)] = expected_count
            panels_by_origin[str(origin)] = panel(all_post[rows],data.y[rows],data.months[rows],config['metric'])
            for seed,mean in seed_means.items():
                seed_vectors[seed][rows] = postprocess(mean.mean()[rows],bounds,data.load('train_nodata.npy')[rows])
        rows = np.flatnonzero(np.isfinite(all_raw))
        if global_rows is not None and not np.array_equal(global_rows,rows):
            raise ValueError('Arms evaluated on different rows')
        global_rows = rows
        results[arm] = {'raw':panel(all_raw[rows],data.y[rows],data.months[rows],config['metric']),
                        'processed':panel(all_post[rows],data.y[rows],data.months[rows],config['metric']),
                        'bounds_by_origin':bounds_by_origin,'model_count_by_origin':counts_by_origin,
                        'processed_by_origin':panels_by_origin,
                        'processed_by_seed':{seed:panel(v[rows],data.y[rows],data.months[rows],config['metric'])
                                             for seed,v in seed_vectors.items()}}
        arrays = {'rows':rows,'sample_ids':data.ids[rows],'raw':all_raw[rows],'processed':all_post[rows]}
        if test_mean is not None:
            if not np.all(test_mean.count == len([j for j in expected if j.arm==arm])):
                raise ValueError('Incomplete test ensemble')
            arrays['test_ids'] = data.load('test_ids.npy')
            arrays['test_raw'] = test_mean.mean()
            arrays['test_processed'] = postprocess(arrays['test_raw'],bounds_by_origin['70'],data.load('test_nodata.npy'))
        save_npz(out/f'aggregate_{arm}.npz',**arrays)
        vectors[arm] = all_post[rows]
    comparisons = {}
    pairs = [('base',arm) for arm in vectors if arm!='base'] if 'base' in vectors else []
    pairs += [(ref,arm) for ref,arm in [('flow','flow31'),('flow','flow_price'),('flow','x30'),('flow_price','x30')]
              if ref in vectors and arm in vectors]
    for ref,arm in pairs:
        comparisons[f'{arm}-{ref}'] = paired_panel(vectors[ref],vectors[arm],data.y[global_rows],data.months[global_rows],config['metric'])
        for field in ('processed_by_origin','processed_by_seed'):
            comparisons[f'{arm}-{ref}'][field+'_delta'] = {
                key:results[arm][field][key]['overall']-base['overall']
                for key,base in results[ref][field].items()}
    result = {'complete':True,'metric_verified':config['metric_verified'],
              'eligible_for_review':config['mode']=='confirm' and config['metric_verified'],
              'independent_outer_score':config['mode']=='confirm',
              'note':'screen/final validation selected checkpoints on these labels; it is not independent OOF performance.',
              'metric':config['metric'],'mode':config['mode'],'arms':results,'paired_comparisons':comparisons,
              'run_signature':meta['run_signature']}
    write_json(out/'summary.json',result)
    if args.submission_template:
        if config['mode']!='final' or not config['metric_verified'] or len(config['arms'])!=1:
            raise ValueError('Export requires one frozen final arm and verified scorer')
        import pandas as pd
        template = pd.read_csv(args.submission_template)
        with np.load(out/f'aggregate_{config["arms"][0]}.npz') as pred:
            ids,p = pred['test_ids'],pred['test_processed']
            if not np.isfinite(p).all() or template.sample_id.duplicated().any() or set(template.sample_id)!=set(ids.tolist()):
                raise ValueError('Submission IDs/finiteness failure')
            target_columns = [c for c in template if c!='sample_id']
            if len(target_columns)!=1:
                raise ValueError('Ambiguous submission schema')
            template[target_columns[0]] = pd.Series(p,index=ids).reindex(template.sample_id).to_numpy()
            template.to_csv(out/'submission.csv',index=False)
    print('Complete:',len(expected),'models. Report:',out/'summary.json',flush=True)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--submission-template')
    summarize(p.parse_args())
