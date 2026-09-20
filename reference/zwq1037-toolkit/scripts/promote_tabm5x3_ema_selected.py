from pathlib import Path
import json, shutil
root=Path('.')
meta_path=root/'outputs/submission_metadata/tabm385_k32_temporal5fold_3seed_ema0999.json'
meta=json.loads(meta_path.read_text(encoding='utf-8'))
by={r['key']:r for r in meta['candidates']}
base=by['last3_equal_ema000']; selected=meta['selected']
deltas={m:selected['monthly'][m]-base['monthly'][m] for m in selected['monthly']}
src=Path(meta['outputs']['selected'])
dst=root/'outputs/submissions/standalone_tabm385_k32_temporal3fold_recent123_ema0999.csv'
shutil.copy2(src,dst)
summary={
 'run_name':'standalone_tabm385_k32_temporal3fold_recent123_ema0999',
 'source_experiment':meta['run_name'],
 'effective_fold_ends':[52,57,62],
 'fold_weights':[1,2,3],
 'seeds':meta['seeds'],
 'model_count':9,
 'ema_decay':meta['ema_decay'],
 'inference_weights':'EMA',
 'baseline':'original temporal3fold x 3seed raw equal',
 'baseline_scores':{k:base[k] for k in ('selection_63_65','forward_67_70','all_63_70_ex66')},
 'selected_scores':{k:selected[k] for k in ('selection_63_65','forward_67_70','all_63_70_ex66')},
 'deltas':{k:selected[k]-base[k] for k in ('selection_63_65','forward_67_70','all_63_70_ex66')},
 'monthly_deltas':deltas,
 'months_improved':sum(v>0 for v in deltas.values()),
 'correlation_with_old_public_3x3':0.999092991058253,
 'submission_path':str(dst.resolve()),
 'submission_status':'prepared_not_submitted',
 'note':'The two added early folds 42 and 47 reduced validation performance, so the selected candidate keeps the recent three folds and uses recency weights plus EMA.'
}
summary_path=root/'outputs/submission_metadata/standalone_tabm385_k32_temporal3fold_recent123_ema0999.json'
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plan_path=root/'outputs/submission_metadata/tabm_temporal5x3_ema_plan_20260918.json'
plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
plan['status']='complete'
plan['result']=summary
plan['next_action']='Complete. The user may manually submit the promoted CSV. Do not submit automatically.'
plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
