"""Screen the already-trained forum-style GRU-representation tree in the current owned block."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values)


def cosine(target, prediction):
    return float(np.dot(target, prediction) / (np.linalg.norm(target) * np.linalg.norm(prediction)))


def main():
    tabm = pd.read_feather(PROJECT/'data/interim/tree_experiments/EXP-TABM-032-MARKET385-K32/train059_valid6270_ex66/validation_predictions.feather')
    tabm = tabm.loc[tabm.month.ne(66), ['sample_id','month','target','candidate']].rename(columns={'candidate':'tabm'})
    real_dir = PROJECT/'data/interim/tree_experiments/EXP-REALMLP-009-OUR379-CORR095/train059_valid6270_ex66'
    real = pd.DataFrame({'sample_id':np.load(real_dir/'validation_sample_ids.npy'), 'realmlp':np.load(real_dir/'validation_predictions.npy')})
    old_trans = pd.read_csv(PROJECT/'outputs/kaggle_transformer_v7_result/factorized_transformer/validation_predictions.csv')[['sample_id','prediction']].rename(columns={'prediction':'transformer_old'})
    new_trans = pd.read_csv(PROJECT/'data/interim/kaggle_outputs/multistream_transformer_multiwindow_v14/factorized_transformer_multiwindow10/validation_predictions.csv')[['sample_id','prediction']].rename(columns={'prediction':'transformer_new'})
    gru = pd.read_csv(PROJECT/'data/interim/kaggle_outputs/multistream_factorized_gru_timeaware_dev/factorized_gru/validation_predictions.csv')[['sample_id','prediction']].rename(columns={'prediction':'gru'})
    embedding = pd.read_feather(PROJECT/'data/interim/tree_experiments/EXP-TREE-072-GRU96-EMBEDDING-SEED137/train059_valid6070/validation_predictions.feather')
    embedding = embedding.loc[embedding.month.between(62,70) & embedding.month.ne(66), ['sample_id','gru_embedding_tree']]
    frame = tabm.merge(real,on='sample_id',validate='one_to_one').merge(old_trans,on='sample_id',validate='one_to_one').merge(new_trans,on='sample_id',validate='one_to_one').merge(gru,on='sample_id',validate='one_to_one').merge(embedding,on='sample_id',validate='one_to_one')
    assert len(frame)==140806 and np.isfinite(frame.select_dtypes('number')).all().all()
    names = ['tabm','realmlp','transformer_old','transformer_new','gru','gru_embedding_tree']
    z = {name:unit(frame[name]) for name in names}
    target = frame.target.to_numpy(np.float64)
    months = frame.month.to_numpy()
    recipes = {
      'current_owned_proxy': {'tabm':.12,'realmlp':.03,'transformer_old':.18,'gru':.03},
      'multiwindow_half': {'tabm':.12,'realmlp':.03,'transformer_old':.09,'transformer_new':.09,'gru':.03},
      # 4.5% is the preregistered effective weight used by the earlier two-stage GRU+tree experiment.
      'embedding45_from_tabm': {'tabm':.075,'realmlp':.03,'transformer_old':.18,'gru':.03,'gru_embedding_tree':.045},
      'multiwindow_half_embedding45_from_tabm': {'tabm':.075,'realmlp':.03,'transformer_old':.09,'transformer_new':.09,'gru':.03,'gru_embedding_tree':.045},
    }
    predictions = {key:sum(weight*z[name] for name,weight in weights.items()) for key,weights in recipes.items()}
    base = predictions['current_owned_proxy']
    rows=[]
    for key,prediction in predictions.items():
        rows.append({'name':key,'weights':recipes[key], 'cosine':cosine(target,prediction),
          'delta_vs_current':cosine(target,prediction)-cosine(target,base),
          'monthly_delta_vs_current':{str(int(m)):cosine(target[months==m],prediction[months==m])-cosine(target[months==m],base[months==m]) for m in np.unique(months)}})
    report={'window':'train0-59 purge60-61 valid62-70 excluding66','rows':len(frame),
      'limitation':'Owned-model proxy only; production components use temporal/full variants and public 64% block has no matching validation.',
      'forum_direction':'frozen GRU representation plus tree; already trained previously, no new fit', 'candidates':rows}
    path=PROJECT/'outputs/submission_metadata/forum_gru_embedding_current_owned_screen.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
