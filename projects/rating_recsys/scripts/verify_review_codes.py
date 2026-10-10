"""Recompute validation metrics and decode saved codes without training."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import numpy as np
import torch
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import load_snapshot
from rating_recsys.experiments.rqvae_cli import file_hash,write_json
from rating_recsys.retrieval.rqvae import ReviewRQVAE,RQVAEConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();directory=args.run_dir
    metrics=json.loads((directory/'metrics.json').read_text())
    for name,expected in metrics['artifacts_sha256'].items():assert file_hash(directory/name)==expected,name
    for name,expected in metrics['source_sha256'].items():assert file_hash(PROJECT_ROOT/name)==expected,name
    checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    model=ReviewRQVAE(RQVAEConfig(**checkpoint['config']));model.load_state_dict(checkpoint['state_dict']);model.eval()
    torch.set_num_threads(4)
    with np.load(directory/'review_codes.npz',allow_pickle=False) as archive:
        codes=archive['codes'].astype(np.int64);review_ids=archive['review_ids'];train=archive['train_indices'];audit=archive['audit_indices']
    assert len(np.unique(review_ids))==metrics['training_reviews']==len(codes)
    assert codes.shape==(len(review_ids),model.config.levels)
    assert set(train.tolist()).isdisjoint(audit.tolist())
    assert sorted(np.concatenate((train,audit)).tolist())==list(range(len(codes)))
    with np.load(directory/'representations.npz',allow_pickle=False) as archive:
        expected_vectors=archive['decoded']
        decoded=[]
        with torch.no_grad():
            for offset in range(0,len(codes),2048):decoded.append(model.decode_codes(torch.from_numpy(codes[offset:offset+2048])).numpy())
        actual_vectors=np.concatenate(decoded)
    max_error=float(np.abs(actual_vectors-expected_vectors).max())
    assert np.allclose(actual_vectors,expected_vectors,atol=2e-6,rtol=1e-5)
    snapshot=PROJECT_ROOT/'artifacts/snapshots/e7896add5b4b5939.jsonl'
    assert file_hash(snapshot)==metrics['snapshot_sha256']
    split=build_global_temporal_split(load_snapshot(snapshot))
    config=ExperimentConfig(satisfaction_mode='history-aware',ranker_training_mode='window')
    queries,_=build_window_queries(split.train,split.validation,config=config,phase='validation',cutoff=split.train_cutoff)
    relevant=[q for q in queries if any(v>0 for v in q.relevance_by_item.values())]
    rankings=json.loads((directory/'rankings_validation.json').read_text())
    counts=Counter(row.restaurant_id for row in split.train)
    visited={q.query_id:{row.restaurant_id for row in q.history} for q in queries}
    recomputed={}
    # Separate direct formulas; do not call evaluate_rankings/query_scores.
    for name,by_query in rankings.items():
        assert set(by_query)=={q.query_id for q in queries}
        for qid,ranking in by_query.items():
            assert len(ranking)==len(set(ranking))
            assert not (set(ranking)&visited[qid])
            assert set(ranking)<=set(counts)
        values={}
        for cutoff in (10,100):
            ndcg=[];recall=[]
            for q in relevant:
                truth={key:value for key,value in q.relevance_by_item.items() if value>0}
                ordered=by_query[q.query_id][:cutoff]
                dcg=sum((2**truth.get(key,0)-1)/math.log2(rank+2) for rank,key in enumerate(ordered))
                ideal=sum((2**value-1)/math.log2(rank+2) for rank,value in enumerate(sorted(truth.values(),reverse=True)[:cutoff]))
                ndcg.append(dcg/ideal if ideal else 0.)
                recall.append(sum(key in truth for key in ordered)/len(truth))
            values[f'ndcg_at_{cutoff}']=float(np.mean(ndcg));values[f'recall_at_{cutoff}']=float(np.mean(recall))
        for key,value in values.items():assert abs(value-metrics['recommendation']['metrics'][name][key])<1e-12,(name,key)
        recomputed[name]=values
    result={'status':'passed','saved_codes_decode_max_abs_error':max_error,
            'reviews':len(codes),'validation_queries':len(queries),'relevant_validation_queries':len(relevant),
            'independent_direct_metric_recalculation':recomputed,
            'visited_catalog_duplicate_checks':True,'source_and_artifact_hashes_match':True,'training_steps':0,'test_evaluated':False}
    write_json(directory/'verification.json',result)
    print(json.dumps({key:result[key] for key in ('status','reviews','validation_queries','relevant_validation_queries','saved_codes_decode_max_abs_error')}))


if __name__=='__main__':main()
