"""Full validation of opinion-channel graph alignment and separate interests.

Candidate generation only: neither ranker inference/fitting nor test scoring.
Method 1 uses local rule-extracted opinions and fresh local pinned E5 inputs.
Method 2 is a frozen text-cluster adapter, not neural ComiRec reproduction.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings
from rating_recsys.experiments.artifacts import read_json, write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path
from rating_recsys.experiments.review_multivector_cli import (
    load_prepared, protect_ranker, paired_change, candidate_changes, validate_rankings,
)
from rating_recsys.retrieval.aspect_graph import (
    CHANNELS, AspectGraphLightGCN, build_opinion_profiles, mean_opinion_profiles, opinion_input,
)
from rating_recsys.retrieval.hybrid import rrf
from rating_recsys.retrieval.multi_interest import MultiInterestConfig, MultiInterestRetriever, build_multi_interest_bank
from rating_recsys.retrieval.review_embeddings import LocalEmbeddingCache, model_embedding_config, E5_MODEL
from rating_recsys.retrieval.review_multivector import ReviewVectorBank, array_hash, supplement_candidates
from rating_recsys.retrieval.review_opinions import extract_opinions
from rating_recsys.retrieval.rlmrec import RLMRecConfig, RLMRecLightGCN

VERSION = "text-candidate-opinion-graph-and-multi-interest-v1"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def packed_bank(path):
    with np.load(path) as saved:
        return dict(saved)


def repeated_mean_profiles(bank):
    means = mean_opinion_profiles(bank)
    result = {}
    for key, value in bank.items():
        channels = value.reshape(len(CHANNELS),384)
        mask = np.any(channels != 0, axis=1)
        repeated = np.zeros_like(channels)
        repeated[mask] = means[key]
        assert np.array_equal(np.any(repeated != 0, axis=1), mask)
        result[key] = repeated.reshape(-1)
    return result


def shuffle_profiles(bank, seed):
    keys = sorted(bank)
    result = {key:value.copy() for key,value in bank.items()}
    rng = np.random.default_rng(seed)
    permutations = {}
    for channel in range(len(CHANNELS)):
        observed = [key for key in keys if np.any(bank[key].reshape(len(CHANNELS),384)[channel])]
        permutation = rng.permutation(len(observed))
        for key,source in zip(observed,permutation):
            result[key].reshape(len(CHANNELS),384)[channel] = bank[observed[source]].reshape(len(CHANNELS),384)[channel]
        permutations[str(channel)] = {'entity_ids':observed,'permutation':permutation.tolist()}
    return result, permutations


def save_profiles(path, users, items):
    arrays = {}
    for name, bank in (("user",users),("item",items)):
        keys = sorted(bank)
        arrays[name+'_ids'] = np.asarray(keys,np.int64)
        arrays[name+'_profiles'] = np.stack([bank[key] for key in keys]) if keys else np.empty((0,len(CHANNELS)*384),np.float32)
    np.savez_compressed(path,**arrays)
    return {name:array_hash(value) for name,value in arrays.items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=PROJECT_ROOT/'artifacts/snapshots/e7896add5b4b5939.jsonl')
    p.add_argument('--baseline-run',type=Path,default=PROJECT_ROOT/'artifacts/runs/20261004T125314601899Z-e7896add')
    p.add_argument('--prepared-validation',type=Path,default=PROJECT_ROOT/'artifacts/prepared/e7896add5b4b5939/validation-1d1692022f92bc38/manifest.json')
    p.add_argument('--review-run',type=Path,default=PROJECT_ROOT/'artifacts/comparisons/review_multivector/20261010T153953859126Z-e7896add')
    p.add_argument('--graph-checkpoint',type=Path,default=PROJECT_ROOT/'artifacts/comparisons/rlmrec/20261010T003719015651Z-e7896add/lambda_0.npz')
    p.add_argument('--embedding-cache',type=Path,default=PROJECT_ROOT/'artifacts/e5_opinion_embedding_cache.sqlite')
    p.add_argument('--local-model-cache',type=Path,default=PROJECT_ROOT/'artifacts/local_models')
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'artifacts/comparisons/text_candidate_views')
    p.add_argument('--threads',type=int,default=4)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    baseline = read_json(args.baseline_run/'manifest.json')
    config = dict(baseline['config'])
    for key in ('num_leaves_grid','min_child_samples_grid'):
        config[key] = tuple(config[key])
    exp = ExperimentConfig(**config)
    rows = load_snapshot(args.snapshot)
    snapshot_id = dataset_digest(rows)
    if snapshot_id != baseline['snapshot']['dataset_snapshot_id']:
        raise ValueError('Snapshot differs from baseline')
    split = build_global_temporal_split(rows,train_fraction=exp.train_fraction,validation_fraction=exp.validation_fraction)
    queries,new_users = build_window_queries(split.train,split.validation,config=exp,phase='validation',cutoff=split.train_cutoff)
    prepared,c5 = load_prepared(args.prepared_validation,snapshot_id,split,exp,queries)
    c1 = {key:tuple(value) for key,value in prepared['data']['ordered']['c1_item_item'].items()}
    c4 = {key:tuple(value) for key,value in prepared['data']['ordered']['c4_lightgcn'].items()}
    catalog = {row.restaurant_id for row in split.train}
    excluded = {q.user_id:{row.restaurant_id for row in q.history} for q in queries}
    user_ids = [q.user_id for q in queries]
    review_manifest = read_json(args.review_run/'metrics.json')
    if review_manifest['snapshot_id'] != snapshot_id or review_manifest['train_cutoff'] != split.train_cutoff.isoformat():
        raise ValueError('Review bank cutoff or snapshot differs')
    if digest_file(args.review_run/'review_bank.npz') != review_manifest['artifacts_sha256']['review_bank.npz']:
        raise ValueError('Review bank file hash differs')
    bank_arrays = packed_bank(args.review_run/'review_bank.npz')
    for name,value in bank_arrays.items():
        if array_hash(value) != review_manifest['profiles']['arrays_sha256'][name]:
            raise ValueError('Review bank array hash differs')
    bank = ReviewVectorBank(**bank_arrays,audit=review_manifest['profiles'])
    if set(bank.user_ids.tolist()) != set(user_ids) or set(bank.item_ids.tolist()) != catalog:
        raise ValueError('Review bank population differs')
    graph = RLMRecLightGCN.load(args.graph_checkpoint)
    if graph.config != exp.lightgcn_config or graph.projector is not None or graph.config.epochs != 20:
        raise ValueError('Require identical textless 20epoch graph')
    run = args.output_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+snapshot_id[:8])
    run.mkdir(parents=True,exist_ok=False)
    protected = protect_ranker(args.baseline_run)
    state = {'version':VERSION,'status':'preparing','created_at':datetime.now(timezone.utc).isoformat(),
             'snapshot_id':snapshot_id,'train_cutoff':split.train_cutoff.isoformat(),
             'validation_through':split.validation_cutoff.isoformat(),'train_rows':len(split.train),
             'validation_queries':len(queries),'relevant_validation_queries':sum(bool(q.relevant) for q in queries),
             'new_validation_users_excluded':new_users,'catalog':len(catalog),
             'experiment':exp.to_dict(),'ranker_trained':False,'ranker_predicted':False,'test_evaluated':False,
             'api_calls':0,'db_queries':0,'ranker_hashes_before':protected,
             'selection':'fixed full20epoch graph arms; K1/K3 all reported; no winner activation',
             'command':[sys.executable,'-m','rating_recsys.experiments.text_candidate_views_cli',*(argv or sys.argv[1:])],
             'sources':{name:{'path':str(path.resolve()),'sha256':digest_file(path)} for name,path in
                        (('snapshot',args.snapshot),('sidecar',review_texts_path(args.snapshot)),
                         ('prepared',args.prepared_validation),('review_metrics',args.review_run/'metrics.json'),
                         ('review_bank',args.review_run/'review_bank.npz'),('graph',args.graph_checkpoint))},
             'variants':[]}
    source = code_manifest(PROJECT_ROOT,allow_dirty=True)
    (run/'source.diff').write_text(source.pop('git_diff'))
    state['code'] = source
    for name in ('retrieval/aspect_graph.py','retrieval/review_opinions.py','retrieval/multi_interest.py',
                 'experiments/text_candidate_views_cli.py'):
        destination = run/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT_ROOT/'src/rating_recsys'/name,destination)
    def log(message):
        print(message,flush=True)
    def update(**values):
        state.update(values)
        write_json(run/'progress.json',state)
    log(str(run))
    arms = {'c5':c5,'c4':c4}
    population = Counter(row.restaurant_id for row in split.train)
    def metrics(rankings):
        return evaluate_rankings([RankingObservation(q.query_id,q.user_id,q.relevance_by_item,tuple(rankings[q.query_id]))
                                  for q in queries],cutoffs=(10,20,50,100),catalog_ids=catalog,item_popularity=population)
    def save_ranks():
        write_jsonl(run/'candidates_validation.jsonl',({'query_id':q.query_id,'user_id':q.user_id,
                     'relevance_by_item':q.relevance_by_item,**{name:value[q.query_id] for name,value in arms.items()}}
                     for q in queries))
    def fuse_graph(rankings):
        return {q.query_id:rrf([c1[q.query_id],rankings[q.query_id]],k=100,constant=exp.rrf_constant) for q in queries}
    try:
        with threadpool_limits(limits=1,user_api='blas'):
            reproduced = graph.recommend(user_ids,excluded,100)
            assert all(reproduced[q.user_id] == c4[q.query_id] for q in queries)
            assert fuse_graph(c4) == c5
            update(status='multi-interest',baseline_c4_c5_reproduced=True)
            state['multi_interest'] = []
            for k in (1,3):
                part_started = time.perf_counter()
                interests = build_multi_interest_bank(bank,split.train,graph.item_keys,graph.item_embeddings,
                                                       config=MultiInterestConfig(interests=k))
                interests.save(run/f'interests_k{k}.npz')
                write_json(run/f'interests_k{k}.json',interests.audit)
                retriever = MultiInterestRetriever(interests)
                scores = retriever.score_all()
                np.savez_compressed(run/f'interests_k{k}.scores.npz',**scores)
                for view,values in scores.items():
                    ranks = retriever.rankings(values,excluded,k=100)
                    name = f'interests_{view}_k{k}'
                    arms[name] = {q.query_id:ranks.merged[q.user_id] for q in queries}
                    arms[name+'_c1'] = fuse_graph(arms[name])
                    arms[name+'_quota'] = {q.query_id:supplement_candidates(c5[q.query_id],arms[name][q.query_id],text_slots=10)
                                          for q in queries}
                    write_jsonl(run/f'{name}.per_interest.jsonl',({'query_id':q.query_id,'user_id':q.user_id,
                                'per_interest':ranks.per_interest[q.user_id],'merged':ranks.merged[q.user_id]} for q in queries))
                record = {'interests':k,'seconds':time.perf_counter()-part_started,'audit':interests.audit,
                          'new_neural_training_steps':0,'source_graph_epochs':20,
                          'metrics':{name:metrics(value) for name,value in arms.items() if name.startswith('interests_') and f'_k{k}' in name}}
                state['multi_interest'].append(record)
                save_ranks()
                update(status='multi-interest-complete')
                log(f'K={k} multi-interest full validation complete in {record["seconds"]:.1f}s')
                del retriever,interests,scores,ranks,values
        del bank,bank_arrays,graph
        gc.collect()
        texts,text_manifest = load_review_texts(review_texts_path(args.snapshot),rows)
        opinions = {}
        extraction_started = time.perf_counter()
        for row in sorted(split.train,key=lambda r:(r.event_date,r.review_id)):
            records = extract_opinions(texts.get(row.review_id,''))
            if records:
                opinions[row.review_id] = records
        from rating_recsys.retrieval import review_opinions
        extraction_version = getattr(review_opinions,'VERSION',getattr(review_opinions,'EXTRACTION_VERSION','local-rule-opinions-v1'))
        write_jsonl(run/'opinions_train.jsonl',({'review_id':rid,'raw_text_sha256':hashlib.sha256(texts[rid].encode()).hexdigest(),
                     'opinions':records} for rid,records in sorted(opinions.items())))
        polarities = Counter(record['polarity'] for records in opinions.values() for record in records)
        categories = Counter(record['category'] for records in opinions.values() for record in records)
        inputs = sorted({opinion_input(record,role) for records in opinions.values() for record in records
                         if (record['category'],record['polarity']) in CHANNELS for role in ('query','passage')})
        update(status='embedding-opinions',extraction={'version':extraction_version,'method':'local conservative rules; no star labels',
               'reviews_with_opinions':len(opinions),'opinions':sum(map(len,opinions.values())),
               'polarity_counts':dict(polarities),'category_counts':dict(categories),'unique_role_inputs':len(inputs),
               'seconds':time.perf_counter()-extraction_started},review_texts=text_manifest)
        log(f'Extracted {sum(map(len,opinions.values()))} opinions; local E5 inputs {len(inputs)}')
        embedding_config = replace(model_embedding_config(E5_MODEL,str(args.embedding_cache)),
                                   batch_size=64,max_document_tokens=128,cpu_threads=args.threads,
                                   local_model_cache=str(args.local_model_cache.resolve()))
        cache = LocalEmbeddingCache(args.embedding_cache,embedding_config,progress=log)
        try:
            embedding_vectors = cache.embed(inputs)
            update(embedding_usage=cache.usage,embedding_identity=cache.identity,
                   embedding_cache={'path':str(args.embedding_cache.resolve()),'totals':cache.totals()})
            user_profiles,item_profiles,audit = build_opinion_profiles(split.train,split.train_cutoff,texts,opinions,embedding_vectors)
            state['opinion_profile_arrays_sha256'] = save_profiles(run/'opinion_profiles.npz',user_profiles,item_profiles)
            write_json(run/'opinion_profiles.json',audit)
        finally:
            cache.close()
        del embedding_vectors,cache,opinions,texts,inputs
        gc.collect()
        state['opinion_profiles'] = audit
        pooled_users,pooled_items = repeated_mean_profiles(user_profiles),repeated_mean_profiles(item_profiles)
        save_profiles(run/'pooled_control_profiles.npz',pooled_users,pooled_items)
        shuffled_users,up = shuffle_profiles(user_profiles,42)
        shuffled_items,ip = shuffle_profiles(item_profiles,43)
        save_profiles(run/'shuffled_profiles.npz',shuffled_users,shuffled_items)
        write_json(run/'shuffle.json',{'users':up,'items':ip})
        variants = [('opinion_pooled_control',.001,pooled_users,pooled_items),
                    ('opinion_channels',.001,user_profiles,item_profiles),
                    ('opinion_channels_weight_01',.01,user_profiles,item_profiles),
                    ('opinion_channels_shuffled',.001,shuffled_users,shuffled_items)]
        with threadpool_limits(limits=1,user_api='blas'):
            for name,weight,users,items in variants:
                record = {'name':name,'alignment_weight':weight,'epochs':[],'new_graph_training':True,
                          'training_mode':'static full T1 graph all-visit BPR; no historical sequential claims',
                          'loss':'original BPR + weight * sum(role mean(valid channel InfoNCE))'}
                state['variants'].append(record)
                update(status='training-'+name)
                model = AspectGraphLightGCN(exp.lightgcn_config,RLMRecConfig(weight=weight,threads=args.threads),users,items)
                def callback(stats,current):
                    raw = current.recommend(user_ids,excluded,100)
                    candidate = {q.query_id:raw[q.user_id] for q in queries}
                    alignment = asdict(current.alignment_history[-1])
                    record['epochs'].append({**asdict(stats),'alignment':alignment,'standalone':metrics(candidate),
                                             'c1_fusion':metrics(fuse_graph(candidate)),
                                             'channel_steps_cumulative':current.channel_steps.tolist()})
                    update(status='training-'+name)
                    log(f'{name} epoch {stats.epoch}/20; fused Recall100={record["epochs"][-1]["c1_fusion"]["recall_at_100"]:.6f}')
                    return False
                fit_started = time.perf_counter()
                model.fit(split.train,callback=callback)
                model.save(run/(name+'.npz'))
                raw = model.recommend(user_ids,excluded,100)
                candidate = {q.query_id:raw[q.user_id] for q in queries}
                loaded = AspectGraphLightGCN.load(run/(name+'.npz'))
                assert loaded.recommend(user_ids,excluded,100) == raw
                assert len(model.history) == 20
                record.update(seconds=time.perf_counter()-fit_started,epochs_completed=len(model.history),
                    bpr_edges=model.edge_count,optimizer_steps=sum(v['alignment']['batches'] for v in record['epochs']),
                    projector_parameters=model.projector_parameter_count,channel_steps=model.channel_steps.tolist(),
                    checkpoint_reload_exact=True,final=model.alignment_history[-1].epoch)
                arms[name] = candidate
                arms[name+'_c1'] = fuse_graph(candidate)
                save_ranks()
                del model,loaded
                gc.collect()
        validate_rankings(queries,arms,catalog,100)
        state['metrics'] = {name:metrics(ranks) for name,ranks in arms.items()}
        changes,paired = {},{}
        comparisons = [('opinion_pooled_control_c1','opinion_channels_c1'),
                       ('opinion_channels_shuffled_c1','opinion_channels_c1')]
        for view in ('text','graph'):
            for suffix in ('','_c1','_quota'):
                comparisons.append((f'interests_{view}_k1'+suffix,f'interests_{view}_k3'+suffix))
        comparisons += [('c5',name) for name in arms if name not in ('c5','c4')]
        for before,after in comparisons:
            paired[after+'_minus_'+before] = {metric:paired_change(queries,arms[before],arms[after],cut,kind)
                 for metric,cut,kind in (('recall_at_100',100,'recall'),('ndcg_at_10',10,'ndcg'))}
        for name,ranks in arms.items():
            if name in ('c5','c4'): continue
            changes[name],per_query = candidate_changes(queries,c5,ranks)
            write_jsonl(run/(name+'.candidate_changes.jsonl'),per_query)
        state.update(paired_comparisons=paired,candidate_changes_vs_c5=changes,
                     ranker_hashes_after=protect_ranker(args.baseline_run))
        assert state['ranker_hashes_after'] == protected
        state['ranker_unchanged'] = True
        state['limitations'] = ['Validation reused; no independent final holdout or final ranker metric.',
            'Conservative local rules are weak labels, not LLM extraction or human-validated sentiment.',
            'Neutral/unknown opinions preserved in extraction, excluded from positive/negative alignment.',
            'Method2 uses frozen graph and unsupervised text clusters, not neural ComiRec training.',
            'Multi-interest inference uses per-interest lists then round-robin; no score mean across interests.',
            'Quota preserves C5 first90; equal Top10 metrics follow structurally.',
            'Method1 static graph fit uses all T1 visits; not historical sequence prediction.',
            'Method1 capacity-matched pooled control preserves observed channel masks/graph heads; opinion-specific contents differ.']
        save_ranks()
        state['artifacts_sha256'] = {path.name:digest_file(path) for path in run.iterdir()
                                     if path.is_file() and path.suffix in ('.npz','.jsonl')}
        update(status='complete',seconds=time.perf_counter()-started)
        write_json(run/'metrics.json',state)
        log(json.dumps({'run':str(run),'metrics':{name:{k:value[k] for k in ('recall_at_100','ndcg_at_10')}
                        for name,value in state['metrics'].items()},'seconds':state['seconds']},ensure_ascii=False))
        return run
    except BaseException as error:
        update(status='failed',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        raise


def console_main():
    main()


if __name__ == '__main__':
    console_main()
