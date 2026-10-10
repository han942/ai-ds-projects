"""Train review semantic codes and measure validation-only quantization effects.

Requires the local-embeddings extra and scikit-learn. Explicit --download-model
permits fetching pinned public E5 weights. No paid API or DB writes are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings, query_scores
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import dataset_digest, load_snapshot, load_review_texts, review_texts_path
from rating_recsys.retrieval.review_embeddings import E5_MODEL, E5_REVISION
from rating_recsys.retrieval.rqvae import ReviewRQVAE, RQVAEConfig

VERSION = 'review-rqvae-v1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def prepare_embeddings(history, texts, folder, *, model_cache, download, threads, batch_size=64, reuse_vectors=None):
    """Encode every nonempty historical review; deduplicate identical inputs.

    One passage-role embedding per whole review, capped at E5's 512 tokens.
    This is a new input policy; it is not the earlier E5 block cache.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rows = sorted((row for row in history if (texts[row.review_id] or '').strip()), key=lambda r:r.review_id)
    inputs = ['passage: ' + texts[row.review_id].strip() for row in rows]
    input_digest = hashlib.sha256()
    for row, text in zip(rows, inputs):
        input_digest.update(str(row.review_id).encode() + b'\0' + text.encode() + b'\n')
    identity = {'version':VERSION, 'model':E5_MODEL, 'revision':E5_REVISION,
                'role':'passage', 'max_tokens':512, 'dtype':'float32', 'normalization':'L2',
                'input_sha256':input_digest.hexdigest(), 'reviews':len(rows)}
    npz, manifest = folder/'reviews.npz', folder/'manifest.json'
    if manifest.exists() and npz.exists():
        metadata = json.loads(manifest.read_text())
        if metadata['identity'] != identity or metadata['arrays_sha256'] != file_hash(npz):
            raise ValueError('Embedding cache identity/checksum mismatch')
        with np.load(npz, allow_pickle=False) as data:
            if data['review_ids'].tolist() != [row.review_id for row in rows]:
                raise ValueError('Embedding cache review order differs')
            return rows, data['vectors'], metadata
    from huggingface_hub import snapshot_download, list_repo_files
    from sentence_transformers import SentenceTransformer
    if download:
        names = list_repo_files(E5_MODEL, revision=E5_REVISION)
        weight = 'model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        model_directory = snapshot_download(E5_MODEL, revision=E5_REVISION, cache_dir=str(model_cache),
                                             allow_patterns=['*.json', weight])
    else:
        model_directory = snapshot_download(E5_MODEL, revision=E5_REVISION,
                                             cache_dir=str(model_cache), local_files_only=True)
    torch.set_num_threads(threads)
    encoder = SentenceTransformer(model_directory, device='cpu', trust_remote_code=False,
                                  local_files_only=True, model_kwargs={'torch_dtype':torch.float32})
    encoder.max_seq_length = 512
    if encoder.get_sentence_embedding_dimension() != 384:
        raise ValueError('Unexpected frozen E5 dimension')
    unique = list(dict.fromkeys(inputs))
    token_lengths = np.asarray([len(ids) for ids in encoder.tokenizer(unique, truncation=False,
                                  add_special_tokens=True)['input_ids']])
    # Length sorting cuts padding costs; retain deterministic input order afterward.
    order = np.argsort(token_lengths, kind='stable')
    staging = folder/'vectors.partial.npy'
    partial_manifest = folder/'partial.json'
    offset = 0
    vectors = np.lib.format.open_memmap(staging, mode='r+' if staging.exists() else 'w+',
                                        dtype='float32', shape=(len(unique),384))
    if partial_manifest.exists():
        partial = json.loads(partial_manifest.read_text())
        if partial['identity'] != identity:
            raise ValueError('Partial embedding identity differs')
        offset = partial['offset']
    started = time.perf_counter()
    reuse_vectors = reuse_vectors or {}
    reused_inputs = sum(text in reuse_vectors for text in unique)
    for start in range(offset, len(order), batch_size):
        indices = order[start:start+batch_size]
        missing = [int(index) for index in indices if unique[int(index)] not in reuse_vectors]
        for index in indices:
            if unique[int(index)] in reuse_vectors:
                vectors[int(index)] = reuse_vectors[unique[int(index)]]
        if missing:
            embedded = encoder.encode([unique[index] for index in missing], batch_size=batch_size,
                                    convert_to_numpy=True, normalize_embeddings=True,
                                    show_progress_bar=False)
            vectors[missing] = embedded
        vectors.flush()
        write_json(partial_manifest, {'identity':identity, 'offset':min(start+batch_size,len(order))})
        if start == offset or (start//batch_size) % 20 == 0:
            print(json.dumps({'stage':'E5', 'completed_unique':min(start+batch_size,len(order)),
                              'total_unique':len(order), 'seconds':round(time.perf_counter()-started,1)}), flush=True)
    lookup = {text:index for index,text in enumerate(unique)}
    mapped = np.asarray(vectors[[lookup[text] for text in inputs]],dtype=np.float32)
    if not np.isfinite(mapped).all() or not np.allclose(np.linalg.norm(mapped,axis=1),1,atol=1e-4):
        raise ValueError('Invalid frozen embeddings')
    np.savez_compressed(npz, review_ids=np.asarray([row.review_id for row in rows],dtype=np.int64), vectors=mapped)
    metadata = {'identity':identity, 'arrays_sha256':file_hash(npz), 'unique_inputs':len(unique),
                'reused_unique_inputs':reused_inputs,
                'over_512_token_unique_inputs':int((token_lengths>512).sum()),
                'unique_token_length_quantiles':dict(zip(('p50','p90','p95','p99'),np.quantile(token_lengths,[.5,.9,.95,.99]).tolist())),
                'seconds_this_invocation':time.perf_counter()-started, 'api_calls':0,
                'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','sentence-transformers')}}
    write_json(manifest,metadata)
    del vectors
    return rows,mapped,metadata


def split_review_groups(rows, texts, seed=42, effective_group_keys=None):
    """Identical effective encoder inputs stay together across fit/audit."""
    hashes = (effective_group_keys if effective_group_keys is not None else
              [hashlib.sha256(texts[row.review_id].strip().encode()).hexdigest() for row in rows])
    if len(hashes) != len(rows):
        raise ValueError('Encoder group key count differs from reviews')
    unique = sorted(set(hashes))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    held = {unique[int(index)] for index in shuffled[:max(1,len(unique)//10)]}
    audit = np.asarray([index for index,key in enumerate(hashes) if key in held],dtype=np.int64)
    train = np.asarray([index for index,key in enumerate(hashes) if key not in held],dtype=np.int64)
    if not len(train) or not len(audit):
        raise ValueError('Need nonempty representation fit/audit subsets')
    return train,audit


def fit_model(vectors, train_indices, audit_indices, config, directory):
    torch.manual_seed(config.seed); torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    model = ReviewRQVAE(config)
    tensor = torch.from_numpy(vectors)
    model.set_scaling(tensor[train_indices])
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    history = []
    def epoch(number, quantize):
        model.train(); totals = np.zeros(3,dtype=float)
        for batch in np.array_split(rng.permutation(train_indices),max(1,int(np.ceil(len(train_indices)/config.batch_size)))):
            optimizer.zero_grad(set_to_none=True)
            _,_,recon,rq = model(tensor[batch],quantize=quantize)
            loss = recon+rq
            if not torch.isfinite(loss):raise ValueError('Nonfinite RQ-VAE loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            totals += np.asarray([float(recon.detach()),float(rq.detach()),float(loss.detach())])*len(batch)
        model.eval()
        with torch.no_grad():
            rec,_,_,_ = model(tensor[audit_indices],quantize=quantize)
            audit_mse = float((rec-tensor[audit_indices]).square().mean())
        row = {'epoch':number, 'quantize':quantize, 'reconstruction_loss':totals[0]/len(train_indices),
               'quantization_loss':totals[1]/len(train_indices),'total_loss':totals[2]/len(train_indices),
               'audit_original_space_mse':audit_mse}
        history.append(row)
        print(json.dumps({'stage':'RQ-VAE',**row}),flush=True)
        write_json(directory/'learning_curve.json',history)
    for index in range(config.warmup_epochs):epoch(index+1,False)
    model.eval()
    with torch.no_grad():
        subset = train_indices[rng.choice(len(train_indices),size=min(len(train_indices),10000),replace=False)]
        latent = model.encoder(model.scaled(tensor[subset]))
        with threadpool_limits(limits=1):model.quantizer.initialize(latent,config.seed)
    # Fixed epoch budget, no recommendation validation used for checkpoint selection.
    for index in range(config.epochs):epoch(index+1,True)
    torch.save({'version':VERSION,'config':config.to_dict(),'state_dict':model.state_dict()},directory/'model.pt')
    return model,history


def representation_metrics(raw, reconstructed, continuous, codes, train_indices, audit_indices, config, model):
    result = {}
    mean = raw[train_indices].mean(0)
    for name,indices in (('fit',train_indices),('heldout_text_groups',audit_indices)):
        target = raw[indices]
        baseline = float(np.mean((target-mean)**2))
        values = {'reviews':len(indices),'train_mean_mse':baseline,
                  'train_mean_cosine':float(np.sum(unit(target)*unit(mean),axis=1).mean())}
        for label,vectors in (('rq_decoded',reconstructed),('rq_continuous_path',continuous)):
            predicted = vectors[indices]
            mse = float(np.mean((predicted-target)**2))
            values[label] = {'mse':mse,'mse_vs_train_mean':mse/max(baseline,1e-12),
                             'cosine_to_original_mean':float(np.sum(unit(predicted)*unit(target),axis=1).mean())}
        result[name] = values
    result['codebook_usage'] = []
    for level in range(config.levels):
        counts = np.bincount(codes[:,level],minlength=config.codebook_size)
        probabilities = counts[counts>0]/len(codes)
        result['codebook_usage'].append({'level':level,'used':int((counts>0).sum()),
                 'capacity':config.codebook_size,'perplexity':float(np.exp(-np.sum(probabilities*np.log(probabilities)))),
                 'largest_share':float(counts.max()/len(codes))})
    for depth in range(1,config.levels+1):
        counts = Counter(map(tuple,codes[:,:depth].tolist()))
        result[f'prefix_{depth}'] = {'unique':len(counts),'excess_collision_reviews':len(codes)-len(counts),
                     'reviews_in_shared_codes':sum(n for n in counts.values() if n>1),'largest_bucket':max(counts.values())}
    result['prefix_geometry'] = []
    model.eval()
    with torch.no_grad():
        latent = model.encoder(model.scaled(torch.as_tensor(raw[audit_indices])))
        selected = torch.as_tensor(codes[audit_indices],dtype=torch.int64)
        quantized = torch.zeros_like(latent)
        denominator = float(latent.square().sum(1).mean())
        for level,book in enumerate(model.quantizer.books):
            quantized += book(selected[:,level])
            residual = float((latent-quantized).square().sum(1).mean())
            partial = (model.decoder(quantized)*model.input_scale+model.input_mean).numpy()
            result['prefix_geometry'].append({'depth':level+1,'heldout_latent_residual_squared_norm':residual,
                                              'residual_over_unquantized_latent_norm':residual/max(denominator,1e-12),
                                              'heldout_prefix_decoded_mse':float(np.mean((partial-raw[audit_indices])**2))})
    # Held-out queries, fitted reference pool; stable ordering and self/text-group exclusion.
    rng = np.random.default_rng(config.seed)
    references = train_indices[rng.choice(len(train_indices),min(10000,len(train_indices)),replace=False)]
    queries = audit_indices[rng.choice(len(audit_indices),min(1000,len(audit_indices)),replace=False)]
    raw_ref = unit(raw[references]); raw_q = unit(raw[queries])
    neighbour = {'queries':len(queries),'reference_reviews':len(references),'k':10}
    raw_top = np.argsort(-(raw_q@raw_ref.T),axis=1,kind='stable')[:,:10]
    for name,vectors in (('rq_decoded',reconstructed),('rq_continuous_path',continuous)):
        pred = unit(vectors[queries])@unit(vectors[references]).T
        top = np.argsort(-pred,axis=1,kind='stable')[:,:10]
        overlap = [len(set(a)&set(b))/10 for a,b in zip(raw_top,top)]
        neighbour[name] = {'original_top10_overlap':float(np.mean(overlap)),
                          'score_mae':float(np.abs(pred-raw_q@raw_ref.T).mean())}
    result['nearest_neighbour_preservation'] = neighbour
    parameters = sum(value.numel()*value.element_size() for value in model.state_dict().values())
    code_dtype_bytes = 1 if config.codebook_size<=256 else 2 if config.codebook_size<=65536 else 4
    result['storage'] = {'reviews':len(raw),'float32_dense_bytes':raw.nbytes,
                        'codes_bytes':len(raw)*config.levels*code_dtype_bytes,
                        'model_state_bytes':parameters,
                        'codes_plus_model_bytes':len(raw)*config.levels*code_dtype_bytes+parameters,
                        'dense_over_codes_and_model':float(raw.nbytes/(len(raw)*config.levels*code_dtype_bytes+parameters)),
                        'note':'tensor payload; excludes IDs, Python/NPZ/container overhead, code lookup and any retained raw vector cache'}
    return result


def profile_banks(history, review_rows, vectors, user_limit=20, item_limit=30):
    vectors = unit(vectors)
    mapping = {row.review_id:index for index,row in enumerate(review_rows)}
    users,items = defaultdict(lambda:deque(maxlen=user_limit)),defaultdict(lambda:deque(maxlen=item_limit))
    for row in sorted(history,key=lambda r:(r.event_date,r.review_id)):
        index = mapping.get(row.review_id)
        if index is not None:
            users[row.user_id].append(index);items[row.restaurant_id].append(index)
    def pool(groups):
        keys = sorted(groups)
        return keys,unit(np.stack([vectors[list(groups[key])].mean(0) for key in keys]))
    return pool(users),pool(items)


def paired_change(before, after, seed=42, samples=2000):
    differences = np.asarray(after)-np.asarray(before)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for sample in range(samples):means[sample]=differences[rng.integers(0,len(differences),len(differences))].mean()
    return {'mean_difference':float(differences.mean()), 'bootstrap_95pct':np.quantile(means,[.025,.975]).tolist(),
            'paired_users':len(differences),'samples':samples}


def recommendation_metrics(history, review_rows, representations, queries, c5_ordered, threads=4):
    catalog = sorted({row.restaurant_id for row in history})
    popularity = Counter(row.restaurant_id for row in history)
    popular = sorted(catalog,key=lambda key:(-popularity[key],key))
    seen = defaultdict(set)
    for row in history:seen[row.user_id].add(row.restaurant_id)
    all_rankings = {'c5':c5_ordered}
    timing,coverage = {},{}
    for name,vectors in representations.items():
        started = time.perf_counter()
        (user_keys,user_bank),(item_keys,item_bank) = profile_banks(history,review_rows,vectors)
        user_lookup = dict(zip(user_keys,user_bank));item_lookup = dict(zip(item_keys,range(len(item_keys))))
        coverage[name] = {'users_with_text':len(user_keys),'items_with_text':len(item_keys),
                          'query_users_with_text':sum(q.user_id in user_lookup for q in queries),'catalog':len(catalog)}
        dense,rerank,fused = {},{},{}
        with threadpool_limits(limits=threads):
            for q in queries:
                banned = seen[q.user_id]
                fallback = [key for key in popular if key not in banned]
                scores = user_lookup[q.user_id]@item_bank.T if q.user_id in user_lookup else np.zeros(len(item_keys),dtype=np.float32)
                eligible = [index for index,key in enumerate(item_keys) if key not in banned]
                ordered_indices = sorted(eligible,key=lambda index:(-float(scores[index]),item_keys[index])) if q.user_id in user_lookup else []
                ids = [item_keys[index] for index in ordered_indices[:100]]
                dense[q.query_id] = tuple((ids+[key for key in fallback if key not in ids])[:100])
                base = tuple(c5_ordered[q.query_id])
                # Preserve original C5 order for missing user/item text; no fake zero similarity.
                reordered = list(base)
                if q.user_id in user_lookup:
                    slots = [index for index,key in enumerate(base) if key in item_lookup]
                    covered = sorted((base[index] for index in slots),key=lambda key:(-float(scores[item_lookup[key]]),base.index(key)))
                    for index,key in zip(slots,covered):
                        reordered[index] = key
                rerank[q.query_id] = tuple(reordered)
                votes = defaultdict(float)
                for ordered in (base,dense[q.query_id]):
                    for position,key in enumerate(ordered,1):votes[key]+=1/(60+position)
                fused[q.query_id] = tuple(sorted(votes,key=lambda key:(-votes[key],key))[:100])
        all_rankings[f'{name}_dense'] = dense
        all_rankings[f'{name}_c5_text_order'] = rerank
        all_rankings[f'{name}_c5_rrf'] = fused
        timing[name] = time.perf_counter()-started
    observations = {}
    metrics = {}
    per_user = {}
    for name,rankings in all_rankings.items():
        data = [RankingObservation(q.query_id,q.user_id,q.relevance_by_item,rankings[q.query_id]) for q in queries]
        observations[name] = data
        metrics[name] = evaluate_rankings(data,cutoffs=(10,100),catalog_ids=catalog,item_popularity=dict(popularity))
        per_user[name] = [query_scores(row,10)['ndcg'] for row in data if row.relevant_items]
    comparisons = {name:paired_change(per_user['raw_dense' if name.endswith('_dense') else 'raw_c5_text_order' if name.endswith('_c5_text_order') else 'raw_c5_rrf'],values)
                   for name,values in per_user.items() if name.startswith('rq_')}
    return {'metrics':metrics,'paired_ndcg10_vs_same_raw_path':comparisons,'profile_coverage':coverage,
            'paired_ndcg10_vs_c5':{name:paired_change(per_user['c5'],values) for name,values in per_user.items() if name!='c5'},
            'profile_and_ranking_seconds':timing,'note':'validation exploration; C5 is pre-LambdaRank candidate order; c5_text_order is cosine reordering, not a trained LambdaRank'},all_rankings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=PROJECT_ROOT/'artifacts/snapshots/e7896add5b4b5939.jsonl')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'artifacts/comparisons/rqvae')
    parser.add_argument('--model-cache',type=Path,default=PROJECT_ROOT/'artifacts/rqvae_e5_model')
    parser.add_argument('--embedding-cache',type=Path,default=PROJECT_ROOT/'artifacts/rqvae_review_embeddings')
    parser.add_argument('--prepared-validation',type=Path,default=PROJECT_ROOT/'artifacts/prepared/e7896add5b4b5939/validation-22c99d8506aab250')
    parser.add_argument('--download-model',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--warmup-epochs',type=int,default=5)
    parser.add_argument('--codebook-size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--embedding-batch-size',type=int,default=64)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    rows = tuple(load_snapshot(args.snapshot))
    split = build_global_temporal_split(rows)
    texts,text_metadata = load_review_texts(review_texts_path(args.snapshot),rows)
    digest = dataset_digest(rows)
    config = ExperimentConfig(satisfaction_mode='history-aware',ranker_training_mode='window',n_jobs=args.threads)
    queries,new_users = build_window_queries(split.train,split.validation,config=config,phase='validation',cutoff=split.train_cutoff)
    directory = args.output_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'progress.json',{'stage':'preparing','snapshot':digest,'cutoff':split.train_cutoff.isoformat()})
    review_rows,vectors,embedding_metadata = prepare_embeddings(split.train,texts,args.embedding_cache,
                                   model_cache=args.model_cache,download=args.download_model,threads=args.threads,batch_size=args.embedding_batch_size)
    if args.prepare_only:
        write_json(directory/'progress.json',{'stage':'prepared','reviews':len(review_rows),'embedding_cache':str(args.embedding_cache)})
        print(json.dumps({'stage':'prepared','reviews':len(review_rows),'metadata':embedding_metadata}),flush=True);return
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    model_directory = snapshot_download(E5_MODEL,revision=E5_REVISION,cache_dir=str(args.model_cache),local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
    tokenized = tokenizer(['passage: '+texts[row.review_id].strip() for row in review_rows],truncation=True,max_length=512)['input_ids']
    effective_keys = [hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() for ids in tokenized]
    train,audit = split_review_groups(review_rows,texts,effective_group_keys=effective_keys)
    assert {effective_keys[int(index)] for index in train}.isdisjoint({effective_keys[int(index)] for index in audit})
    rq_config = RQVAEConfig(epochs=args.epochs,warmup_epochs=args.warmup_epochs,codebook_size=args.codebook_size,threads=args.threads)
    fit_started = time.perf_counter()
    model,curve = fit_model(vectors,train,audit,rq_config,directory)
    fit_seconds = time.perf_counter()-fit_started
    decode_started = time.perf_counter()
    decoded,continuous,codes = model.representations(vectors)
    encode_decode_seconds = time.perf_counter()-decode_started
    storage_dtype = np.uint8 if args.codebook_size<=256 else np.uint16 if args.codebook_size<=65536 else np.uint32
    np.savez_compressed(directory/'review_codes.npz',review_ids=np.asarray([row.review_id for row in review_rows],dtype=np.int64),
                        codes=codes.astype(storage_dtype),train_indices=train,audit_indices=audit)
    repr_metrics = representation_metrics(vectors,decoded,continuous,codes,train,audit,rq_config,model)
    representatives = {}
    for index,key in enumerate(effective_keys):
        representatives.setdefault(key,index)
    unique_input_codes = codes[list(representatives.values())]
    repr_metrics['distinct_encoder_input_collisions'] = {
        'encoder_inputs':len(unique_input_codes),
        'unique_full_codes':len({tuple(code) for code in unique_input_codes.tolist()}),
        'meaning':'Different effective token inputs sharing codes, not independently annotated semantic collisions',
    }
    prepared = json.loads((args.prepared_validation/'manifest.json').read_text())
    if prepared['identity']['snapshot_id'] not in (digest,digest[:16]) or prepared['identity']['cutoffs'] != split.summary()['cutoffs']:
        raise ValueError('Prepared validation snapshot/cutoff differs')
    if prepared['identity']['satisfaction']['satisfaction_mode'] != config.satisfaction_mode:
        raise ValueError('Prepared labels differ')
    c5 = prepared['data']['ordered']['c5_c1_lightgcn_rrf']
    if set(c5) != {q.query_id for q in queries}:raise ValueError('Prepared validation query population differs')
    if prepared['identity']['candidate_k'] != config.candidate_k or prepared['identity']['rrf_constant'] != config.rrf_constant:
        raise ValueError('Prepared retrieval settings differ')
    catalog = {row.restaurant_id for row in split.train}
    for query in queries:
        candidates = c5[query.query_id]
        visited = {row.restaurant_id for row in query.history}
        if len(candidates)!=len(set(candidates)) or set(candidates)-catalog or set(candidates)&visited:
            raise ValueError('Invalid prepared C5 catalog/duplicate/visited candidate')
    metrics,rankings = recommendation_metrics(split.train,review_rows,{'raw':vectors,'rq_decoded':decoded,'rq_continuous':continuous},queries,c5,threads=args.threads)
    write_json(directory/'rankings_validation.json',rankings)
    np.savez_compressed(directory/'representations.npz',raw=vectors,decoded=decoded,continuous=continuous)
    result = {'version':VERSION,'status':'complete','snapshot':digest,'snapshot_sha256':file_hash(args.snapshot),
              'review_sidecar':text_metadata,'train_cutoff':split.train_cutoff.isoformat(),
              'prepared_validation_manifest_sha256':file_hash(args.prepared_validation/'manifest.json'),
              'validation_through':split.validation_cutoff.isoformat(),'test_evaluated':False,
              'training_reviews':len(review_rows),'training_entities':{'users':len({r.user_id for r in split.train}),'items':len({r.restaurant_id for r in split.train})},
              'representation_fit_reviews':len(train),'representation_audit_reviews':len(audit),
              'representation_split_grouping':'exact E5 token IDs after passage prefix and 512-token truncation',
              'effective_input_groups':len(set(effective_keys)),
              'model_config':rq_config.to_dict(),'recommendation_config':config.to_dict(),
              'embedding_metadata':embedding_metadata,'representation':repr_metrics,'recommendation':metrics,
              'new_validation_users_excluded':new_users,'seconds':time.perf_counter()-started,
              'model_fit_seconds':fit_seconds,'rq_encode_decode_both_paths_seconds':encode_decode_seconds,
              'source_sha256':{str(path.relative_to(PROJECT_ROOT)):file_hash(path) for path in (Path(__file__),PROJECT_ROOT/'src/rating_recsys/retrieval/rqvae.py')},
              'artifacts_sha256':{name:file_hash(directory/name) for name in ('model.pt','review_codes.npz','representations.npz','rankings_validation.json','learning_curve.json')},
              'limitations':['Review codes are not restaurant IDs; collisions are retained rather than mislabeled as unique IDs.',
                             'This adapts RQ-VAE to whole-review E5 passage embeddings, not a full TIGER model or the earlier RLMRec block policy.',
                             'Representation audit holds out identical text groups within past reviews; this is not future recommendation evaluation.',
                             'Continuous path uses the same RQ-trained encoder/decoder with quantization bypassed, not an independently trained AE.',
                             'Recommendation validation is exploratory and contains no newly independent holdout.',
                             'Stored model/code/raw arrays mean runtime disk usage is larger than a hypothetical codes-only deployment.']}
    write_json(directory/'metrics.json',result);write_json(directory/'progress.json',{'stage':'complete','seconds':result['seconds']})
    print(json.dumps({'output_dir':str(directory),'representation':repr_metrics,'recommendation':metrics},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
