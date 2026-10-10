"""Apply a frozen historical RQ-VAE to every text-bearing Supabase review.

The database is read only; later reviews are encoded without model updates.
No review bodies, user names or database credentials are printed or persisted.
"""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))

import numpy as np
import psycopg
from psycopg.rows import dict_row
import torch

from rating_recsys.config import get_settings
from rating_recsys.experiments.rqvae_cli import VERSION, file_hash, prepare_embeddings, write_json
from rating_recsys.experiments.snapshot import load_review_texts, load_snapshot, review_texts_path
from rating_recsys.retrieval.rqvae import ReviewRQVAE,RQVAEConfig

SQL = '''SELECT review_id, review_text,
       COALESCE(reviewed_at, (scraped_at AT TIME ZONE 'Asia/Seoul')::date) AS event_date
FROM recsys.reviews ORDER BY review_id'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    metrics=json.loads((args.run_dir/'metrics.json').read_text())
    if metrics['version']!=VERSION or metrics['status']!='complete':
        raise ValueError('A complete compatible training run is required')
    checkpoint=torch.load(args.run_dir/'model.pt',map_location='cpu',weights_only=True)
    config=RQVAEConfig(**checkpoint['config'])
    model=ReviewRQVAE(config);model.load_state_dict(checkpoint['state_dict']);model.eval()
    torch.set_num_threads(args.threads)
    frozen_model_hash=file_hash(args.run_dir/'model.pt')
    settings=get_settings(require_database=True,require_user_hash_salt=False)
    records=[]
    with psycopg.connect(settings.database_url,connect_timeout=10,prepare_threshold=None) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            cursor.execute("SET LOCAL statement_timeout='45s'")
            cursor.execute("SELECT transaction_timestamp() AS observed_at, current_setting('transaction_read_only') AS readonly, current_setting('transaction_isolation') AS isolation")
            source=cursor.fetchone()
            cursor.execute('SELECT count(*) AS reviews FROM recsys.reviews'); count=cursor.fetchone()['reviews']
        with connection.cursor(name='frozen_review_codes',row_factory=dict_row) as cursor:
            cursor.execute(SQL)
            while batch:=cursor.fetchmany(20000):records.extend(batch)
        connection.rollback()
    assert source['readonly']=='on' and source['isolation']=='repeatable read'
    assert len(records)==count
    texts={row['review_id']:row['review_text'] for row in records}
    rows=[SimpleNamespace(review_id=row['review_id'],event_date=row['event_date']) for row in records]
    source_hash=hashlib.sha256()
    for record in records:
        source_hash.update(json.dumps(record,ensure_ascii=False,sort_keys=True,default=lambda d:d.isoformat(),separators=(',',':')).encode()+b'\n')
    del records
    snapshot=PROJECT_ROOT/'artifacts/snapshots/e7896add5b4b5939.jsonl'
    old_rows=tuple(load_snapshot(snapshot))
    old_texts,_=load_review_texts(review_texts_path(snapshot),old_rows)
    old_folder=PROJECT_ROOT/'artifacts/rqvae_review_embeddings'
    old_meta=json.loads((old_folder/'manifest.json').read_text())
    assert old_meta['identity']==metrics['embedding_metadata']['identity']
    assert file_hash(old_folder/'reviews.npz')==old_meta['arrays_sha256']
    with np.load(old_folder/'reviews.npz',allow_pickle=False) as archive:
        input_digest=hashlib.sha256()
        for key in archive['review_ids']:
            input_text='passage: '+old_texts[int(key)].strip()
            input_digest.update(str(int(key)).encode()+b'\0'+input_text.encode()+b'\n')
        assert input_digest.hexdigest()==old_meta['identity']['input_sha256']
        reuse={'passage: '+old_texts[int(key)].strip():value for key,value in zip(archive['review_ids'],archive['vectors'])}
    review_rows,vectors,meta=prepare_embeddings(rows,texts,PROJECT_ROOT/'artifacts/rqvae_all_db_embeddings',
            model_cache=PROJECT_ROOT/'artifacts/rqvae_e5_model',download=False,threads=args.threads,reuse_vectors=reuse)
    decoded,_,codes=model.representations(vectors)
    dtype=np.uint8 if config.codebook_size<=256 else np.uint16 if config.codebook_size<=65536 else np.uint32
    destination=args.run_dir/'supabase_review_codes.npz'
    np.savez_compressed(destination,review_ids=np.asarray([row.review_id for row in review_rows],dtype=np.int64),codes=codes.astype(dtype))
    assert file_hash(args.run_dir/'model.pt')==frozen_model_hash
    # Check decode from saved codes without original embedding input.
    with np.load(destination,allow_pickle=False) as archive:
        actual=model.decode_codes(torch.as_tensor(archive['codes'][:100].astype(np.int64))).detach().numpy()
    assert np.allclose(actual,decoded[:100],rtol=1e-5,atol=1e-6)
    unique=len({tuple(code) for code in codes.tolist()})
    representatives={}
    for index,row in enumerate(review_rows):
        representatives.setdefault(texts[row.review_id].strip(),index)
    distinct_text_codes=codes[list(representatives.values())]
    result={'observed_at':source['observed_at'].isoformat(),'transaction':{k:source[k] for k in ('readonly','isolation')},
            'source':'Supabase recsys.reviews','sql':SQL,'source_rows_sha256':source_hash.hexdigest(),
            'db_reviews':count,'text_reviews_encoded':len(review_rows),'reviews_without_text':count-len(review_rows),
            'quantizer_training_cutoff':metrics['train_cutoff'],'quantizer_model_sha256':frozen_model_hash,
            'quantizer_updates_during_application':0,'unique_codes':unique,'collision_excess_reviews':len(codes)-unique,
            'distinct_full_review_texts':len(representatives),
            'collision_excess_distinct_full_texts':len(representatives)-len({tuple(code) for code in distinct_text_codes.tolist()}),
            'codes_file':destination.name,'codes_sha256':file_hash(destination),
            'embedding_metadata':meta,'implementation_sha256':file_hash(__file__),
            'note':'Later reviews receive codes without training; these codes are excluded from earlier validation inputs.'}
    write_json(args.run_dir/'supabase_application.json',result)
    print(json.dumps({k:result[k] for k in ('observed_at','db_reviews','text_reviews_encoded','reviews_without_text','unique_codes','quantizer_updates_during_application')},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'error_type':type(error).__name__,'sqlstate':getattr(error,'sqlstate',None)}),file=sys.stderr)
        raise SystemExit(1) from None
