"""Independent numerical replay and text-only spot checks for aspect audits."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from rating_recsys.experiments.review_aspect_audit import ASPECTS, LABELS, write_json


def independent_pairs(rows, field):
    valid = [r for r in rows if r[field] is not None]
    y = np.array([r['rating'] for r in valid])
    x = np.array([r[field] for r in valid])
    upper = np.triu(np.ones((len(valid), len(valid)), dtype=bool), 1)
    delta_y = y[:, None]-y[None, :]
    products = (x[:, None]-x[None, :])*delta_y
    mask = upper & (delta_y != 0)
    n = int(mask.sum())
    return {'pairs':n, 'correct':int(((products > 0)&mask).sum()),
            'ties':int(((products == 0)&mask).sum()), 'wrong':int(((products < 0)&mask).sum())}


def independent_rho(rows, field):
    valid = [r for r in rows if r[field] is not None]
    x = rankdata([r[field] for r in valid])
    y = rankdata([r['rating'] for r in valid])
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None


def verify_number(actual, expected):
    assert actual is None and expected is None or actual is not None and expected is not None and abs(actual-expected) < 1e-12


def verify_source_evidence(rows, input_rows):
    """Replay provenance without the extraction resolver; support v3/v4."""
    sources = {r['review_id']:r for r in input_rows}
    assert len(sources) == len(input_rows) == len(rows) == len({r['review_id'] for r in rows})
    assert set(sources) == {r['review_id'] for r in rows}
    quote_count = 0
    for r in rows:
        source = sources[r['review_id']]
        assert set(source) in ({'review_id','text'}, {'review_id','text','sentences'})
        text = source['text']
        assert r['extraction_text'] == text
        sentences = source.get('sentences')
        if sentences is not None:
            previous_end = 0
            for expected_id, sentence in enumerate(sentences, 1):
                assert set(sentence) == {'sentence_id','start','end','text'}
                assert type(sentence['sentence_id']) is int and sentence['sentence_id'] == expected_id
                start, end = sentence['start'], sentence['end']
                assert type(start) is int and type(end) is int and previous_end <= start < end <= len(text)
                assert not text[previous_end:start].strip()
                assert sentence['text'] == text[start:end]
                previous_end = end
            assert not text[previous_end:].strip()
        evidence = [(a['evidence'], a.get('evidence_sentence_ids')) for a in r['aspects']]
        assert all(1 <= len(quotes) <= 2 for quotes, _ in evidence)
        assert (r['overall'] is None and r['overall_evidence'] == [] or
                r['overall'] is not None and 1 <= len(r['overall_evidence']) <= 2)
        evidence.append((r['overall_evidence'], r.get('overall_evidence_sentence_ids')))
        for quotes, ids in evidence:
            if sentences is not None:
                assert isinstance(ids, list) and len(ids) == len(quotes) <= 2
                assert all(type(i) is int and 1 <= i <= len(sentences) for i in ids)
                assert len(set(ids)) == len(ids)
                assert quotes == [text[sentences[i-1]['start']:sentences[i-1]['end']] for i in ids]
            for quote in quotes:
                assert quote and quote in text and '[별점표현 제거]' not in quote
                quote_count += 1
    return quote_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args = parser.parse_args()
    folder = args.folder
    rows = [json.loads(x) for x in (folder/'extracted_reviews.jsonl').open()]
    result = json.loads((folder/'results.json').read_text())
    input_rows = list(map(json.loads,(folder/'extraction_input.jsonl').open()))
    quote_count = verify_source_evidence(rows, input_rows)
    assert len(rows) == result['reviews']
    for r in rows:
        axes = [a['sentiment'] for a in r['aspects'] if a['aspect'] != 'revisit']
        assert r['aspect_mean'] == (sum(axes)/len(axes) if axes else None)
    for u in result['users']:
        subset = [r for r in rows if r['user_id']==u['user_id']]
        assert len(subset) == u['text_reviews']
        for field, measured in u['metrics'].items():
            counts = independent_pairs(subset,field)
            for key, value in counts.items():
                assert measured['pair'][key] == value
            if counts['pairs']:
                assert abs(measured['pair']['concordance']-(counts['correct']+.5*counts['ties'])/counts['pairs']) < 1e-12
            rho = independent_rho(subset,field)
            verify_number(measured['spearman'], rho)
    bootstrap_rng=np.random.default_rng(42)
    for field, measured in result['summary'].items():
        assert measured['reviews'] == sum(r[field] is not None for r in rows)
        rates = [u['metrics'][field]['pair']['concordance'] for u in result['users'] if u['metrics'][field]['pair']['concordance'] is not None]
        verify_number(measured['macro_user_pair_concordance'], float(np.mean(rates)) if rates else None)
        if rates:
            indices=bootstrap_rng.integers(0,len(rates),size=(2000,len(rates)))
            ci=np.quantile(np.array(rates)[indices].mean(axis=1),[.025,.975])
            assert np.max(np.abs(ci-np.array(measured['macro_user_pair_ci95']))) < 1e-12
        else:
            assert measured['macro_user_pair_ci95'] is None
        verify_number(measured['spearman'], independent_rho(rows,field))
        high = [r for r in rows if r[field] is not None and r['rating']>=4]
        low = [r for r in rows if r[field] is not None and r['rating']<=3]
        ba = .5*(sum(r[field]>0 for r in high)/len(high)+sum(r[field]<0 for r in low)/len(low)) if high and low else None
        verify_number(measured['balanced_accuracy'], ba)
    annotations_path = folder/'manual_blind_annotations.json'
    human = json.loads(annotations_path.read_text()) if annotations_path.exists() else {'reviewer':None, 'annotations':[]}
    # This reference is assistant-authored, not a human-labelled benchmark.
    lookup = {r['review_id']:r for r in rows}
    checks=[]
    for reference in human['annotations']:
        extracted = lookup[reference['review_id']]
        model_axes = {a['aspect']:int(np.sign(a['sentiment'])) for a in extracted['aspects']}
        known = reference['aspect_signs']
        keys = set(known) | set(model_axes)
        discrepancies = {k:{'reference':known.get(k),'model':model_axes.get(k)} for k in sorted(keys) if known.get(k)!=model_axes.get(k)}
        overall = int(np.sign(extracted['overall'])) if extracted['overall'] is not None else None
        checks.append({'review_id':reference['review_id'], 'overall_reference':reference['overall_sign'],
                       'overall_model':overall, 'overall_match':reference['overall_sign']==overall,
                       'aspect_discrepancies':discrepancies, 'review_text':extracted['review_text']})
    verification = {'reviews':len(rows),'users':len(result['users']),'verbatim_quotes_checked':quote_count,
                    'all_user_pairs_replayed':True,'all_user_spearman_replayed':True,
                    'summary_metrics_replayed':True,'model_inputs_only_review_id_and_source_text':True,
                    'source_sentence_offsets_checked':all('sentences' in r for r in input_rows),
                    'user_bootstrap_intervals_replayed':True,
                    'spot_check_reference':human['reviewer'], 'spot_check_size':len(checks),
                    'spot_check_overall_sign_matches':sum(x['overall_match'] for x in checks),
                    'spot_check_reviews_with_aspect_discrepancies':sum(bool(x['aspect_discrepancies']) for x in checks),
                    'invalid_sentence_evidence_opinions':sum('rejection_reason' in a for r in rows for a in r['evidence_rejections']),
                    'unmentioned_empty_evidence_slots':sum(not a['evidence'] and 'rejection_reason' not in a for r in rows for a in r['evidence_rejections']),
                    'nonempty_unverified_evidence_opinions':sum(bool(a['evidence']) for r in rows for a in r['evidence_rejections']),
                    'reviews_with_nonempty_unverified_evidence':sum(any(a['evidence'] for a in r['evidence_rejections']) for r in rows)}
    write_json(folder/'verification.json',verification)
    write_json(folder/'spot_check_results.json',checks)
    matched={}
    rng=np.random.default_rng(42)
    for alternative in ['overall','taste','revisit']:
        same=[r for r in rows if r['aspect_mean'] is not None and r[alternative] is not None]
        deltas=[]; users=[]
        for u in result['users']:
            subset=[r for r in same if r['user_id']==u['user_id']]
            counts=[independent_pairs(subset,k) for k in ['aspect_mean',alternative]]
            if not counts[0]['pairs']:
                continue
            rates=[(c['correct']+.5*c['ties'])/c['pairs'] for c in counts]
            deltas.append(rates[1]-rates[0])
            users.append({'user_id':u['user_id'],'reviews':len(subset),'pairs':counts[0]['pairs'],
                          'aspect_mean_concordance':rates[0], 'alternative_concordance':rates[1],
                          'delta':rates[1]-rates[0]})
        ci=np.quantile(np.mean(rng.choice(deltas,size=(2000,len(deltas)),replace=True),axis=1),[.025,.975]).tolist() if deltas else None
        matched[alternative]={'reviews':len(same),'users':users,'macro_delta':float(np.mean(deltas)) if deltas else None,'ci95':ci}
    write_json(folder/'matched_comparisons.json',matched)
    with (folder/'review_aspect_vectors.csv').open('w') as f:
        fields=['review_id','user_id','rating',*ASPECTS,*[k+'_observed' for k in ASPECTS]]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in ['review_id','user_id','rating',*ASPECTS]},
                             **{k+'_observed':int(r[k] is not None) for k in ASPECTS}})
    profiles = ['# 사용자별 속성 근거와 평점 불일치 사례','',
                '사후 진단용이며 일반화된 취향 프로필로 확정하지 않았다. 언급 빈도는 중요도 가중치가 아니다. 각 사용자의 모든 텍스트 리뷰를 사용했다.','']
    for u in result['users']:
        subset=[r for r in rows if r['user_id']==u['user_id']]
        profiles += [f"## 사용자 {u['user_id']} ({u['text_reviews']}개 텍스트)",'',
                     '| 속성 | 언급 리뷰 | 긍정/혼재·중립/부정 | 평점 상관 |','|---|---:|---|---:|']
        for aspect in ASPECTS:
            vals=[r[aspect] for r in subset if r[aspect] is not None]
            c=Counter(int(np.sign(x)) for x in vals)
            rho=u['metrics'][aspect]['spearman']
            rho_text='미정의' if rho is None else f'{rho:.3f}'
            profiles.append(f'| {LABELS[aspect]} | {len(vals)} | {c[1]}/{c[0]}/{c[-1]} | {rho_text} |')
        profiles += ['', '**실제 원문 근거**','']
        # Deterministic first positive and negative evidence per axis. Never
        # infer a preference from a star label or from other users' reviews.
        for aspect in ASPECTS:
            for polarity,label in [(1,'긍정'),(-1,'부정')]:
                matches=[(r,a) for r in subset for a in r['aspects'] if a['aspect']==aspect and np.sign(a['sentiment'])==polarity]
                if matches:
                    r,a=matches[0]
                    profiles.append(f"- {LABELS[aspect]} {label}: {a['detail']} — “{' / '.join(a['evidence'])}” (실제 {r['rating']}점, 리뷰 {r['review_id']})")
        profiles += ['', '**평점과 전체 만족 표현이 엇갈린 리뷰**','']
        conflicts=[r for r in subset if r['overall'] is not None and ((r['rating']<=3 and r['overall']>0) or (r['rating']>=4 and r['overall']<0))]
        profiles.append(f'명확한 방향 불일치 {len(conflicts)}건. 3.5점·중립 감정은 이 목록에서 제외한다.')
        for r in conflicts[:3]:
            profiles.append(f"- 리뷰 {r['review_id']}: 실제 {r['rating']}점 / 전체 감정 {r['overall']} — {r['review_text']}")
        profiles += ['']
    (folder/'user_profiles.md').write_text('\n'.join(profiles)+'\n')
    base=(folder/'report.md').read_text().split('\n## 진단 결론과 추출 품질 검토')[0].split('\n## 근거 검증과 표본 검토')[0]
    notes = ['', '## 근거 검증과 표본 검토', '',
             f"입력/출력 {len(rows):,}개와 원문 근거 {quote_count:,}개를 독립 검증했다. 문장 번호 방식은 번호·원문 위치·복원된 인용을 함께 확인한다. 원문 존재 검사는 감정 해석의 정확성이나 누락 없는 추출을 보장하지 않는다.", '',
             f"별도 표본 검토 {len(checks)}개. 검토 라벨이 없는 실행은 수치·원문 검증만 수행하며 추출 정답률을 주장하지 않는다. assistant 작성 라벨도 사람 정답 라벨로 간주하지 않는다.", '',
             '재방문 의향의 명시성, 대기 불만 누락, 부정어·대조 표현의 해석은 표본 검토에서 확인해야 한다. 같은 리뷰와 평점의 사후 일치성을 미래 추천 성능으로 해석하지 않는다.', '']
    if (folder/'cost_summary.json').exists():
        cost=json.loads((folder/'cost_summary.json').read_text());q=cost['quota_after']
        notes += [f"비용 요약에 기록된 사용 가능한 추출 응답 {cost['usable_response_batches']}개, 입력 {cost['prompt_tokens']:,}토큰 / 출력 {cost['completion_tokens']:,}토큰, 해당 응답의 API 기록 비용 ${cost['reported_usable_response_cost_usd']}.", '',
                  f"실행 후 기록된 일일 무료 쿼터는 {q['used']}/{q['limit']}회 사용·{q['remaining']}회 잔여다. 현재 남은 쿼터를 뜻하지 않는다. 실패·중단한 요청의 포함 범위는 원본 사용량 기록을 확인한다.", '']
    notes += ['개별 원문 근거는 [사용자별 프로필](./user_profiles.md), 6개 속성 값과 미언급 mask는 `review_aspect_vectors.csv`, 표본 검토 전체는 `spot_check_results.json`, 수치 검증은 `verification.json`에 있다.','']
    (folder/'report.md').write_text(base+'\n'.join(notes))
    print(json.dumps(verification,ensure_ascii=False))


if __name__ == '__main__':
    main()
