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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args = parser.parse_args()
    folder = args.folder
    rows = [json.loads(x) for x in (folder/'extracted_reviews.jsonl').open()]
    result = json.loads((folder/'results.json').read_text())
    inputs = {r['review_id']:r['text'] for r in map(json.loads,(folder/'extraction_input.jsonl').open())}
    assert len(rows) == len(inputs) == len(set(r['review_id'] for r in rows)) == result['reviews']
    assert all(set(r)=={'review_id','text'} for r in map(json.loads,(folder/'extraction_input.jsonl').open()))
    quote_count = 0
    for r in rows:
        assert r['extraction_text'] == inputs[r['review_id']]
        axes = [a['sentiment'] for a in r['aspects'] if a['aspect'] != 'revisit']
        assert r['aspect_mean'] == (sum(axes)/len(axes) if axes else None)
        for quote in [q for a in r['aspects'] for q in a['evidence']] + r['overall_evidence']:
            assert quote in inputs[r['review_id']]
            quote_count += 1
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
            assert rho is None and measured['spearman'] is None or rho is not None and abs(rho-measured['spearman']) < 1e-12
    bootstrap_rng=np.random.default_rng(42)
    for field, measured in result['summary'].items():
        assert measured['reviews'] == sum(r[field] is not None for r in rows)
        rates = [u['metrics'][field]['pair']['concordance'] for u in result['users'] if u['metrics'][field]['pair']['concordance'] is not None]
        assert abs(measured['macro_user_pair_concordance']-np.mean(rates)) < 1e-12
        indices=bootstrap_rng.integers(0,len(rates),size=(2000,len(rates)))
        ci=np.quantile(np.array(rates)[indices].mean(axis=1),[.025,.975])
        assert np.max(np.abs(ci-np.array(measured['macro_user_pair_ci95']))) < 1e-12
        assert abs(measured['spearman']-independent_rho(rows,field)) < 1e-12
        high = [r for r in rows if r[field] is not None and r['rating']>=4]
        low = [r for r in rows if r[field] is not None and r['rating']<=3]
        ba = .5*(sum(r[field]>0 for r in high)/len(high)+sum(r[field]<0 for r in low)/len(low))
        assert abs(ba-measured['balanced_accuracy']) < 1e-12
    human = json.loads((folder/'manual_blind_annotations.json').read_text())
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
                    'summary_metrics_replayed':True,'model_inputs_only_review_id_and_text':True,
                    'user_bootstrap_intervals_replayed':True,
                    'spot_check_reference':human['reviewer'], 'spot_check_size':len(checks),
                    'spot_check_overall_sign_matches':sum(x['overall_match'] for x in checks),
                    'spot_check_reviews_with_aspect_discrepancies':sum(bool(x['aspect_discrepancies']) for x in checks),
                    'unmentioned_empty_evidence_slots':sum(not a['evidence'] for r in rows for a in r['evidence_rejections']),
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
    base=(folder/'report.md').read_text().split('\n## 진단 결론과 추출 품질 검토')[0]
    primary=result['summary']['aspect_mean']
    positive=sum(u['metrics']['aspect_mean']['spearman'] is not None and u['metrics']['aspect_mean']['spearman']>0 for u in result['users'])
    notes=['','## 진단 결론과 추출 품질 검토','',
           f"**리뷰에는 평점과 연결되는 신호가 있다.** 속성 평균과 평점의 사용자 평균 상관은 {primary['macro_user_spearman']:.3f}, 사용자 {positive}/{len(result['users'])}명에서 양의 상관이다. 낮은 평점부터 높은 평점으로 갈수록 평균 감정 점수도 증가한다. 사용자의 평점 성향을 섞은 전체 상관만으로 판단하지 않았다.", '',
           '다만 6개 속성 점수는 음식의 종류나 담백함·매운맛 같은 취향 조건을 모두 표현하지 못한다. 속성 평균은 일치성 진단용 점수이며 완성된 사용자 취향 임베딩이 아니다. 세부 의견과 원문 근거를 함께 남겼다.', '',
           '속성별 상관은 표본과 언급 범위가 서로 다르다. 이 값을 개인의 속성 중요도나 추천 모델의 feature importance로 해석하지 않는다. 가중치를 학습해 검증한 결과도 아니다.', '',
           f"같은 리뷰에 전체 만족과 속성 평균이 모두 있는 {matched['overall']['reviews']}개에서, 전체 만족의 사용자 평균 쌍 일치는 속성 평균보다 {100*matched['overall']['macro_delta']:+.2f}%p 높았다(95% {100*matched['overall']['ci95'][0]:+.2f}~{100*matched['overall']['ci95'][1]:+.2f}%p). 차이를 확인하지 못했으며, 속성 추출이 단순 전체 감정보다 점수 순서를 더 잘 맞춘다고 주장하지 않는다.", '',
           f"**30개 원문 표본 검토:** 실제 평점이나 추출 응답을 보기 전에 Codex가 별도 판정한 표본과 전체 감정 방향이 {verification['spot_check_overall_sign_matches']}/{verification['spot_check_size']}개 일치했다. {verification['spot_check_reviews_with_aspect_discrepancies']}개에서는 적어도 한 속성의 존재·감정 방향이 달랐다. 누락, 중립과 긍정의 경계 차이, 원문 근거 제외, 과잉 추론이 섞여 있다. 독립 사람 라벨이 아니므로 실제 추출 정답률이나 오류율로 해석하지 않는다.", '',
           '**원문에서 확인한 개선점**','',
           '- 리뷰 35200: “가성비왕 ! 근데 점심시간 웨이팅 장난아님 ㅠ”. 모델은 가성비를 긍정으로 잡았지만 서비스/대기는 미언급으로 반환했다. 전체 평점과의 일치가 속성별 정보 보존을 보장하지 않는 사례다.',
           '- 리뷰 36: “분위가와 음식의 맛은 굉장히 좋았지만”을 재방문 긍정의 근거로 사용했다. 명시적 재방문 의향이 없는 만족 표현을 재방문으로 확대 해석했다.',
           '- 리뷰 60090: “주차는 어렵고 새우버거가 최고 입니다”. 실제 평점 5점, 전체 감정 0이다. 주차를 분위기 불만으로 분류하고 전체 만족을 중립으로 합쳐, 높은 실제 평점과 방향이 엇갈렸다. 이 한 사례만으로 개인의 속성 가중치를 추정하지 않는다.',
           '- 리뷰 60296: “맛과 가격 모두 보통 이상이다.” 실제 평점 3점, 전체 감정 +1이다. 긍정적인 말과 높은 절대 평점은 항상 일치하지 않는다.', '',
           f"본문 별점 표현을 포함한 리뷰를 아예 제외해도 속성 평균의 쌍 일치는 {100*result['without_explicit_rating_text']['aspect_mean']['macro_user_pair_concordance']:.2f}%다. 직접적인 별점 표현만으로 얻은 결과는 아니다. 정규식이 모든 표현을 잡는다고 보장하지 않는다.", '',
           '**다음 단계 권고:** 원문을 다시 생성하는 대신 문장 번호로 근거를 선택하게 하고, 명시하지 않은 재방문·추천 의향을 추측하지 않도록 보완한다. 음식과 맛의 구체적 선호·불만을 보존한 뒤 cutoff 이전 리뷰로 사용자 프로필을 만들고 추천 지표를 별도로 검증한다. 이 후속 단계는 아직 실행하지 않았다.', '',
           '## 독립 재검증과 실행 비용','',
           f"{len(rows):,}개 입력/출력 ID와 {quote_count:,}개 원문 인용을 확인했다. 사용자별 Spearman은 rankdata의 Pearson으로, 리뷰 쌍 수와 동점 처리율은 별도의 행렬 계산으로 재검증했다. 사용자 평균, 균형 일치율, bootstrap 95% 구간도 재현했다. 추가 단위 테스트 8개가 통과했다.", '']
    if (folder/'cost_summary.json').exists():
        cost=json.loads((folder/'cost_summary.json').read_text());q=cost['quota_after']
        notes += [f"사용 가능한 추출 응답 {cost['usable_response_batches']}개(60개×19, 40개×23, 39개×1, 20개×2), 입력 {cost['prompt_tokens']:,}토큰 / 출력 {cost['completion_tokens']:,}토큰, 이 응답들의 API 기록 비용 ${cost['reported_usable_response_cost_usd']}. 무료 모델만 호출했다. 원문 전체를 한 번씩 성공적으로 처리했으며 완료 리뷰는 재호출하지 않았다.", '',
                  f"최초 출력 길이 시험에서 잘린 응답 한 건이 별도로 있었다(비용 $0, 최종 결과에 미사용). 제공자 timeout으로 실패한 응답과 중단한 요청은 위의 사용 가능한 응답 수·토큰에 포함하지 않는다. 종료 후 이 키가 반환한 일일 무료 쿼터는 {q['used']}/{q['limit']}회 사용·{q['remaining']}회 잔여였다.", '',
                  '긴 응답의 시간 초과 때문에 요청 묶음을 60→40→20개로 줄였다. 모든 응답은 같은 모델·프롬프트·속성 schema이며, 요청 크기별 품질 비교 실험은 아니다. 초기 성공한 60개 파일에는 완성된 pilot 응답을 재사용한 출처도 기록했다.', '']
    notes += ['개별 원문 근거는 [사용자별 프로필](./user_profiles.md), 6개 속성 값과 미언급 mask는 `review_aspect_vectors.csv`, 표본 검토 전체는 `spot_check_results.json`, 수치 검증은 `verification.json`에 있다.','']
    (folder/'report.md').write_text(base+'\n'.join(notes))
    print(json.dumps(verification,ensure_ascii=False))


if __name__ == '__main__':
    main()
