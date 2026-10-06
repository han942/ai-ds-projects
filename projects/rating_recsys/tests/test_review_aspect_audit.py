import json
import asyncio

import pytest

from rating_recsys.experiments.review_aspect_audit import (
    RATING_PATTERN, ground_response, metrics, pair_concordance, read_stream, select_rows, validate_response,
)


def test_selection_uses_all_history_count_not_positive_or_text_count(tmp_path):
    rows = [dict(review_id=i, user_id=u, rating=r) for i, u, r in
            [(1, 9, 1), (2, 9, 2), (3, 9, 5), (4, 2, 5), (5, 2, 5)]]
    texts = [dict(review_id=i, review_text=t) for i,t in [(1,'별 1개 불친절'), (2,None), (3,'맛있다'), (4,'최고'), (5,'최고')]]
    for name, data in [('rows',rows), ('texts',texts)]:
        (tmp_path/name).write_text('\n'.join(map(json.dumps,data)))
    selected, cohort = select_rows(tmp_path/'rows', tmp_path/'texts', 1)
    assert cohort == [{'user_id':9, 'total_reviews':3, 'text_reviews':2}]
    assert {r['rating'] for r in selected} == {1,5}
    assert '별 1개' not in selected[0]['extraction_text']


def test_evidence_must_be_exact_source_and_not_borrowed():
    item = {'review_id':1, 'aspects':[{'aspect':'taste', 'sentiment':1, 'detail':'맛', 'evidence':['맛있다']}], 'overall':1, 'overall_evidence':['맛있다']}
    validate_response([item], [{'review_id':1, 'text':'맛있다. 하지만 비싸다'}])
    with pytest.raises(ValueError, match='substring'):
        validate_response([item], [{'review_id':1, 'text':'별로다'}])
    with pytest.raises(ValueError, match='IDs'):
        validate_response([item,item], [{'review_id':1, 'text':'맛있다'}])


def test_unknown_is_not_neutral_and_ties_count_half():
    rows = [{'rating':1, 'score':-1}, {'rating':4, 'score':1}, {'rating':5, 'score':1}, {'rating':2,'score':None}]
    p = pair_concordance(rows,'score')
    assert p == {'pairs':3,'correct':2,'ties':1,'wrong':0,'concordance':5/6,'strict_accuracy':2/3}
    m = metrics(rows,'score')
    assert m['coverage'] == .75 and m['balanced_accuracy'] == 1


def test_positive_only_accuracy_does_not_hide_negative_failure():
    rows = [{'rating':5,'s':2}]*9 + [{'rating':1,'s':2}]
    m = metrics(rows,'s')
    assert m['always_positive_accuracy'] == .9
    assert m['balanced_accuracy'] == .5 and m['spearman'] is None


def test_rating_mask_preserves_price_and_aspect_sentiment():
    text = '4.5점. 18000원이라 비싸지만 맛있다'
    assert RATING_PATTERN.sub('[별점표현 제거]',text) == '[별점표현 제거]. 18000원이라 비싸지만 맛있다'


def test_fabricated_evidence_is_unknown_not_neutral_and_remains_auditable():
    raw=[{'review_id':1,'aspects':{'taste':{'sentiment':2,'detail':'칭찬','evidence':['맛있다']},
                                 'value':{'sentiment':-1,'detail':'비싸다','evidence':['비싸다']}},
          'overall':2,'overall_evidence':['최고다']}]
    result=ground_response(raw,[{'review_id':1,'text':'맛은 별로인데 비싸다'}])[0]
    assert [a['aspect'] for a in result['aspects']] == ['value']
    assert result['overall'] is None
    assert [a['aspect'] for a in result['evidence_rejections']] == ['taste','overall']
    assert raw[0]['overall'] == 2  # original response is preserved


class StreamResponse:
    def __init__(self, lines):
        self.content=self.generate(lines)

    async def generate(self,lines):
        for line in lines:
            yield line.encode()


def test_stream_keeps_content_comments_terminal_usage_and_empty_choices():
    events=[{'id':'test','choices':[{'delta':{'content':'{"reviews":'},'finish_reason':None}]},
            {'choices':[{'delta':{'content':'[]}'},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'cost':0,'prompt_tokens':10,'completion_tokens':5}}]
    lines=[': OPENROUTER PROCESSING\n','\n']
    for e in events: lines += ['data: '+json.dumps(e)+'\n','\n']
    lines += ['data: [DONE]\n','\n']
    result=asyncio.run(read_stream(StreamResponse(lines)))
    assert result['choices'][0]['message']['content'] == '{"reviews":[]}'
    assert result['choices'][0]['finish_reason'] == 'stop'
    assert result['usage']['cost'] == 0 and result['stream_completed']


def test_stream_provider_error_is_not_a_completed_extraction():
    error={'error':{'code':504,'message':'Upstream timeout'}}
    result=asyncio.run(read_stream(StreamResponse(['data: '+json.dumps(error)+'\n','\n'])))
    assert result['error']['code'] == 504 and not result['stream_completed']
