import json
import asyncio

import pytest

from rating_recsys.experiments.review_aspect_audit import (
    MODEL, SCHEMA, VERSION, RATING_PATTERN, digest, extract, ground_response,
    load_cached_records, metrics, pair_concordance, read_stream, select_rows, validate_response,
)
from rating_recsys.experiments.review_evidence import extraction_inputs, model_inputs, source_sentences
from rating_recsys.experiments.review_aspect_validation import verify_source_evidence


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


def sentence_response(ids=(1,), overall_ids=(1,)):
    return [{'review_id':1, 'aspects':{'taste':{'sentiment':1, 'detail':'담백한 국물',
                                             'evidence_sentence_ids':list(ids)}},
             'overall':1, 'overall_evidence_sentence_ids':list(overall_ids)}]


def test_sentence_split_preserves_context_decimal_and_original_offsets():
    text = '  4.5점 가격은 18000원.\r\n국물은 좋지만 면은 별로!  대기는 길다\n\n'
    sentences = source_sentences(text)
    assert [s['text'] for s in sentences] == ['4.5점 가격은 18000원.', '국물은 좋지만 면은 별로!', '대기는 길다']
    assert [s['sentence_id'] for s in sentences] == [1,2,3]
    assert all(text[s['start']:s['end']] == s['text'] for s in sentences)
    assert source_sentences(' \n\r\n ') == []


def test_sentence_selection_reconstructs_full_evidence_and_ignores_generated_quote():
    text = '국물은 담백하지만 면은 퍼져서 아쉽다. 대기는 길다.'
    inputs = extraction_inputs([{'review_id':1, 'extraction_text':text}])
    raw = sentence_response()
    raw[0]['aspects']['taste']['evidence'] = ['아주 맛있다']
    result = ground_response(raw, inputs)
    assert result[0]['aspects'][0]['evidence'] == ['국물은 담백하지만 면은 퍼져서 아쉽다.']
    assert raw[0]['aspects']['taste']['evidence'] == ['아주 맛있다']
    validate_response(result, inputs)
    assert verify_source_evidence([{**result[0], 'extraction_text':text}], inputs) == 2
    result[0]['aspects'][0]['evidence'] = ['담백하지만']
    with pytest.raises(ValueError, match='selected source sentences'):
        validate_response(result, inputs)
    with pytest.raises(AssertionError):
        verify_source_evidence([{**result[0], 'extraction_text':text}], inputs)


@pytest.mark.parametrize('ids', [[0], [3], [1,1], [True], ['1'], [], [1,2,3], None])
def test_invalid_sentence_evidence_is_unknown_with_rejection_reason(ids):
    inputs = extraction_inputs([{'review_id':1, 'extraction_text':'국물이 담백하다. 대기는 길다.'}])
    raw = sentence_response()
    raw[0]['aspects']['taste']['evidence_sentence_ids'] = ids
    raw[0]['overall_evidence_sentence_ids'] = ids
    result = ground_response(raw, inputs)[0]
    assert result['aspects'] == [] and result['overall'] is None
    assert result['overall_evidence'] == result['overall_evidence_sentence_ids'] == []
    assert [r['aspect'] for r in result['evidence_rejections']] == ['taste','overall']
    assert all(r['rejection_reason'] for r in result['evidence_rejections'])


def test_redacted_rating_sentence_cannot_ground_an_opinion():
    inputs = extraction_inputs([{'review_id':1, 'extraction_text':'[별점표현 제거] 최고다. 국물은 담백하다.'}])
    result = ground_response(sentence_response(ids=(1,), overall_ids=(3,)), inputs)[0]
    assert result['aspects'] == [] and result['overall_evidence'] == ['국물은 담백하다.']
    assert 'redacted rating' in result['evidence_rejections'][0]['rejection_reason']


def test_rating_marker_is_isolated_without_losing_adjacent_opinion():
    text = RATING_PATTERN.sub('[별점표현 제거]', '5점 맛은 좋지만 대기가 길다')
    inputs = extraction_inputs([{'review_id':1,'extraction_text':text}])
    assert [s['text'] for s in inputs[0]['sentences']] == ['[별점표현 제거]','맛은 좋지만 대기가 길다']
    result = ground_response(sentence_response(ids=(2,),overall_ids=(2,)),inputs)[0]
    assert result['aspects'][0]['evidence'] == ['맛은 좋지만 대기가 길다']


def test_sentence_numbers_are_scoped_to_the_review_and_ids_are_checked_before_resolution():
    inputs = extraction_inputs([{'review_id':1, 'extraction_text':'국물이 담백하다.'},
                                {'review_id':2, 'extraction_text':'비싸다. 대기는 길다.'}])
    first = sentence_response(ids=(2,))[0]
    second = {'review_id':2, 'aspects':{}, 'overall':None, 'overall_evidence_sentence_ids':[]}
    assert ground_response([first,second],inputs)[0]['aspects'] == []
    with pytest.raises(ValueError, match='IDs'):
        ground_response([first,first], inputs)
    first['review_id'] = 99
    with pytest.raises(ValueError, match='IDs'):
        ground_response([first,second], inputs)


def test_model_inputs_exclude_labels_and_positions_and_schema_requires_selections():
    inputs = extraction_inputs([{'review_id':1, 'extraction_text':'국물이 담백하다.',
                                'rating':5, 'user_id':2, 'restaurant_id':3}])
    assert model_inputs(inputs) == [{'review_id':1, 'sentences':[{'sentence_id':1,'text':'국물이 담백하다.'}]}]
    aspect_schema = SCHEMA['properties']['reviews']['items']['properties']['aspects']['properties']['taste']['anyOf'][0]
    assert 'evidence_sentence_ids' in aspect_schema['required']
    assert 'evidence' not in aspect_schema['properties']


@pytest.mark.parametrize('version', ['aspect-audit-v3', VERSION])
def test_cache_replay_preserves_versions_and_detects_changed_inputs(tmp_path, version):
    source = {'review_id':1, 'extraction_text':'국물이 담백하다.'}
    if version == VERSION:
        inputs = extraction_inputs([source])
        payload = model_inputs(inputs)
        records = ground_response(sentence_response(), inputs)
    else:
        inputs = [{'review_id':1, 'text':source['extraction_text']}]
        payload = inputs
        records = [{'review_id':1,'aspects':[], 'overall':1,'overall_evidence':['국물이 담백하다.']}]
    manifest = {'version':version, 'model':MODEL, 'provider':'openrouter',
                'prompt':'original saved prompt', 'schema':{'original':'schema'}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'batches').mkdir()
    saved = {'hash':digest({**manifest,'input':payload}), 'reviews':records}
    (tmp_path/'batches'/'0000-cache.json').write_text(json.dumps(saved))
    assert list(load_cached_records(tmp_path,inputs).values()) == records
    inputs[0]['text'] = '불친절하다.'
    if version == VERSION:
        inputs[0]['sentences'] = source_sentences(inputs[0]['text'])
    with pytest.raises(ValueError, match='changed'):
        load_cached_records(tmp_path,inputs)


def test_extraction_sends_numbered_sentences_and_reuses_completed_batches(tmp_path, monkeypatch):
    from rating_recsys.experiments import review_aspect_audit as audit
    posts = []
    response_content = json.dumps({'reviews':sentence_response()})

    class Response:
        status = 200
        headers = {'Content-Type':'application/json'}

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self, **kwargs):
            return self.body

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, url, **kwargs):
            if url.endswith('/models'):
                return Response({'data':[{'id':MODEL,'pricing':{'prompt':'0','completion':'0'}}]})
            return Response({'data':{'free_model_daily_requests':{'remaining':10}}})

        def post(self, url, json, **kwargs):
            posts.append(json)
            return Response({'id':'test', 'model':MODEL, 'usage':{'cost':0},
                             'choices':[{'finish_reason':'stop', 'message':{
                                 'content':response_content}}]})

    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-placeholder')
    monkeypatch.setattr(audit, 'dotenv_values', lambda path: {})
    monkeypatch.setattr(audit.aiohttp, 'ClientSession', lambda **kwargs: Session())
    rows = [{'review_id':1, 'extraction_text':'국물이 담백하다.', 'rating':5,'user_id':99}]
    records = asyncio.run(extract(rows,tmp_path,1,1,0))
    assert json.loads(posts[0]['messages'][1]['content']) == model_inputs(extraction_inputs(rows))
    assert records[0]['overall_evidence'] == [rows[0]['extraction_text']]
    assert asyncio.run(extract(rows,tmp_path,1,1,0)) == records
    assert len(posts) == 1


@pytest.mark.parametrize('version', ['aspect-audit-v3', VERSION])
def test_prepare_and_analyze_cli_and_independent_validation_without_api(tmp_path, monkeypatch, version):
    import sys
    from rating_recsys.experiments import review_aspect_audit as audit
    from rating_recsys.experiments import review_aspect_validation as validation
    snapshot_dir = tmp_path/'artifacts'/'snapshots'
    snapshot_dir.mkdir(parents=True)
    rows = [dict(review_id=i, user_id=9, rating=rating) for i,rating in [(1,1),(2,3),(3,5)]]
    texts = [{'review_id':i,'review_text':t} for i,t in
             [(1,'국물이 너무 짜다.'),(2,'국물이 보통이다.'),(3,'국물이 담백하다.')]]
    for suffix, data in [('.jsonl',rows),('.reviews.jsonl',texts)]:
        (snapshot_dir/('e7896add5b4b5939'+suffix)).write_text('\n'.join(map(json.dumps,data)))
    monkeypatch.setattr(audit, 'PROJECT_ROOT', tmp_path)

    def forbid_api(*args, **kwargs):
        raise AssertionError('No API extraction in prepare/analyze modes')

    monkeypatch.setattr(audit,'extract',forbid_api)
    folder = tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['audit','--output',str(folder),'--users','1','--prepare-only'])
    audit.main()
    inputs = [json.loads(line) for line in (folder/'extraction_input.jsonl').read_text().splitlines()]
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['version'] == VERSION
    assert not (folder/'results.json').exists()
    if version == 'aspect-audit-v3':
        inputs = [{'review_id':i['review_id'],'text':i['text']} for i in inputs]
        manifest.update(version=version, prompt='saved legacy prompt', schema={'saved':'legacy schema'})
        (folder/'manifest.json').write_text(json.dumps(manifest))
        (folder/'extraction_input.jsonl').write_text('\n'.join(map(json.dumps,inputs)))
        records = [{'review_id':i['review_id'], 'aspects':[{'aspect':'taste','sentiment':score,
                    'detail':'국물 의견','evidence':[i['text']]}], 'overall':score,
                    'overall_evidence':[i['text']], 'evidence_rejections':[]}
                   for i,score in zip(inputs,[-1,0,1])]
        payload = inputs
    else:
        raw = []
        for i,score in zip(inputs,[-1,0,1]):
            r = sentence_response()[0]
            r['review_id'] = i['review_id']
            r['aspects']['taste']['sentiment'] = r['overall'] = score
            raw.append(r)
        records = ground_response(raw,inputs)
        payload = model_inputs(inputs)
    (folder/'batches').mkdir()
    cache_hash = digest({**{k:manifest[k] for k in ('version','model','provider','prompt','schema')},'input':payload})
    (folder/'batches'/'0000-cached.json').write_text(json.dumps({'hash':cache_hash,'reviews':records}))
    original_manifest = (folder/'manifest.json').read_bytes()
    original_inputs = (folder/'extraction_input.jsonl').read_bytes()
    monkeypatch.setattr(sys,'argv',['audit','--output',str(folder),'--users','1','--analyze-only'])
    audit.main()
    assert (folder/'manifest.json').read_bytes() == original_manifest
    assert (folder/'extraction_input.jsonl').read_bytes() == original_inputs
    monkeypatch.setattr(sys,'argv',['validation',str(folder)])
    validation.main()
    verification = json.loads((folder/'verification.json').read_text())
    assert verification['verbatim_quotes_checked'] == 6
    assert verification['source_sentence_offsets_checked'] == (version == VERSION)
    assert verification['spot_check_size'] == 0
    validation.main()
    assert (folder/'report.md').read_text().count('## 근거 검증과 표본 검토') == 1
