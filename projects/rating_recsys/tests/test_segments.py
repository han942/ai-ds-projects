from types import SimpleNamespace
import math
import pytest
from rating_recsys.evaluation.segments import analyze_segments, count_bin, validate_saved_rows

def query(key, user, truth, history=1):
    return SimpleNamespace(query_id=key, user_id=user, relevance_by_item=truth,
                           history=[SimpleNamespace(restaurant_id=-i-1) for i in range(history)])

def test_boundaries():
    assert [count_bin(n) for n in (0,1,4,5,9,10,19,20)] == ["0-unseen","1-4","1-4","5-9","5-9","10-19","10-19","20+"]
    with pytest.raises(ValueError): count_bin(-1)

def test_original_rank_full_idcg_unseen_denominator_and_clustered_changes():
    qs = [query("a",1,{1:2,2:1,99:2}), query("b",2,{1:1},5),query("c",3,{3:0})]
    arms = {"c5":{"a":(3,1,2),"b":(3,2),"c":(3,2)}, "text":{"a":(2,3,1),"b":(1,3),"c":(2,3)}}
    result, changes = analyze_segments(qs, arms, {1:2,2:20,3:5}, samples=100)
    bins = {r["visit_bin"]:r for r in result["item_visit_segments"]}
    assert bins["0-unseen"]["relevant_user_item_pairs"] == 1
    assert bins["0-unseen"]["arms"]["c5"]["macro_recall_at_100"] == 0
    assert bins["0-unseen"]["arms"]["text"]["exposures_at_100_all_queries"] == 0
    assert bins["1-4"]["arms"]["c5"]["macro_recall_at_100"] == .5
    ideal = 3 + 3/math.log2(3) + 1/math.log2(4)
    assert bins["1-4"]["arms"]["c5"]["dcg_contribution_at_10_over_full_idcg"] == pytest.approx((3/math.log2(3)/ideal)/2)
    assert bins["1-4"]["arms"]["text"]["gained_pairs_vs_baseline"] == 1
    assert changes == [{"arm":"text","query_id":"b","user_id":2,"visit_bin":"1-4","gained_relevant_items":[1],"lost_relevant_items":[]}]
    for arm in arms:
        contributions = sum(r["arms"][arm]["dcg_contribution_at_10_over_full_idcg"] for r in bins.values())
        expected_a = sum((2**qs[0].relevance_by_item.get(i,0)-1)/math.log2(j+2) for j,i in enumerate(arms[arm]["a"]))/ideal
        expected_b = 1 if arm == "text" else 0
        assert contributions == pytest.approx((expected_a+expected_b)/2)

@pytest.mark.parametrize("mutate", [lambda r:r.update(user_id=9),lambda r:r.update(relevance_by_item={"1":1}),lambda r:r.pop("c5"),lambda r:r.update(query_id="other")])
def test_saved_rows_fail_closed(mutate):
    q=query("a",1,{1:2})
    row={"query_id":"a","user_id":1,"relevance_by_item":{"1":2},"c5":[1]}
    mutate(row)
    with pytest.raises((ValueError,KeyError)): validate_saved_rows([q],[row],["c5"])

def test_empty_bins_undefined_and_duplicate_users_rejected():
    q=query("a",1,{1:2})
    result,_=analyze_segments([q],{"c5":{"a":(1,)}},{1:1},samples=10)
    assert result["user_history_segments"][1]["arms"]["c5"]["recall_at_100"] is None
    with pytest.raises(ValueError): analyze_segments([q,query("b",1,{1:2})],{"c5":{"a":(1,),"b":(1,)}},{1:1},samples=10)
