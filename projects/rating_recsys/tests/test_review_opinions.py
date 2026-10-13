"""Rule opinions retain contrasting targets, source offsets and uncertainty."""
import pytest

from rating_recsys.retrieval.review_opinions import CATEGORIES, VERSION, extract_opinions


def compact(text):
    return [(row["category"], row["target"], row["polarity"]) for row in extract_opinions(text)]


def test_contrast_keeps_food_praise_and_price_complaint_separate():
    text = "맛있지만 비싸다."
    records = extract_opinions(text)
    assert [(r["category"], r["polarity"]) for r in records] == [("taste", "positive"), ("value", "negative")]
    assert records[0]["evidence"] == "맛있지만"
    assert records[1]["evidence"] == "비싸다."
    assert records[1]["target"] == "가격"
    assert records[1]["target_kind"] == "implicit_category"
    assert records[1]["target_start"] is None


def test_concrete_food_targets_do_not_collapse_with_opposite_sentiments():
    text = "국물이 진해서 좋다. 고기는 질겼고 가격은 비쌌다."
    assert compact(text) == [("taste", "국물", "positive"), ("taste", "고기", "negative"),
                             ("value", "가격", "negative")]


@pytest.mark.parametrize("text", ["직원은 친절하지 않다.", "직원은 안 친절하다.",
                                 "안 맛있다.", "맛있지 않다.", "분위기가 좋지 않다."])
def test_explicit_negation_is_not_positive_and_is_inside_opinion_span(text):
    records = extract_opinions(text)
    assert records
    assert all(r["polarity"] == "negative" for r in records)
    assert all("local-negation:1" in r["rule_trace"] for r in records)
    assert any("안" in r["opinion"] or "않" in r["opinion"] for r in records)


def test_negated_attribute_with_satisfaction_is_not_a_negative_taste_opinion():
    records = extract_opinions("맵지 않아 좋다.")
    assert [(r["category"], r["polarity"]) for r in records] == [("taste", "positive")]
    assert records[0]["evidence"] == "맵지 않아 좋다."


def test_negation_scope_ends_at_contrast_and_does_not_flip_next_opinion():
    records = extract_opinions("직원이 친절하지 않지만 국물은 맛있다.")
    assert [(r["category"], r["polarity"]) for r in records] == [("service", "negative"), ("taste", "positive")]


@pytest.mark.parametrize("text,category,target", [("양이 많다.", "portion", "양"),
                                                    ("양이 적다.", "portion", "양"),
                                                    ("가격은 12,000원이다.", "value", "가격"),
                                                    ("분위기가 조용하다.", "atmosphere", "분위기")])
def test_objective_facts_are_not_fabricated_praise(text, category, target):
    assert compact(text) == [(category, target, "neutral")]


def test_quantity_satisfaction_keeps_the_judgement_not_a_second_neutral_record():
    assert compact("양이 많아서 좋다.") == [("portion", "양", "positive")]


def test_waiting_complaint_and_numerical_wait_fact_are_different():
    assert compact("웨이팅이 길다.") == [("service", "웨이팅", "negative")]
    # No numerical-duration rule turns a bare observation into dissatisfaction.
    assert compact("웨이팅 10분.") == []


@pytest.mark.parametrize("text", ["국물이 맛있을 수도 있다.", "국물이 맛있다는 소문이다.",
                                 "국물 맛이 애매하다.", "맛없지 않다.", "안 맛있지 않다."])
def test_ambiguous_attributed_and_double_negative_praise_remains_unknown(text):
    records = extract_opinions(text)
    assert records
    assert all(r["polarity"] == "unknown" for r in records)


def test_overall_praise_does_not_invent_unmentioned_categories():
    assert extract_opinions("좋아요! 최고!") == []
    assert extract_opinions("메뉴를 주문했다.") == []
    assert extract_opinions("5점") == []
    assert extract_opinions(None) == []
    assert extract_opinions("  ") == []
    assert extract_opinions("불쾌하다.") == []


def test_raw_offsets_roundtrip_unicode_and_repeated_source_occurrences():
    text = "  🍜 국물은 맛있다!\n직원은 친절하지 않다. 국물은 맛있다!  "
    records = extract_opinions(text)
    assert len(records) == 3
    for record in records:
        assert record["version"] == VERSION
        assert record["category"] in CATEGORIES
        assert text[record["start"]:record["end"]] == record["evidence"]
        assert text[record["opinion_start"]:record["opinion_end"]] == record["opinion"]
        assert record["start"] <= record["opinion_start"] < record["opinion_end"] <= record["end"]
        if record["target_kind"] == "explicit":
            assert text[record["target_start"]:record["target_end"]] == record["target"]
        assert record["rule_trace"]
    assert records[0]["start"] != records[-1]["start"]
    assert extract_opinions(text) == records


def test_different_targets_in_same_clause_are_not_sentiment_averaged():
    records = extract_opinions("국물은 맛있다 고기는 질기다")
    assert [(r["category"], r["target"], r["polarity"]) for r in records] == [
        ("taste", "국물", "positive"), ("taste", "고기", "negative")]


def test_non_text_input_is_rejected_without_an_implicit_rating_channel():
    with pytest.raises(TypeError):
        extract_opinions(5)


@pytest.mark.parametrize("text,category", [
    ("웨이팅하는 걸 좋아하지 않아 방문하지 못했었는데", "service"),
    ("가성비가 좋은 느낌은 아니라 아쉬웠어요", "value"),
    ("그에 비해 탕수육은 맛있다고 느껴지지 않는다.", "taste"),
    ("가격대비 가성비가 좋은 편은 아니지만", "value"),
    ("맛은 전반적으로 아주 맛있다 까지는 아니었습니다.", "taste"),
])
def test_real_review_indirect_denial_does_not_enter_positive_channels(text, category):
    records = [record for record in extract_opinions(text) if record["category"] == category]
    assert records
    assert all(record["polarity"] in ("negative", "unknown") for record in records)
    denied = [record for record in records if "negation-kind:" in " ".join(record["rule_trace"])]
    assert denied
    assert all("않" in record["opinion"] or "아니" in record["opinion"] for record in denied)
    for record in records:
        assert text[record["start"]:record["end"]] == record["evidence"]
        assert text[record["opinion_start"]:record["opinion_end"]] == record["opinion"]


@pytest.mark.parametrize("text", ["국물은 좋은 것은 아니다.", "국물은 좋은 건 아니에요.",
                                 "국물은 맛있다고 생각하진 않아요.", "국물은 맛있는 편은 아니었다."])
def test_qualified_and_cognitive_denial_is_unknown_instead_of_invented_opposite(text):
    records = extract_opinions(text)
    assert len(records) == 1
    assert records[0]["polarity"] == "unknown"
    assert "negation-kind:" in " ".join(records[0]["rule_trace"])


def test_explicit_taste_denial_still_negates_without_changing_other_target():
    text = "국물은 맛있지는 않았어요. 고기는 맛있다 직원은 친절하지 않았다."
    assert compact(text) == [("taste", "국물", "negative"), ("taste", "고기", "positive"),
                             ("service", "직원", "negative")]


def test_later_nonpreference_negation_does_not_invert_food_praise():
    text = "국물은 맛있다 직원에게 설명을 듣지 못했다."
    records = extract_opinions(text)
    assert [(record["category"], record["polarity"]) for record in records] == [("taste", "positive")]
    assert records[0]["opinion"] == "맛있다"


def test_non_spiciness_followed_by_satisfaction_keeps_the_positive_polarity():
    assert compact("맵지 않아 좋다.") == [("taste", "맵", "positive")]
    assert VERSION == "raw-multi-opinion-rules-v3"


@pytest.mark.parametrize("text,polarity", [
    ("직원은 친절하지도 않다.", "negative"),
    ("친절하지도않음", "negative"),
    ("국물은 맛있지도 않았다.", "negative"),
    ("맛이 좋지도 않다.", "negative"),
    ("웨이팅을 좋아하지도 않아요.", "negative"),
    ("국물은 맛있다고 생각하지도 않는다.", "unknown"),
    ("국물은 안 맛있지도 않다.", "unknown"),
])
def test_emphatic_negation_preserves_polarity_and_original_scope(text, polarity):
    records = extract_opinions(text)
    assert len(records) == 1
    record = records[0]
    assert record["polarity"] == polarity
    assert "지도" in record["opinion"] and "않" in record["opinion"]
    assert text[record["opinion_start"]:record["opinion_end"]] == record["opinion"]
    assert text[record["start"]:record["end"]] == record["evidence"]


def test_emphatic_negation_does_not_flip_other_opinions():
    assert compact("직원은 친절하지도 않지만 국물은 맛있다.") == [
        ("service", "직원", "negative"), ("taste", "국물", "positive")]
