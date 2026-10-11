import numpy as np

from rating_recsys.retrieval.aspect_graph import CHANNELS
from rating_recsys.experiments.text_candidate_views_cli import repeated_mean_profiles, shuffle_profiles


def profiles():
    bank = {}
    for key, channels in ((1,(0,1)),(2,(0,2)),(3,(1,2)),(4,(0,1,2))):
        value = np.zeros((len(CHANNELS),384),np.float32)
        for channel in channels:
            value[channel,key+channel] = 1
        bank[key] = value.reshape(-1)
    return bank


def test_pooled_control_matches_capacity_masks_but_removes_channel_content():
    original = profiles()
    pooled = repeated_mean_profiles(original)
    assert set(original) == set(pooled)
    for key in original:
        actual = original[key].reshape(len(CHANNELS),384)
        control = pooled[key].reshape(len(CHANNELS),384)
        mask = np.any(actual != 0,axis=1)
        np.testing.assert_array_equal(mask,np.any(control != 0,axis=1))
        assert np.linalg.matrix_rank(control[mask]) == 1
        assert np.linalg.matrix_rank(actual[mask]) > 1


def test_channel_shuffle_preserves_each_entity_mask_and_each_channel_vector_set():
    original = profiles()
    shuffled,trace = shuffle_profiles(original,42)
    repeat,_ = shuffle_profiles(original,42)
    assert len(trace) == len(CHANNELS)
    for key in original:
        np.testing.assert_array_equal(shuffled[key],repeat[key])
        a,b = original[key].reshape(len(CHANNELS),384),shuffled[key].reshape(len(CHANNELS),384)
        np.testing.assert_array_equal(np.any(a != 0,1),np.any(b != 0,1))
    for channel in range(len(CHANNELS)):
        before = sorted(value.reshape(len(CHANNELS),384)[channel].tobytes() for value in original.values())
        after = sorted(value.reshape(len(CHANNELS),384)[channel].tobytes() for value in shuffled.values())
        assert before == after
