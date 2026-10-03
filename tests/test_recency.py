from dataclasses import replace
import numpy as np
import pytest
from audit.protocol import Job
from evaluate_recency import combine


def records():
    ends = [37, 47, 52, 57, 62]
    return [(Job('flow31', s, f, 70, t, t+3, t+7),
             {'score_rows': np.arange(3), 'score_pred': np.array([f, s, f+s], dtype=float)})
            for s in [1, 2] for f, t in enumerate(ends)]


def test_recency_weights_are_origin_invariant_and_seed_balanced():
    rec = records()
    rows, base, pred, weights = combine(rec, [1, 2], 20)
    expected = np.exp2((np.array([37, 47, 52, 57, 62])-62)/20)
    expected /= expected.sum()
    np.testing.assert_allclose(list(weights.values()), expected)
    np.testing.assert_allclose(base, [2, 1.5, 3.5])
    np.testing.assert_allclose(pred, [expected @ np.arange(5), 1.5, expected @ np.arange(5)+1.5])
    shifted = [(replace(j, origin=59, train_end=j.train_end-11), p) for j, p in rec]
    _, _, other, other_weights = combine(shifted, [1, 2], 20)
    np.testing.assert_array_equal(pred, other)
    assert weights == other_weights


def test_recency_rejects_missing_models_and_misaligned_rows():
    rec = records()
    with pytest.raises(ValueError, match='coverage'):
        combine(rec[:-1], [1, 2], 20)
    rec[-1][1]['score_rows'] = np.array([2, 1, 0])
    with pytest.raises(ValueError, match='Misaligned'):
        combine(rec, [1, 2], 20)
