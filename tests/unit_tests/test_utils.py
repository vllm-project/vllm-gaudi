import itertools

import pytest

from vllm_gaudi.extension.utils import align_and_pad


@pytest.fixture
def padding_gen():
    # Fresh infinite generator of -1 for each test invocation
    return itertools.repeat(-1)


class TestAlignAndPad:

    @staticmethod
    def _materialize_rows(rows):
        out = []
        for r in rows:
            if isinstance(r, list):
                out.append(r)
            else:
                out.append(list(r))
        return out

    def test_flatten_and_pad_rows(self, padding_gen):
        data = [[1, 2], [3]]
        out = align_and_pad(data, (1, 5), padding_gen)
        out = self._materialize_rows(out)
        assert len(out) == 1
        assert out[0] == [1, 2, 3, -1, -1]

    def test_row_and_batch_padding(self, padding_gen):
        data = [[1], [2, 3]]
        out = align_and_pad(data, (4, 3), padding_gen)
        out_materialized = self._materialize_rows(out)
        assert len(out_materialized) == 4
        for row in out_materialized:
            assert len(row) == 3
        assert out_materialized[0] == [1, -1, -1]
        assert out_materialized[1] == [2, 3, -1]
        # Added rows become identical, sourced from the same padded slice (-1s)
        assert out_materialized[2] == [-1, -1, -1]
        assert out_materialized[3] == [-1, -1, -1]

    def test_no_padding_needed(self, padding_gen):
        data = [[1, 2], [3, 4]]
        out = align_and_pad(data, (2, 2), padding_gen)
        out = self._materialize_rows(out)
        assert out == [[1, 2], [3, 4]]

    def test_only_batch_padding(self, padding_gen):
        data = [[5, 6]]
        out = align_and_pad(data, (3, 2), padding_gen)
        out_materialized = self._materialize_rows(out)
        assert len(out_materialized) == 3
        assert out_materialized[0] == [5, 6]
        assert out_materialized[1] == [-1, -1]
        assert out_materialized[2] == [-1, -1]

    @pytest.mark.parametrize(
        "data,bucketing,expected_first,expected_last,expected_shape",
        [
            ([[1, 2, 3]], (1, 3), [1, 2, 3], [1, 2, 3], (1, 3)),
            ([[1, 2, 3]], (2, 5), [1, 2, 3, -1, -1], [-1, -1, -1, -1, -1], (2, 5)),
        ],
    )
    def test_parametrized(self, data, bucketing, expected_first, expected_last, expected_shape, padding_gen):
        out = align_and_pad(data, bucketing, padding_gen)
        out_materialized = self._materialize_rows(out)
        target_bs, target_len = expected_shape
        assert len(out_materialized) == target_bs
        assert out_materialized[0] == expected_first
        assert out_materialized[-1] == expected_last
        for row in out_materialized:
            assert len(row) == target_len
