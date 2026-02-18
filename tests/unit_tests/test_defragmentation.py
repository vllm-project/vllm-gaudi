###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest
import torch
from unittest.mock import MagicMock, patch

from vllm_gaudi.extension.defragmentation import OnlineDefragmenter, CacheSwapUtils


@pytest.fixture
def mock_config():
    """Mock configuration for defragmenter"""
    with patch('vllm_gaudi.extension.defragmentation.get_config') as mock_cfg:
        config = MagicMock()
        config.defrag = True
        config.VLLM_DEFRAG_THRESHOLD = 32
        config.VLLM_DEFRAG_WITH_GRAPHS = False
        config.bridge_mode = 'eager'
        mock_cfg.return_value = config
        yield config


@pytest.fixture
def mock_debug_logger():
    """Mock debug logger"""
    with patch('vllm_gaudi.extension.defragmentation.init_debug_logger') as mock_debug:
        mock_debug.return_value = None
        yield mock_debug


@pytest.fixture
def defragmenter(mock_config, mock_debug_logger):
    """Create OnlineDefragmenter instance"""
    return OnlineDefragmenter()


class TestOnlineDefragmenter:
    """Test suite for OnlineDefragmenter"""

    def test_extend_mapping_table(self, defragmenter):
        """Test mapping table extension"""
        defragmenter._extend_mapping_table(5)
        assert len(defragmenter.fwd_mapping_table) == 6  # 0-5
        assert len(defragmenter.bwd_mapping_table) == 6
        assert defragmenter.fwd_mapping_table == [0, 1, 2, 3, 4, 5]
        assert defragmenter.bwd_mapping_table == [0, 1, 2, 3, 4, 5]

    def test_use_block(self, defragmenter):
        """Test block reference counting - use"""
        defragmenter.use_block(10)
        assert defragmenter.get_ref_count(10) == 1

        defragmenter.use_block(10)
        assert defragmenter.get_ref_count(10) == 2

        defragmenter.use_block(20)
        assert defragmenter.get_ref_count(20) == 1

    def test_free_block(self, defragmenter):
        """Test block reference counting - free"""
        defragmenter.use_block(10)
        defragmenter.use_block(10)
        assert defragmenter.get_ref_count(10) == 2

        defragmenter.free_block(10)
        assert defragmenter.get_ref_count(10) == 1

        defragmenter.free_block(10)
        assert defragmenter.get_ref_count(10) == 0
        assert 10 not in defragmenter.used_blocks

    def test_swap_refs(self, defragmenter):
        """Test swapping reference counts between blocks"""
        defragmenter.use_block(10)
        defragmenter.use_block(10)  # block 10: 2 refs
        defragmenter.use_block(20)  # block 20: 1 ref

        defragmenter.swap_refs(10, 20)

        # After swap: block 10 gets 20's refs (1), block 20 gets 10's refs (2)
        assert defragmenter.get_ref_count(10) == 1
        assert defragmenter.get_ref_count(20) == 2

        # Swapping with non-existent block (0 refs)
        defragmenter.swap_refs(20, 30)
        assert defragmenter.get_ref_count(20) == 0
        assert 20 not in defragmenter.used_blocks
        assert defragmenter.get_ref_count(30) == 2

    def test_resolve_no_mapping(self, defragmenter):
        """Test block ID resolution without mapping"""
        assert defragmenter.resolve(5) == 5
        assert defragmenter.resolve(100) == 100

    def test_resolve_with_mapping(self, defragmenter):
        """Test block ID resolution with mapping"""
        defragmenter._extend_mapping_table(10)
        defragmenter.update_mapping(5, 3)  # Map 5 -> 3

        assert defragmenter.resolve(5) == 3
        assert defragmenter.unresolve(3) == 5

    def test_resolve_all(self, defragmenter):
        """Test batch block ID resolution"""
        defragmenter._extend_mapping_table(10)
        defragmenter.update_mapping(5, 3)
        defragmenter.update_mapping(7, 2)

        block_tables = [[5, 6, 7], [8, 9]]
        resolved = defragmenter.resolve_all(block_tables)

        assert resolved == [[3, 6, 2], [8, 9]]

    def test_update_mapping(self, defragmenter):
        """Test mapping table updates"""
        defragmenter._extend_mapping_table(10)

        defragmenter.update_mapping(8, 2)
        assert defragmenter.fwd_mapping_table[8] == 2
        assert defragmenter.bwd_mapping_table[2] == 8

    def test_update_state_new_blocks(self, defragmenter):
        """Test updating state with new blocks"""
        new_blocks = {'req_1': [10, 11, 12], 'req_2': [20, 21]}

        defragmenter.update_state(new_blocks, [])

        assert defragmenter.req_blocks == {'req_1': [10, 11, 12], 'req_2': [20, 21]}
        assert defragmenter.get_ref_count(10) == 1
        assert defragmenter.get_ref_count(11) == 1
        assert defragmenter.get_ref_count(12) == 1
        assert defragmenter.get_ref_count(20) == 1
        assert defragmenter.get_ref_count(21) == 1

        # Mapping table extended to max block
        assert len(defragmenter.fwd_mapping_table) >= 22

    def test_update_state_finished_requests(self, defragmenter):
        """Test updating state when requests finish"""
        new_blocks = {'req_1': [10, 11]}
        defragmenter.update_state(new_blocks, [])

        assert defragmenter.get_ref_count(10) == 1
        assert defragmenter.get_ref_count(11) == 1

        # no blocks to add, mark req_1 as finished
        defragmenter.update_state({}, ['req_1'])

        assert 10 not in defragmenter.used_blocks
        assert 11 not in defragmenter.used_blocks
        assert 'req_1' not in defragmenter.req_blocks

    def test_free_blocks_generator(self, defragmenter):
        """Test free blocks generator yields correct sequence"""
        defragmenter.use_block(5)
        defragmenter.use_block(10)
        defragmenter.use_block(15)

        # yield 20 candidates to free
        free_gen = defragmenter.free_blocks()
        free_list = [next(free_gen) for _ in range(20)]

        # Should yield: 1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19,20,21,22,23
        assert free_list[:4] == [1, 2, 3, 4]
        assert 5 not in free_list
        assert free_list[4:8] == [6, 7, 8, 9]
        assert 10 not in free_list

    def test_defragment_disabled(self, mock_config, mock_debug_logger):
        """Test defragmentation when disabled"""
        mock_config.defrag = False
        defrag = OnlineDefragmenter()

        defrag.use_block(100)
        defrag.defragment()

        # No defragmentation should occur
        assert defrag.get_ref_count(100) == 1

    def test_defragment_no_blocks(self, defragmenter):
        """Test defragmentation with no used blocks"""
        defragmenter.defragment()
        # Should return early without errors
        assert defragmenter.used_blocks == {}

    def test_defragment_below_threshold(self, defragmenter):
        """Test defragmentation when fragmentation below threshold"""
        # Use blocks close together (max_used - threshold <= num_used)
        # With threshold=32, using blocks 1-19 means max_used=19, num_used=19
        # 19 - 32 = -13, which is <= 19, so no defrag
        for i in range(1, 20):
            defragmenter.use_block(i)

        max_before = max(defragmenter.used_blocks.keys())
        defragmenter._extend_mapping_table(max_before)
        defragmenter.cache_utils = MagicMock()
        defragmenter.defragment()

        # Should not trigger defragmentation
        defragmenter.cache_utils.swap.assert_not_called()
        assert max(defragmenter.used_blocks.keys()) == max_before

    def test_defragment_triggers(self, defragmenter):
        """Test defragmentation triggers when fragmented"""
        # Create fragmentation: blocks at 1, 2, 3 and 100, 101, 102
        # max_used=102, num_used=6, threshold=32
        # Check: 102 - 32 = 70 > 6, so triggers defrag
        for i in [1, 2, 3, 100, 101, 102]:
            defragmenter.use_block(i)

        defragmenter._extend_mapping_table(102)
        defragmenter.cache_utils = MagicMock()

        defragmenter.defragment()

        # Should call swap with high blocks moved to low positions
        defragmenter.cache_utils.swap.assert_called_once()
        args = defragmenter.cache_utils.swap.call_args[0]
        to_swap = args[0]
        threshold = args[1]

        # Verify swaps move high blocks to low free positions
        assert len(to_swap) > 0
        for high, low in to_swap:
            assert high > low

        # Verify threshold is padded correctly
        assert threshold in defragmenter.to_swap_pad_thresholds
        assert threshold >= len(to_swap)

    def test_defragment_early_exit(self, defragmenter):
        """Test defragmentation exits early when no swaps needed"""
        # Scenario where free_block > used_block early
        defragmenter.use_block(2)
        defragmenter.use_block(100)

        defragmenter._extend_mapping_table(100)
        defragmenter.cache_utils = MagicMock()

        defragmenter.defragment()

        # Free blocks: 1, 3, 4, 5...
        # Used blocks (descending): 100, 2
        # Pair (100, 1): valid swap
        # Pair (2, 3): 3 > 2, so break
        args = defragmenter.cache_utils.swap.call_args[0]
        to_swap = args[0]

        assert len(to_swap) == 1
        assert to_swap[0] == (100, 1)


class TestCacheSwapUtils:
    """Test suite for CacheSwapUtils"""

    @pytest.fixture
    def mock_kv_caches(self):
        """Create mock KV cache tensors"""
        num_blocks = 100
        block_size = 16
        num_heads = 8
        head_dim = 64
        num_layers = 2

        kv_caches = []
        for _ in range(num_layers):
            k_cache = torch.randn(num_blocks * block_size, num_heads, head_dim)
            v_cache = torch.randn(num_blocks * block_size, num_heads, head_dim)
            kv_caches.append((k_cache, v_cache))
        return tuple(kv_caches)

    @pytest.fixture
    def swap_utils(self, mock_kv_caches):
        """Create CacheSwapUtils instance"""
        with patch('vllm_gaudi.extension.defragmentation.htorch'):
            return CacheSwapUtils(mock_kv_caches, block_size=16)

    def test_cache_swap_utils_init(self, swap_utils):
        """Test CacheSwapUtils initialization"""
        assert swap_utils.block_size == 16
        assert len(swap_utils.kv_caches) == 2
        assert swap_utils.block_slots.shape == (16, )
        assert swap_utils.is_mla is False

    def test_cache_swap_utils_mla_detection(self):
        """Test MLA (multi-layer attention) detection"""
        # Create MLA-style caches (no value cache)
        mla_caches = [(torch.randn(100, 8, 64), None), (torch.randn(100, 8, 64), None)]

        with patch('vllm_gaudi.extension.defragmentation.htorch'):
            utils = CacheSwapUtils(tuple(mla_caches), block_size=16)
            assert utils.is_mla is True

    def test_swap_execution(self):
        """Test swap method execution flow on HPU"""
        import habana_frameworks.torch as htorch

        num_blocks = 100
        block_size = 16
        num_heads = 8
        head_dim = 64
        num_layers = 2

        kv_caches = []
        for _ in range(num_layers):
            k_cache = torch.randn(num_blocks * block_size, num_heads, head_dim, device='hpu')
            v_cache = torch.randn(num_blocks * block_size, num_heads, head_dim, device='hpu')
            kv_caches.append((k_cache, v_cache))

        swap_utils = CacheSwapUtils(tuple(kv_caches), block_size=16)

        to_swap = [(10, 5), (20, 6)]
        threshold = 8

        # Store original values to verify swap
        orig_k_10 = kv_caches[0][0][10 * block_size:(10 + 1) * block_size].clone()
        orig_k_5 = kv_caches[0][0][5 * block_size:(5 + 1) * block_size].clone()

        swap_utils.swap(to_swap, threshold)
        htorch.core.mark_step()

        # Verify blocks were swapped
        swapped_k_10 = kv_caches[0][0][10 * block_size:(10 + 1) * block_size]
        swapped_k_5 = kv_caches[0][0][5 * block_size:(5 + 1) * block_size]

        assert torch.allclose(swapped_k_10, orig_k_5)
        assert torch.allclose(swapped_k_5, orig_k_10)

    @patch('vllm_gaudi.extension.defragmentation.htorch')
    def test_swap_mla_single_call(self, mock_htorch):
        """Test MLA swap only calls forward once (no value cache)"""
        mla_caches = [(torch.randn(100, 8, 64), None), (torch.randn(100, 8, 64), None)]
        utils = CacheSwapUtils(tuple(mla_caches), block_size=16)

        to_swap = [(10, 5)]
        threshold = 8

        with patch.object(utils, 'forward') as mock_forward:
            utils.swap(to_swap, threshold)

            # Should only be called once for keys (no values)
            assert mock_forward.call_count == 1


class TestDefragmentationIntegration:
    """Integration tests for defragmentation workflow"""

    @pytest.fixture
    def setup_defragmenter(self, mock_config, mock_debug_logger):
        """Setup defragmenter with mock caches"""
        defrag = OnlineDefragmenter()

        # Create simple mock caches
        kv_caches = [(torch.zeros(1600, 8, 64), torch.zeros(1600, 8, 64)),
                     (torch.zeros(1600, 8, 64), torch.zeros(1600, 8, 64))]

        with patch('vllm_gaudi.extension.defragmentation.htorch'):
            defrag.initialize(tuple(kv_caches), block_size=16)

        return defrag

    def test_full_lifecycle(self, setup_defragmenter):
        """Test complete request lifecycle with defragmentation"""
        defrag = setup_defragmenter

        # Add requests
        defrag.update_state({'req_1': [10, 11, 12]}, [])
        defrag.update_state({'req_2': [50, 51]}, [])

        assert len(defrag.used_blocks) == 5
        assert len(defrag.req_blocks) == 2

        # Finish req_1
        defrag.update_state({}, ['req_1'])

        assert len(defrag.used_blocks) == 2
        assert len(defrag.req_blocks) == 1

        # Add more fragmented requests
        defrag.update_state({'req_3': [100, 101, 102]}, [])

        # Trigger defragmentation
        with patch.object(defrag.cache_utils, 'swap'):
            defrag.defragment()

    def test_mapping_persistence(self, setup_defragmenter):
        """Test that mappings persist across multiple defragmentations"""
        defrag = setup_defragmenter

        # Setup fragmented blocks
        for i in [1, 2, 3, 100]:
            defrag.use_block(i)

        defrag._extend_mapping_table(100)

        with patch.object(defrag.cache_utils, 'swap'):
            defrag.defragment()

        # Verify mappings exist
        assert len(defrag.fwd_mapping_table) > 0
        assert len(defrag.bwd_mapping_table) > 0

        # All used blocks should be resolvable
        for block_id in list(defrag.used_blocks.keys()):
            resolved = defrag.resolve(block_id)
            unresolve = defrag.unresolve(resolved)
            assert unresolve == block_id

    def test_finish_request_cancelled_before_blocks_allocated(self, setup_defragmenter):
        """Test that finishing a request that was never added to req_blocks
        does not crash (e.g. user Ctrl+C before blocks are allocated)."""
        defrag = setup_defragmenter

        # Add a normal request so there is some state
        defrag.update_state({'req_1': [10, 11]}, [])
        assert defrag.get_ref_count(10) == 1
        assert defrag.get_ref_count(11) == 1

        # Finish a request that was never registered (simulates Ctrl+C
        # cancellation before blocks were allocated)
        defrag.update_state({}, ['unknown_req'])

        # The server must survive — no KeyError
        # Existing state should be untouched
        assert defrag.get_ref_count(10) == 1
        assert defrag.get_ref_count(11) == 1
        assert 'req_1' in defrag.req_blocks
        assert 'unknown_req' not in defrag.req_blocks

    def test_finish_cancelled_and_normal_requests_together(self, setup_defragmenter):
        """Test finishing both a known and an unknown request in the same call."""
        defrag = setup_defragmenter

        defrag.update_state({'req_1': [10], 'req_2': [20]}, [])

        # Finish req_1 (known) and req_ghost (never allocated) together
        defrag.update_state({}, ['req_1', 'req_ghost'])

        # req_1 should be cleaned up properly
        assert 'req_1' not in defrag.req_blocks
        assert defrag.get_ref_count(10) == 0

        # req_2 should be unaffected
        assert 'req_2' in defrag.req_blocks
        assert defrag.get_ref_count(20) == 1

        # req_ghost should not appear anywhere
        assert 'req_ghost' not in defrag.req_blocks
