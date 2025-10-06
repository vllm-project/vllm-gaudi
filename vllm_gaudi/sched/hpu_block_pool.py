from vllm.v1.core.block_pool import BlockPool, BlockHashToBlockMap
from vllm_gaudi.extension.logger import logger as init_logger
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock, )
from vllm.distributed.kv_events import (
    KVCacheEvent, )
from vllm_gaudi.sched.hpu_block_queue import HPUFreeKVCacheBlockQueue

logger = init_logger()


class HPUBlockPool(BlockPool):

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        msg = f"Using {type(self).__name__}"
        logger.info(msg)
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        #from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
        #elf.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        self.free_block_queue = HPUFreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []
