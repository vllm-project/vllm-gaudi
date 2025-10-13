from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm_gaudi.extension.logger import logger as init_logger

from types import FunctionType
from functools import wraps
import os

logger = init_logger()


def ensure_valid_and_sorted(first_block: KVCacheBlock, last_block: KVCacheBlock):
    current_block = first_block
    fwd_blocks = []
    while current_block is not None and current_block != last_block:
        if current_block.next_free_block is None:
            raise RuntimeError("The blocks are not properly linked.")
        if current_block.block_id > current_block.next_free_block.block_id:
            raise RuntimeError("The blocks are not sorted by block_id.")
        fwd_blocks.append(current_block)
        current_block = current_block.next_free_block
    fwd_blocks.append(current_block)
    bwd_blocks = []
    current_block = last_block
    while current_block is not None and current_block != first_block:
        if current_block.prev_free_block is None:
            raise RuntimeError("The blocks are not properly linked.")
        if current_block.block_id < current_block.prev_free_block.block_id:
            raise RuntimeError("The blocks are not sorted by block_id.")
        bwd_blocks.append(current_block)
        current_block = current_block.prev_free_block
    bwd_blocks.append(current_block)
    bwd_blocks.reverse()
    if fwd_blocks != bwd_blocks:
        raise RuntimeError("The blocks are not properly linked.")


def wrapper(method):

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block
        ensure_valid_and_sorted(first_block, last_block)
        out = method(self, *args, **kwargs)

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block
        ensure_valid_and_sorted(first_block, last_block)
        return out

    return wrapped


class DebuggingMetaClass(type):

    def __new__(meta, classname, bases, classDict):
        enable_debugging = os.environ.get("VLLM_HPU_BLOCK_QUEUE_DEBUG",
                                          '').lower() in ("y", "yes", "t", "true", "on", "1")
        if enable_debugging:
            logger.warning("HPUFreeKVCacheBlockQueue debugging is enabled. This will impact the performance severely.")
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            # Wrap non-constructor methods with the debugging wrapper checking for sortedness
            if enable_debugging and isinstance(attribute, FunctionType) and attributeName != '__init__':
                attribute = wrapper(attribute)
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)


class HPUFreeKVCacheBlockQueue(metaclass=DebuggingMetaClass):
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        msg = f"Using {type(self).__name__}"
        logger.info(msg)
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if (self.fake_free_list_head.next_free_block is self.fake_free_list_tail
                or self.fake_free_list_head.next_free_block is None):
            assert self.num_free_blocks == 0, (f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                                               "with the free list.")
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError("Invalid block found in popleft() "
                               "which doesn't have a valid next_free_block")

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def _append_single_block(self, block: KVCacheBlock) -> None:
        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block
        # Connect the new block, preserving the order.
        current_block = first_block
        # NOTE(kzawora): this will come crashing spectacularly hard if current_block
        # happens to be None, but that should never happen, since the only block that's
        # allowed to be None is the fake tail (last_block).
        while current_block != last_block and current_block.block_id < block.block_id:
            current_block = current_block.next_free_block
        # Update the prev and next pointers of block and its neighbors
        # current_block is the first block with block_id > block.block_id
        block.prev_free_block = current_block.prev_free_block
        block.prev_free_block.next_free_block = block
        block.next_free_block = current_block
        block.next_free_block.prev_free_block = block

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError("prev_free_block of fake_free_list_tail should always exist")
        self._append_single_block(block)

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """

        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, ("prev_free_block of fake_free_list_tail should always exist")
        # Add inter-connections between consecutive blocks
        for block in blocks:
            # NOTE(kzawora): this is a shortcut and might be inefficient, but should work
            self._append_single_block(block)

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError("next_free_block of fake_free_list_head should always exist")
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret
