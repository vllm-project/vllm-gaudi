from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm_gaudi.sched.hpu_kv_cache_coordinator import (get_kv_cache_coordinator)
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HPUKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ) -> None:
        msg = f"Using {type(self).__name__}"
        logger.info(msg)
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self.block_pool = self.coordinator.block_pool
