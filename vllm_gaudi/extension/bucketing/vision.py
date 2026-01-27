import os
from vllm.logger import init_logger

logger = init_logger(__name__)

MULTIMODAL_CONFIG = {
    # Batch-based models
    'gemma-3': {
        'is_batch_based': True,
        'buckets': [1, 2, 4, 8]
    },

    # Pixel-based models
    'ovis2.5': {
        'is_batch_based': False,
        'buckets': [784, 1600, 3136, 4096, 6400, 7744, 9216, 12544]
    },
    'qwen2_5_vl': {
        'is_batch_based': False,
        'buckets': [1600, 3136, 4096, 6400, 7744, 9216, 12544]
    },
    'qwen3_vl': {
        'is_batch_based': False,
        # patches per image
        'buckets': [256, 480, 512, 660, 900, 1024, 1200, 1590, 2048, 3520]
    }
}

class HPUVisionBucketManager:
    '''
    This class is used to bucket image tokens
    '''

    def __init__(self, model_name, is_batch_based=None):
        config = self._get_multimodal_config(model_name)

        self.is_batch_based = is_batch_based if is_batch_based is not None else config['is_batch_based']
        
        self.qwen2_5_vl = 'qwen2_5_vl' in model_name.lower()

        envvar = os.environ.get('VLLM_MULTIMODAL_BUCKETS', "")

        if envvar == 'None':
            self.multimodal_buckets = None
        else:
            if envvar == "":
                multimodal_buckets = config['buckets']
            else:
                multimodal_buckets = [int(x) for x in envvar.split(',')]
            self.multimodal_buckets = self._process_buckets(multimodal_buckets)

    def _get_multimodal_config(self, model_name):
        """Get configuration for model"""
        model_name_lower = model_name.lower()

        # Find matching config
        for key, config in MULTIMODAL_CONFIG.items():
            if key.replace('-', '').replace('.', '') in model_name_lower.replace('-', '').replace('.', ''):
                return config

        # Default config
        logger.info(f"MultiModal bucket config file for {model_name} not found.")
        return {'is_batch_based': True, 'buckets': [1, 2, 4, 8]}

    def _process_buckets(self, buckets):
        #TODO If there is any limitation(such as if batch bucket need to be aligned by n, then put the assert check here!)

        return sorted(buckets)

    def get_multimodal_bucket(self, curr_num_image_patches):
        if self.multimodal_buckets is not None:
            for mm_bucket in self.multimodal_buckets:
                if curr_num_image_patches <= mm_bucket:
                    return mm_bucket
            return curr_num_image_patches
        else:
            return 0

    def find_factor(self, desired_patches, orig):
        for i in range(orig + 1, desired_patches + 1):
            if desired_patches % i == 0:
                if i % 2 != 0:
                    continue
                else:
                    return i
        return None

    def find_padding(self, h_orig, w_orig, desired_patches):
        merge_size = 2
        best_pad_h, best_pad_w = 0, 0
        if desired_patches % h_orig == 0:
            best_pad_h = 0
            w_factor = desired_patches // h_orig
            best_pad_w = w_factor - w_orig if (w_factor > w_orig and w_factor % merge_size == 0) else 0
        elif desired_patches % w_orig == 0:
            best_pad_w = 0
            h_factor = desired_patches // w_orig
            best_pad_h = h_factor - h_orig if (h_factor > h_orig and h_factor % merge_size == 0) else 0
        elif desired_patches % h_orig != 0 and desired_patches % w_orig != 0:
            if h_orig > w_orig:
                w_factor = self.find_factor(desired_patches, w_orig)
                if w_factor is not None:
                    best_pad_w = w_factor - w_orig
                    h_factor = desired_patches // w_factor
                    if h_factor > h_orig:
                        best_pad_h = h_factor - h_orig
            else:
                h_factor = self.find_factor(desired_patches, h_orig)
                if h_factor is not None:
                    best_pad_h = h_factor - h_orig
                    w_factor = desired_patches // h_factor
                    if w_factor > w_orig:
                        best_pad_w = w_factor - w_orig

        if (best_pad_h + h_orig) * (best_pad_w + w_orig) != desired_patches:
            best_pad_h, best_pad_w = 0, 0

        return best_pad_h, best_pad_w

    def greedy_plan(self, batchsize, available_batchsizes):
        # sort descending
        available_batchsizes_sorted = sorted(available_batchsizes, key=lambda x: -x)
        idx = 0
        left_to_process = batchsize
        result = []
        while (left_to_process > 0 and idx < len(available_batchsizes_sorted)):
            if available_batchsizes_sorted[idx] <= left_to_process:
                result += [available_batchsizes_sorted[idx]]
                left_to_process -= available_batchsizes_sorted[idx]
            else:
                idx += 1
        if left_to_process > 0:
            result += [available_batchsizes_sorted[-1]]  # this will be padded
        return result

    def __repr__(self):
        return str(self.multimodal_buckets)

    def bucket_to_image_resolution(self,
                                   target_patches: int,
                                   ratio_w: int,
                                   ratio_h: int,
                                   patch_size: int = 14) -> tuple[int, int]:
        """
        Convert bucket patch count to image resolution for specific aspect ratio.
        Assumption is patch number are the same for each image
        Args:
            target_patches: Number of patches from bucket
            ratio_w, ratio_h: Target aspect ratio (width:height)
            patch_size: Vision model patch size (default 14 for Qwen3VL)

        Returns:
            (width, height) in pixels
        """
        # Find largest scale that fits within patch budget
        max_scale = int((target_patches / (ratio_w * ratio_h)) ** 0.5)
        for scale in range(max_scale, 0, -1):
            grid_w = ratio_w * scale
            grid_h = ratio_h * scale
            if grid_w * grid_h <= target_patches:
                break
        # Convert grid dimensions to pixel dimensions
        width = grid_w * patch_size
        height = grid_h * patch_size
        return width, height

    def _patches_per_image(self, width: int, height: int, patch_size: int = 14):
        # Calculate patches
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches_per_image = grid_h * grid_w
        return patches_per_image

    def add_to_bucket(self, width: int, height: int, patch_size: int = 14):
        patches_per_image = self._patches_per_image(width, height, patch_size)
        if value not in lst:
            self.multimodal_buckets.append(patches_per_image)
            self._process_buckets()
