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
        #coverage for lmarena-ai/VisionArena-Chat
        'buckets': [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 131076]
    }
}

class HPUVisionBucketManager:
    '''
    This class is used to bucket image tokens
    '''

    def __init__(self, model_name, is_batch_based=None):
        config = self._get_multimodal_config(model_name)

        self.is_batch_based = is_batch_based if is_batch_based is not None else config['is_batch_based']

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
        best_pad_h, best_pad_w = 0, 0
        if desired_patches % h_orig == 0:
            best_pad_h = 0
            w_factor = desired_patches // h_orig
            best_pad_w = w_factor - w_orig if (w_factor > w_orig and w_factor % 2 == 0) else 0
        elif desired_patches % w_orig == 0:
            best_pad_w = 0
            h_factor = desired_patches // w_orig
            best_pad_h = h_factor - h_orig if (h_factor > h_orig and h_factor % 2 == 0) else 0
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

    def pad_multimodal_data(self, pixel_values, image_grid_thw):
        desired_number_of_pixels = self.get_multimodal_bucket(pixel_values.shape[0])
        padding_len = desired_number_of_pixels - pixel_values.shape[0]
        if padding_len <= 0:
            return pixel_values, image_grid_thw

        logger_msg = "Padding current number pixel " \
            + str(pixel_values.shape[0]) \
            + " to " \
            + str(desired_number_of_pixels)
        logger.info(logger_msg)

        h_orig, w_orig = image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item()
        pad_h, pad_w = self.find_padding(h_orig, w_orig, desired_number_of_pixels)
        if pad_h == 0 and pad_w == 0:
            return pixel_values, image_grid_thw

        constant_value = -100
        pixel_values = torch.cat([
            pixel_values,
            torch.ones((padding_len, pixel_values.shape[1]), device=pixel_values.device) * constant_value
        ])

        image_grid_thw = torch.tensor([[1, h_orig + pad_h, w_orig + pad_w]],
                                      device=image_grid_thw.device,
                                      dtype=image_grid_thw.dtype)

        assert image_grid_thw.prod(-1).sum() == desired_number_of_pixels
        return pixel_values, image_grid_thw

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
