import os
import torch
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
                #multimodal_buckets = \
                    #self.get_buckets_from_lists_with_ratios([480], 16)
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
        final_h = h_orig + best_pad_h  
        final_w = w_orig + best_pad_w  
            
        if final_h % merge_size != 0:  
            best_pad_h += merge_size - (final_h % merge_size)
        final_w = desired_patches 
        if final_w % merge_size != 0:
            best_pad_w += merge_size - (final_w % merge_size)  
      
        return best_pad_h, best_pad_w

    def find_padding2(self, h_orig: int, w_orig: int, desired_patches: int) -> tuple[int, int]:
        """Direct calculation without explicit loops - match h_adj and calculate w_adj."""
        merge_size = 2
        # Calculate target aspect ratio preservation
        target_ratio = h_orig / w_orig

        # Calculate height based on aspect ratio and desired patches
        # h * w = desired_patches and h/w = target_ratio
        # h = sqrt(desired_patches * target_ratio)
        h_target = int((desired_patches * target_ratio) ** 0.5)
        
        # Adjust height for merge_size divisibility
        h_adj = ((h_target + merge_size - 1) // merge_size) * merge_size
        # Calculate width to achieve exact patch count
        w_adj = desired_patches // h_adj
        # Ensure width is also divisible by merge_size
        if w_adj % merge_size != 0:
            w_adj = ((w_adj + merge_size - 1) // merge_size) * merge_size
        # Final verification - adjust if needed
        if h_adj * w_adj != desired_patches:
            # If still not matching, try the reverse approach
            w_target = int((desired_patches / target_ratio) ** 0.5)
            w_adj = ((w_target + merge_size - 1) // merge_size) * merge_size
            h_adj = desired_patches // w_adj
            if h_adj % merge_size != 0:
                h_adj = ((h_adj + merge_size - 1) // merge_size) * merge_size

        # Final check
        if h_adj * w_adj != desired_patches:
            print(f"libin debug padding 0")
            return 0, 0
        print(f"libin debug padding {h_adj - h_orig}, {w_adj - w_orig}")
        return h_adj - h_orig, w_adj - w_orig

    def pad_multimodal_data(self, pixel_values, image_grid_thw):
        # Only position 0 is dynamic
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
        if self.qwen2_5_vl:
            pad_h, pad_w = self.find_padding(h_orig, w_orig, desired_number_of_pixels)
        else:
            pad_h, pad_w = self.find_padding2(h_orig, w_orig, desired_number_of_pixels)
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
        print(f"libin debug padded {image_grid_thw.prod(-1).sum()=}{desired_number_of_pixels=} {image_grid_thw=} ")
        assert image_grid_thw.prod(-1).sum() == desired_number_of_pixels
        print(f"libin debug padded {image_grid_thw=} {desired_number_of_pixels=}")
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

    def bucket_to_image_resolution(self,
        target_patches: int,
        ratio_w: int,
        ratio_h: int,
        patch_size: int = 14
    ) -> tuple[int, int]:
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
        max_scale = int((target_patches  / (ratio_w * ratio_h)) ** 0.5)
        print(f"libin debug bucket_to_image_resolution1 {ratio_w=} {ratio_h=} {target_patches=}")
        for scale in range(max_scale, 0, -1):
            grid_w = ratio_w * scale
            grid_h = ratio_h * scale
            print(f"libin debug bucket_to_image_resolution2 {scale=} {grid_w=}{grid_h=} {grid_h * grid_w=} ")
            if grid_w * grid_h <= target_patches:
                break
        # Convert grid dimensions to pixel dimensions
        width = grid_w * patch_size
        height = grid_h * patch_size
        print(f"libin debug bucket_to_image_resolution3 {grid_w=} {grid_h=} {width=} {height=} ")
        return width, height

    def _patches_per_image(self, width: int, height: int, patch_size: int = 14):
        # Calculate patches 
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches_per_image = grid_h * grid_w
        print(f"libin debug {height=} {width=} {grid_h=} {grid_w=} {patches_per_image=}")
        return patches_per_image

    def add_to_bucket(self, width: int, height: int, patch_size: int = 14):
        patches_per_image = self._patches_per_image(width, height, patch_size)
        if value not in lst:
            self.multimodal_buckets.append(patches_per_image)
            self._process_buckets()
    '''
    def get_buckets_from_lists_with_ratios(self,
        heights: list[int],   
        patch_size: int = 14 
    ) -> list[int]:
        # Total patches = (height/patch_size) * (width/patch_size)
        ratio_wh =[(1, 1), (3, 4), (4, 3), (16, 9), (9,16)]
        buckets = []
        for ratios in ratio_wh:
            for height in heights:
                ratio_w, ratio_h = ratios
                # Calculate width from height and ratio  
                width = int(height * ratio_w / ratio_h)

                patches_per_image = self._patches_per_image(width, height, patch_size)
                buckets.append(patches_per_image)
                print(f"libin debug get_bucket_from_list {ratio_w=} {ratio_h=} {height=} {width=} {patches_per_image=} ")
        print(f"debug bucket = {buckets=}")
        return buckets
        '''