###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import habana_frameworks.torch.hpu as hthpu


def detect_hpu():
    return hthpu.get_device_name()[-1]


if __name__ == "__main__":
    print(detect_hpu())
