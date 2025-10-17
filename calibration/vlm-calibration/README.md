# FP8 Calibration Procedure for VLM models

The model calibration procedure for LLM models is a little bit different than VLM, we need to change it to adapt VLM models. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **dir to the source dataset:** It's path of HuggingFace Cache dir, which is used to store various calibration dataset. We currently hard-coded the calibration dataset to MMMU datasets. We add some valid check, please make sure the provided dataset path meet any of the following conditions: (1) The provided dir contain raw MMMU dataset and processed MMMU dataset. Specially, the dir of dataset has two subfolders, `${your_dataset_dir}/hub` contains raw datasets, `${your_dataset_dir}/datasets` contains processed datasets. (2) The provided dir `${your_dataset_dir}` contains processed datasets. (3) If users don't provided the dataset path, we will directly download it from huggingface.
- `-o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.
- `-t`, i.e., **tensor parallel size:** Tensor parallel size to run at.

Here are some examples of how to use the script:

```bash
cd vlm-calibration
./calibrate_model.sh \
    -m $MODEL_PATH \
    -o $INC_OUTPUT_PATH \
    -t $TP_SIZE \
    -d $DATASET_PATH
```
