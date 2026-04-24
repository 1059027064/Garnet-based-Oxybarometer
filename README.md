# Garnet-based Machine Learning Oxybarometer

> **System Notice:** This repository was developed and validated in a Windows environment using NVIDIA CUDA-enabled GPUs (GeForce RTX 4070 and RTX 2060). Compatibility with other operating systems or hardware configurations has not been comprehensively verified.

This repository provides a machine learning workflow for predicting **pressure (P)**, **temperature (T)**, and **oxygen fugacity (logfO2)** from garnet compositions.

The repository contains two main components:

- `Code/Train/` — model training, evaluation, SHAP analysis, and figure generation  
- `Code/Prediction/` — standalone prediction workflow using pre-trained models  

A shared Conda environment file is provided:

- `Code/environment_full.yml`

---

## Repository Structure

```text
.
├── Code/
│   ├── Train/
│   │   ├── dataset/
│   │   ├── images/
│   │   ├── models/
│   │   ├── results/
│   │   │   └── SHAP_results/
│   │   └── Model Training.ipynb
│   │
│   ├── Prediction/
│   │   ├── input/
│   │   ├── models/
│   │   ├── train_dataset/
│   │   ├── output/
│   │   ├── images/
│   │   ├── predict.py
│   │   └── run_prediction.bat
│   │
│   └── environment_full.yml
│
├── Dataset.xlsx
└── README.md
```

---

## Overview

1. Environment setup  
2. Model training  
3. Prediction workflow  

---

## Environment Setup

### Step 1. Create the Conda environment

```bash
conda env create -f Code/environment_full.yml
```

### Step 2. Activate the environment

```bash
conda activate tabpfn_shap
```

### Step 3. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 4. Verify GPU availability

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## Model Training Workflow

Training notebook:

```text
Code/Train/Model Training.ipynb
```

### Required datasets

```text
Code/Train/dataset/
```

Files include:

- `df_P_T_train.xlsx`  
- `df_P_T_test.xlsx`  
- `df_fO2_train.xlsx`  
- `df_fO2_test.xlsx`

### Training outputs

Generated files will be saved to:

- `Code/Train/models/`  
- `Code/Train/results/`  
- `Code/Train/images/`  
- `Code/Train/results/SHAP_results/`

---

## Prediction Workflow

### Input file

```text
Code/Prediction/input/input_samples.xlsx
```

### Required input variables

- `Si_Grt`  
- `Ti_Grt`  
- `Al_Grt`  
- `Cr_Grt`  
- `Fe_Grt`  
- `Mn_Grt`  
- `Mg_Grt`  
- `Ca_Grt`  
- `Na_Grt`

### Pre-trained models

```text
Code/Prediction/models/
```

### Reference training datasets (for OOD detection)

```text
Code/Prediction/train_dataset/
```

Files include:

- `df_P_T_train.xlsx`  
- `df_fO2_train.xlsx`

### Prediction pipeline

1. Read input data  
2. Convert oxide compositions to garnet cations  
3. Apply cation filtering  
4. Predict **P**, **T**, and **logfO2**  
5. Calculate reference oxygen buffers (**IW**, **FMQ**, **NNO**, **MH**)  
6. Compute ΔIW, ΔFMQ, ΔNNO, and ΔMH  
7. Perform out-of-distribution (OOD) detection  
8. Generate PCA visualization

### Run prediction

```bash
conda activate tabpfn_shap
cd Code/Prediction
python predict.py
```

or run:

```text
run_prediction.bat
```

### Output files

Results will be written to:

```text
Code/Prediction/output/
Code/Prediction/images/
```

Outputs include:

- Predicted **P**, **T**, **logfO2**  
- Calculated **IW**, **FMQ**, **NNO**, **MH**  
- ΔIW, ΔFMQ, ΔNNO, ΔMH  
- OOD classification results

### OOD categories

- In-Distribution  
- P-T OOD  
- Feature OOD  
- Feature Range OOD  

---

## Reproducibility

This workflow was tested using Windows systems equipped with NVIDIA GeForce RTX 4070 and RTX 2060 GPUs. The provided Conda environment file is intended to facilitate reproduction of the reported results.

---

## Citation

If you use this repository in academic research, please cite the associated publication.

*Citation details will be updated upon publication.*
