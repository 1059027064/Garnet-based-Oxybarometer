# Garnet-based Machine Learning Oxybarometer

This repository provides a machine learning workflow for predicting **pressure (P)**, **temperature (T)**, and **oxygen fugacity (logfO2)** from garnet compositions.

The project consists of two main components:

- `Train/`: model training, evaluation, SHAP analysis, and figure generation  
- `Prediction/`: standalone prediction tool using pre-trained models  

A shared Conda environment is provided:

- `environment_full.yml`

---

## Repository Structure

```text
.
├── Code/
│   ├── Train/
│   │   ├── dataset/
│   │   │   ├── df_fO2_test.xlsx
│   │   │   ├── df_fO2_train.xlsx
│   │   │   ├── df_P_T_test.xlsx
│   │   │   ├── df_P_T_train.xlsx
│   │   │   ├── predicted-Diamond.xlsx
│   │   │   └── predicted-Mantle.xlsx
│   │   ├── images/
│   │   ├── models/
│   │   ├── results/
│   │   │   └── SHAP_results/
│   │   └── Model Training.ipynb
│   │
│   ├── Prediction/
│   │   ├── input/
│   │   │   └── input_samples.xlsx
│   │   ├── models/
│   │   │   ├── P_model_tabpfn.pkl
│   │   │   ├── T_model_tabpfn.pkl
│   │   │   └── fO2_model_tabpfn.pkl
│   │   ├── train_dataset/
│   │   │   ├── df_P_T_train.xlsx
│   │   │   └── df_fO2_train.xlsx
│   │   ├── output/
│   │   ├── images/
│   │   ├── predict.py
│   │   └── run_prediction.bat
│   │
│   └── environment_full.yml
│
├── README.md
└── Dataset.xlsx
```

---

## Overview

- Environment Setup  
- Training Workflow
- Prediction Workflow

---

## Environment Setup

### Step 1. Create environment

```bash
conda env create -f environment_full.yml
```

### Step 2. Activate environment

```bash
conda activate tabpfn_shap
```

### Step 3. Install PyTorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 4. Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## Training Workflow

```text
Train/Model Training.ipynb
```

### Required datasets

```text
Train/dataset/
```

- df_P_T_train.xlsx  
- df_P_T_test.xlsx  
- df_fO2_train.xlsx  
- df_fO2_test.xlsx  

### Outputs

- Train/models/  
- Train/results/  
- Train/images/  
- Train/results/SHAP_results/  

---

## Prediction Workflow

### Input

```text
Prediction/input/input_samples.xlsx
```


Input Format

- Si_Grt  
- Ti_Grt  
- Al_Grt  
- Cr_Grt  
- Fe_Grt  
- Mn_Grt  
- Mg_Grt  
- Ca_Grt  
- Na_Grt  


### Models

```text
Prediction/models/
```


### Training data (OOD)

```text
Prediction/train_dataset/
```

- df_P_T_train.xlsx  
- df_fO2_train.xlsx  


### Prediction Pipeline

1. Read input  
2. Convert to garnet cations  
3. Apply cation filter  
4. Predict P, T, logfO2  
5. Calculate IW, FMQ, NNO, MH  
6. Compute ΔIW, ΔFMQ, ΔNNO, ΔMH  
7. OOD detection  
8. PCA visualization  


### Running Prediction


```bash
conda activate tabpfn_shap
cd Prediction
python predict.py
```

or

```text
Double click: run_prediction.bat
```


### Output

```text
Prediction/output/
Prediction/images/
```

Includes:

- P, T, logfO2  
- IW, FMQ, NNO, MH  
- ΔIW, ΔFMQ, ΔNNO, ΔMH  
- OOD Categories 


### OOD Categories

- In-Distribution  
- P-T OOD  
- Feature OOD  
- Feature range OOD  

---

## Citation

- 
