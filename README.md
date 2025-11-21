# WatchSleepNet Reproducibility Project (CSE6250 - Fall 2025)

## Objective
Reproduce the results from *WatchSleepNet: A Novel Model and Pretraining Approach for Advancing Sleep Staging with Smartwatches* (Wang et al., CHIL 2025).

## Team K4
- Dawoud Mahmud
- Vijay Gopalsamy

## Goals
- Reimplement the WatchSleepNet architecture (CNN + TCN + BiLSTM + Attention)
- Reproduce fine-tuning results on the DREAMT dataset
- Compare with baseline sleep staging models
- Evaluate transfer learning improvements

## Setup

### Using Conda (Recommended for Mac)
```bash
conda create -n watchsleepnet python=3.10 -y
conda activate watchsleepnet
conda install pytorch torchvision numpy pandas scipy matplotlib seaborn scikit-learn jupyter tqdm -c pytorch -y
python -m ipykernel install --user --name=watchsleepnet
```

### Using pip
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
WatchSleepNet-Reproduction/
│
├── configs/ # Config files for experiments and paths
├── data/ # Data folders (not tracked in Git)
│ ├── raw/ # Original DREAMT dataset
│ ├── interim/ # Intermediate extracted IBI or cleaned data
│ └── processed/ # Final tensors ready for training
│
├── notebooks/ # Colab/Jupyter notebooks for exploration
│
├── src/ # Main source code
│ ├── data/ # Preprocessing and data loading
│ ├── models/ # WatchSleepNet architecture
│ ├── training/ # Training and fine-tuning scripts
│ ├── eval/ # Evaluation metrics and plots
│ └── utils/ # Helper functions and configs
│
├── reports/ # Report drafts, figures, and final paper
├── requirements.txt # Python dependencies
└── README.md # Project overview and documentation
```

## Expected Deliverables
1. Preprocessing scripts for DREAMT data  
2. WatchSleepNet model implementation  
3. Training and evaluation pipeline  
4. Final report and presentation (following BDH guidelines)
