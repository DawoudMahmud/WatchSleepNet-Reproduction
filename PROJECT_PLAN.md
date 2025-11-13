# WatchSleepNet Reproduction Plan

## Objective
Reproduce the results of WatchSleepNet (CHIL 2025) and evaluate transfer learning on DREAMT dataset.

## Steps
1. **Data Preparation**
   - Extract IBI/PPG and align 30s epochs
   - Label sleep stages (Wake, NREM, REM)
   - Normalize signals and save as tensors
2. **Model Implementation**
   - Rebuild WatchSleepNet architecture
   - Verify model compiles and outputs correct shapes
3. **Training**
   - Fine-tune pretrained weights (if available)
   - Experiment with batch sizes, learning rates
4. **Evaluation**
   - Compare results (Accuracy, F1, κ) with paper
   - Run ablation: no pretraining vs pretrained
5. **Report**
   - Document reproducibility challenges and findings
