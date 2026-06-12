# CIFAR-100 Image Recognition Benchmark

This repository compares classical image-processing features and modern deep visual models on CIFAR-100.

The project started as an image processing course experiment and was extended into a compact benchmark that covers:

- `pca_svm`: grayscale pixels + PCA + Linear SVM
- `hog_svm`: HOG + color histogram features + Linear SVM
- `resnet18`: ResNet18 adapted for CIFAR images
- `convnext_tiny`: ImageNet-pretrained ConvNeXt-Tiny transfer learning

## Final Results

The final run was trained on CIFAR-100 for 20 epochs on a Tesla V100 cloud GPU.

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| ResNet18 | **74.71%** | **74.90%** | **74.71%** | **74.67%** |
| ConvNeXt-Tiny | 73.93% | 74.25% | 73.93% | 73.92% |
| HOG + SVM | 25.92% | 22.10% | 25.92% | 22.60% |
| PCA + SVM | 9.02% | 7.20% | 9.02% | 5.58% |

See [RESULTS.md](RESULTS.md) for training curves, confusion analysis, and interpretation.

## Result Preview

### Deep Model Training Curves

![ResNet18 training curve](results/cifar100_frontier/training_curve_resnet18.png)

![ConvNeXt-Tiny training curve](results/cifar100_frontier/training_curve_convnext_tiny.png)

### Confusion Matrices

![ResNet18 confusion matrix](results/cifar100_frontier/confusion_matrix_resnet18.png)

![ConvNeXt-Tiny confusion matrix](results/cifar100_frontier/confusion_matrix_convnext_tiny.png)

## Project Structure

```text
.
├── README.md
├── RESULTS.md
├── requirements.txt
├── scripts/
│   └── run_experiments.py
├── src/
│   ├── classical.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── plots.py
│   └── train.py
└── results/
    └── cifar100_frontier/
```

The `data/`, `outputs/`, and model checkpoint files are intentionally excluded from Git. CIFAR datasets download automatically through `torchvision`.

## Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your driver before installing the remaining packages. For example, on a CUDA 11.8-compatible server:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn scikit-image opencv-python matplotlib pandas tqdm
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Quick Tests

CIFAR-10 smoke test:

```bash
python scripts/run_experiments.py --methods pca_svm hog_svm resnet18 --train-limit 1000 --test-limit 500 --epochs 1
```

CIFAR-100 ConvNeXt-Tiny smoke test:

```bash
python scripts/run_experiments.py --dataset cifar100 --methods convnext_tiny --pretrained --train-limit 256 --test-limit 128 --epochs 1 --batch-size 16 --output-dir outputs/cifar100_smoke
```

## Full Experiments

Full CIFAR-100 comparison:

```bash
python scripts/run_experiments.py --dataset cifar100 --methods pca_svm hog_svm resnet18 convnext_tiny --pretrained --epochs 20 --batch-size 32 --output-dir outputs/cifar100_frontier
```

ConvNeXt-Tiny low-learning-rate follow-up:

```bash
python scripts/run_experiments.py --dataset cifar100 --methods convnext_tiny --pretrained --epochs 30 --batch-size 32 --lr 0.0003 --output-dir outputs/cifar100_convnext_lr3e4_ep30
```

ModelArts example:

```bash
cd /home/ma-user/work/image-processing-final
/home/ma-user/anaconda3/envs/image-final-gpu/bin/python scripts/run_experiments.py --dataset cifar100 --methods pca_svm hog_svm resnet18 convnext_tiny --pretrained --epochs 20 --batch-size 32 --output-dir outputs/cifar100_frontier
```

## Outputs

Each run writes:

- `metrics.csv`: accuracy, macro precision, macro recall, macro F1, and per-class accuracy.
- `metrics_summary.csv`: grouped mean and standard deviation when multiple seeds are used.
- `training_curve_*.png`: train/validation loss and accuracy curves.
- `training_history_*.csv`: per-epoch training history.
- `confusion_matrix_*.png`: confusion matrix visualization.
- `top_confusions_*.csv`: largest off-diagonal confusion pairs for large-class datasets.
- `sample_predictions_*.png`: sample predictions.
- `misclassified_*.png`: misclassified samples.
- `checkpoints/best_*.pt`: best model checkpoint, excluded from Git.

## Notes

- CIFAR-100 has 100 classes, so dense confusion-matrix cell labels are omitted for readability.
- Deep models automatically use CUDA when PyTorch can see a compatible GPU.
- Checkpoints are intentionally excluded because they are large; use GitHub Releases or Git LFS if you need to publish them.
