
# Medical Image to Text Report Generation

## Project Overview
This project implements a deep learning model that generates textual descriptions (radiology reports) from chest X-ray images. It uses a CNN+Transformer architecture to learn the mapping between visual patterns in chest X-rays and corresponding medical language.

---

## Table of Contents
- Project Description
- Model Architecture
- Requirements
- Dataset
- Project Structure
- Installation
- Usage
  - Data Preparation
  - Training
  - Inference
- Results
- Future Improvements
- References

---

## Project Description
Radiological report generation is a time-consuming task for medical professionals. This project aims to automate this process using deep learning, providing a tool that could potentially assist radiologists by generating initial draft reports. The system takes a chest X-ray image as input and produces a textual report that describes the findings in medical language.

---

## Model Architecture

### Image Encoder
- **DenseNet-121**: Pre-trained on ImageNet and fine-tuned on chest X-rays
- Outputs feature maps of size **14×14×1024**
- Adaptive pooling followed by linear projection to embedding dimension

### Text Decoder
- **Transformer**: 6-layer transformer decoder
- 8 attention heads per layer
- Embedding dimension: 512
- Feed-forward dimension: 2048
- Dropout: 0.2

> Total Parameters: **34.64M**

---

## Requirements

```txt
python>=3.8
torch>=1.10.0
torchvision>=0.11.0
nltk>=3.6.5
pillow>=8.3.2
matplotlib>=3.4.3
tqdm>=4.62.3
numpy>=1.21.2
```

---

## Dataset
- **Name**: Indiana University Chest X-ray Collection (IU X-Ray dataset)
- **Images**: 7,470 chest X-rays (frontal and lateral views)
- **Reports**: 3,955 corresponding reports
- **Access**: [OpenI NIH](https://openi.nlm.nih.gov/faq)

---

## Project Structure

```
medical-report-generation/
├── data/
│   ├── download_data.py       # Script to download the IU X-Ray dataset
│   └── process_data.py        # Data preprocessing script
├── models/
│   ├── encoder.py             # DenseNet Encoder implementation
│   ├── decoder.py             # Transformer Decoder implementation
│   └── model.py               # Full model implementation
├── utils/
│   ├── dataset.py             # Dataset class for loading images and reports
│   ├── vocab.py               # Vocabulary building and tokenization
│   └── metrics.py             # Evaluation metrics including BLEU
├── train.py                   # Training script
├── inference.py               # Inference script for generating reports
├── evaluate.py                # Evaluation script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## Installation

1. Clone the repository:
```bash
git clone 
cd medical-report-generation
```

2. Create and activate a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage

### Data Preparation

1. Download the IU X-Ray dataset:
```bash
python data/download_data.py --output_dir ./data/IU_X-Ray_dataset
```

2. Process the dataset:
```bash
python data/process_data.py --data_dir ./data/IU_X-Ray_dataset --output_dir ./data/processed
```

### Training

Train the model:
```bash
python train.py --data_dir ./data/processed --save_dir ./checkpoints --epochs 30 --batch_size 16 --lr 5e-4
```

Additional training options:
- `--resume`: Path to checkpoint to resume training from
- `--freeze_encoder`: Freeze encoder parameters during training
- `--augment`: Apply data augmentation
- `--mixed_precision`: Use mixed precision training

### Inference

Generate reports from chest X-ray images:
```bash
python inference.py --model_path ./checkpoints/model_epoch8_bleu0.1216.pth --image_path ./test_images/sample.png
```

Visualize results on the validation set:
```bash
python inference.py --model_path ./checkpoints/model_epoch8_bleu0.1216.pth --val_samples 5
```

---

## Results

The model achieved a BLEU score of **0.1216** on the validation set, which is competitive with recent research in medical report generation. The model performs well on identifying normal findings but shows limitations in detecting subtle abnormalities.

### Performance Metrics

- **BLEU-1**: 0.3215  
- **BLEU-2**: 0.2148  
- **BLEU-3**: 0.1512  
- **BLEU-4**: 0.1216

### Example Output

**Input**: Chest X-ray image  
**Generated Report**:  
> "none xxxx year old female with xxxx lungs are clear bilaterally specifically no evidence of focal consolidation pneumothorax or pleural effusion cardiomediastinal silhouette is unremarkable no acute bony abnormalities identified"

---

## Future Improvements

- Pre-training on larger datasets like MIMIC-CXR
- Implementing Vision Transformers for image encoding
- Incorporating medical ontologies and knowledge graphs
- Developing clinically relevant metrics beyond BLEU
- Combining frontal and lateral views for multi-view integration
- Implementing attention visualization for explainability

---

## References

1. [Indiana University Chest X-Ray Dataset](https://openi.nlm.nih.gov/faq)  
2. Huang, G., Liu, Z., et al. (2017). *Densely connected convolutional networks*. CVPR 2017.  
3. Vaswani, A., Shazeer, N., et al. (2017). *Attention is all you need*. NeurIPS 2017.  
4. Chen, Z., Song, Y., et al. (2020). *Generating radiology reports via memory-driven transformer*. EMNLP 2020.
