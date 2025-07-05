# Paddy Disease Classification using Deep Learning

## Overview
Multi-model deep learning system for agricultural disease classification using state-of-the-art CNN and Vision Transformer architectures.

## Project Structure
.
├── main.ipynb                    # Main implementation notebook


├── mydatasets.py                 # Custom dataset class


├── paddy_disease_train.csv       # Training dataset metadata


├── rice_leaf_disease_images.csv  # Rice disease dataset metadata


├── paddy_train_images/           # Training images


├── Rice Leaf Disease Images/     # Rice disease images


├── test_images/                  # Test dataset


├── transformer_model.pth         # Trained model weights


├── Project_Report.pdf            # Comprehensive project report


├── Test_Labels_using_CNN.csv     # CNN model predictions


├── Test_Labels_using_Transformer.csv # Transformer model predictions


└── README.md                     # This file

## Dataset
- **Paddy Doctor Dataset**: 10,407 labeled images across 10 classes
- **Rice Leaf Disease Dataset**: 5,932 images with 4 disease types
- **Total**: 16,339+ images for comprehensive training

## Models Implemented
- **CNN**: EfficientNet-B7 with transfer learning
- **Vision Transformer**: ViT-Base-Patch16-224

## Key Features
- Object-oriented training pipeline
- Advanced data augmentation techniques
- Hyperparameter optimization with OneCycleLR
- Comprehensive evaluation metrics
- Production-ready model deployment

## Results
- Achieved competitive accuracy on validation datasets
- Generated predictions for 3,469 test images
- Scalable architecture for production deployment

## Metrics & Analysis
- See Project_Report.pdf for:
- Comparative accuracy, precision, recall graphs
- ROC curves & confusion matrices
- Hyperparameter tuning rationale
- Final model recommendations

## Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.6.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- Pillow >= 8.3.0
- tqdm >= 4.62.0

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Open `main.ipynb` in Jupyter Notebook
3. Run all cells to train models and generate predictions


## Author
Bindela Jaswanth Kumar Reddy

```markdown
# Paddy & Rice Leaf Disease Classification

This repository implements state‑of‑the‑art image classification models (CNN and Vision Transformer) to detect diseases in paddy and rice leaves.

## 📂 Project Structure

```

paddy-rice-disease-classification/
├── main.ipynb                    # Training & evaluation notebook
├── mydatasets.py                 # Custom Dataset class
├── requirements.txt              # Python dependencies
├── transformer\_model.pth         # Saved ViT model weights
├── Project\_Report.pdf            # Detailed report & analysis
├── paddy\_disease\_train.csv       # Paddy dataset labels
├── rice\_leaf\_disease\_images.csv  # Rice leaf dataset labels
├── paddy\_train\_images/           # (ignored) image data
├── Rice Leaf Disease Images/     # (ignored) image data
├── test\_images/                  # (ignored) test images
└── README.md                     # This file

````

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/paddy-rice-disease-classification.git
cd paddy-rice-disease-classification
````

### 2. Create a virtual environment & install

```bash
python3 -m venv venv
source venv/bin/activate         # on Mac/Linux
# venv\Scripts\activate.bat      # on Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download datasets

#### Kaggle (Paddy Doctor)

1. Install Kaggle CLI:

   ```bash
   pip install kaggle
   ```
2. Place your `kaggle.json` in `~/.kaggle/`.
3. Run:

   ```bash
   kaggle competitions download -c paddy-disease-classification
   unzip paddy-disease-classification.zip -d paddy_train_images
   ```

#### Mendeley Data (Rice Leaf)

```bash
wget -O rice_leaf.zip "https://data.mendeley.com/public-files/datasets/fwcj7stb8r/1/files/..."
unzip rice_leaf.zip -d "Rice Leaf Disease Images"
```

### 4. Run training & evaluation

1. Open `main.ipynb` in Jupyter or VS Code.
2. Follow the notebook cells to:

   * Load and preprocess data
   * Train EfficientNet‑B7 (CNN) and ViT Base (Transformer) models
   * Plot loss & accuracy curves

### 5. Generate predictions

At the end of `main.ipynb`, run the inference cells on `test_images/` to produce a CSV:

```bash
# Example output file:
Test_Labels_using_Transformer.csv
```

## 📊 Metrics & Analysis

See `Project_Report.pdf` for:

* Accuracy, precision, recall comparisons
* ROC curves & confusion matrices
* Hyperparameter tuning rationale
* Final model recommendations

## 📝 Assumptions & Design

* **OOP design**: Modular `Trainer` class
* **Scalability**: DataLoaders with multiprocessing, adjustable batch size
* **Fault tolerance**: Early stopping, LR scheduler, model checkpointing

## 🤝 Contributing

1. Fork this repository
2. Create a branch:

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes
4. Open a Pull Request

> **Note:** Don’t commit large image folders—use the download scripts above or Git LFS.

---

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

```
```


