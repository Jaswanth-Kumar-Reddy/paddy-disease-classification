# Paddy Disease Classification using Deep Learning

## Overview
Multi-model deep learning system for agricultural disease classification using state-of-the-art CNN and Vision Transformer architectures.

This repository implements state‑of‑the‑art image classification models (CNN and Vision Transformer) to detect diseases in paddy and rice leaves.

## 📂 Project Structure



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

- **Paddy Doctor Dataset**: 10,407 labeled images across 10 classes
- **Rice Leaf Disease Dataset**: 5,932 images with 4 disease types
- **Total**: 16,339+ images for comprehensive training

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

## Models Implemented
- **CNN**: EfficientNet-B7 with transfer learning
- **Vision Transformer**: ViT-Base-Patch16-224


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

## Author
Bindela Jaswanth Kumar Reddy


```
```


