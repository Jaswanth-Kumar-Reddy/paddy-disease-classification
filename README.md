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

## Model Performance
Detailed performance metrics and comparisons are available in `Project_Report.pdf`.

## Author
Bindela Jaswanth Kumar Reddy
