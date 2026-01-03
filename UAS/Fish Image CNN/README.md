# 🐟 Fish Image Classification using CNN

## 📌 Project Overview
An end-to-end deep learning project for classifying fish species using Convolutional Neural Networks (CNN). This project implements both custom CNN architectures and transfer learning approaches to achieve high accuracy in multi-class image classification.

## 👨‍🎓 Student Information
- **Name**: [Your Name]
- **Class**: [Your Class]
- **NIM**: [Your Student ID]
- **Institution**: Telkom University

## 📊 Dataset Information
The Fish Image Dataset consists of images of various fish species organized in a structured directory format:

### Dataset Structure
```
Fish Image CNN/
├── train/          # Training images
│   ├── species_1/
│   ├── species_2/
│   └── ...
├── val/            # Validation images
│   ├── species_1/
│   ├── species_2/
│   └── ...
└── test/           # Test images
    ├── species_1/
    ├── species_2/
    └── ...
```

### Image Specifications
- **Image Size**: 224x224 pixels (standardized)
- **Color Space**: RGB (3 channels)
- **Format**: JPEG/PNG
- **Classes**: Multiple fish species (varies by dataset)

## 🧠 Models Implemented

### 1. **Custom CNN Architecture**
A deep convolutional neural network built from scratch with:
- **4 Convolutional Blocks** with increasing filters (32 → 64 → 128 → 256)
- **Batch Normalization** after each convolutional layer for stable training
- **MaxPooling** layers for spatial dimension reduction
- **Dropout** layers (25% after conv blocks, 50% in dense layers) for regularization
- **Dense Layers**: 512 → 256 → num_classes
- **Total Parameters**: ~5-10 million (varies with number of classes)

**Architecture Highlights**:
```python
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → Dropout(0.5) → Softmax
```

### 2. **VGG16 (Transfer Learning)**
Pre-trained on ImageNet with custom top layers:
- **Base Model**: VGG16 (frozen weights)
- **Feature Extractor**: 16 weight layers with 3x3 convolutions
- **Custom Top**: GlobalAveragePooling → Dense(512) → Dense(256) → Softmax
- **Parameters**: ~15 million (14.7M frozen, ~2M trainable)
- **Advantage**: Excellent feature extraction from deep architecture

### 3. **ResNet50 (Transfer Learning)**
Residual Network with skip connections:
- **Base Model**: ResNet50 (frozen weights)
- **Innovation**: Residual blocks that solve vanishing gradient problem
- **Custom Top**: GlobalAveragePooling → Dense(512) → Dense(256) → Softmax
- **Parameters**: ~25 million (23.5M frozen, ~2M trainable)
- **Advantage**: Very deep network (50 layers) without degradation

### 4. **MobileNetV2 (Transfer Learning)**
Efficient architecture for deployment:
- **Base Model**: MobileNetV2 (frozen weights)
- **Innovation**: Depthwise separable convolutions (lightweight)
- **Custom Top**: GlobalAveragePooling → Dense(512) → Dense(256) → Softmax
- **Parameters**: ~3.5 million (2.2M frozen, ~1.3M trainable)
- **Advantage**: Fast inference, small model size, mobile-friendly

## 🔧 Technical Implementation

### Data Preprocessing
```python
# Training augmentation
- Rescaling (1/255)
- Rotation (±20°)
- Width/Height shift (20%)
- Shear transformation (20%)
- Zoom range (20%)
- Horizontal flip
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Top-3 Accuracy
- **Batch Size**: 32
- **Image Size**: 224×224×3

### Callbacks
1. **EarlyStopping**: Stops training if validation loss doesn't improve for 10 epochs
2. **ReduceLROnPlateau**: Reduces learning rate by 0.5 when validation loss plateaus
3. **ModelCheckpoint**: Saves best model based on validation accuracy

## 📈 Evaluation Metrics

### Classification Metrics
1. **Accuracy**: Overall correct predictions / total predictions
2. **Top-3 Accuracy**: True label in top 3 predictions
3. **Precision**: TP / (TP + FP) for each class
4. **Recall**: TP / (TP + FN) for each class
5. **F1-Score**: Harmonic mean of precision and recall

### Visualization
- Training/Validation accuracy and loss curves
- Confusion matrices for all models
- Sample predictions with confidence scores
- Feature map visualization from convolutional layers
- Model performance comparison bar chart

## 🎯 Results Summary

### Model Performance Comparison
| Model | Test Accuracy | Parameters | Training Time | Strengths |
|-------|--------------|------------|---------------|-----------|
| Custom CNN | [Fill after running] | ~5-10M | Moderate | Custom design, interpretable |
| VGG16 | [Fill after running] | ~15M | Fast (frozen base) | Strong features, proven |
| ResNet50 | [Fill after running] | ~25M | Fast (frozen base) | Very deep, skip connections |
| MobileNetV2 | [Fill after running] | ~3.5M | Very fast | Lightweight, efficient |

### Key Findings
- **Best Model**: [Will be determined after training]
- **Transfer Learning** generally outperforms custom CNN due to pre-trained ImageNet features
- **MobileNetV2** offers best speed/accuracy tradeoff for deployment
- **ResNet50** typically achieves highest accuracy due to depth
- **Data Augmentation** significantly improves generalization

## 📁 Project Files
```
Fish Image CNN/
├── Fish_Image_Classification_CNN.ipynb  # Main notebook
├── README.md                            # This file
├── best_custom_cnn.keras               # Saved best model
├── train/                              # Training data
├── val/                                # Validation data
└── test/                               # Test data
```

## 🚀 How to Run

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow opencv-python scikit-learn
```

### Execution Steps
1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook Fish_Image_Classification_CNN.ipynb
   ```

2. **Run All Cells**: Execute cells sequentially from top to bottom

3. **Training Process**:
   - Data loading and exploration
   - Custom CNN training (~30 epochs)
   - VGG16 training (~20 epochs)
   - ResNet50 training (~20 epochs)
   - MobileNetV2 training (~20 epochs)
   - Evaluation and comparison

4. **Expected Runtime**: 2-4 hours (depends on GPU availability and dataset size)

### GPU Acceleration
- **Recommended**: Use GPU for faster training
- **Check GPU**: First cell verifies CUDA availability
- **Without GPU**: Training will take significantly longer

## 📊 Detailed Analysis

### Custom CNN Learning Process
The custom CNN learns hierarchical features:
1. **Early Layers**: Edges, colors, basic textures
2. **Middle Layers**: Fish parts (fins, scales, patterns)
3. **Deep Layers**: High-level species-specific features

### Transfer Learning Advantages
- **Pre-trained Features**: ImageNet knowledge transfers to fish classification
- **Faster Convergence**: Less training time needed
- **Better Performance**: Especially with limited data
- **Regularization**: Pre-trained weights act as strong priors

### Feature Map Visualization
The notebook includes visualization of:
- Convolutional filters at different depths
- Learned feature representations
- What the network "sees" at each layer

## 🔍 Confusion Matrix Interpretation
- **Diagonal Values**: Correct classifications (higher is better)
- **Off-Diagonal**: Misclassifications between species
- **Common Confusions**: Identify visually similar species

## 💡 Key Insights

### What Works Well
✅ Data augmentation prevents overfitting  
✅ Batch normalization accelerates training  
✅ Transfer learning leverages ImageNet knowledge  
✅ Dropout regularization improves generalization  
✅ Learning rate scheduling optimizes convergence  

### Potential Improvements
📌 Fine-tuning transfer learning models (unfreeze last layers)  
📌 Ensemble methods combining multiple models  
📌 Advanced augmentations (CutMix, MixUp)  
📌 Attention mechanisms for focus on important regions  
📌 Larger image resolution for more detail  

## 🎓 Learning Outcomes
This project demonstrates:
- End-to-end CNN pipeline for image classification
- Custom CNN architecture design principles
- Transfer learning implementation and benefits
- Data augmentation strategies
- Model evaluation and comparison
- Visualization of deep learning models
- Best practices in computer vision

## 📚 References
- **VGG16**: Simonyan & Zisserman (2014) - Very Deep Convolutional Networks
- **ResNet50**: He et al. (2016) - Deep Residual Learning for Image Recognition
- **MobileNetV2**: Sandler et al. (2018) - Inverted Residuals and Linear Bottlenecks
- **ImageNet**: Deng et al. (2009) - Large Scale Visual Recognition Challenge

## 📝 Notes
- All models use categorical crossentropy for multi-class classification
- Early stopping prevents overfitting
- Model checkpointing saves best performing weights
- Top-3 accuracy useful when classes are visually similar
- Feature maps reveal what CNNs learn at different depths

## 🏆 Conclusion
This comprehensive CNN project covers the full spectrum of image classification:
- ✅ **Data handling** with proper train/val/test splits
- ✅ **Multiple architectures** for thorough comparison
- ✅ **Transfer learning** for state-of-the-art results
- ✅ **Rigorous evaluation** with multiple metrics
- ✅ **Rich visualizations** for interpretability

The combination of custom and transfer learning approaches provides both educational value and practical performance, making this a robust solution for fish species classification.

---

**Project Status**: ✅ Complete  
**Last Updated**: [Current Date]  
**Course**: Machine Learning - Telkom University
