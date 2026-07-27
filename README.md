# 🛩️ Aircraft Damage Detection with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/aircraft-damage-detection)

> An end-to-end deep learning solution for automated aircraft damage classification and image captioning using pre-trained models (VGG16 and BLIP).

![Demo](https://via.placeholder.com/1200x400/1a237e/ffffff?text=Aircraft+Damage+Detection+Demo)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Training](#-model-training)
- [Results](#-results)
- [Image Captioning](#-image-captioning-with-blip)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Citations](#-citations)

---

## 🎯 Project Overview

Aircraft damage detection is **critical** for maintaining aviation safety and extending aircraft longevity. Traditional manual inspection methods are:
- ❌ Time-consuming
- ❌ Subjective
- ❌ Prone to human error

This project addresses these challenges by implementing an **automated dual-component system**:

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Classification** | Classify aircraft damage into **dent** or **crack** | VGG16 + Custom Dense Layers |
| **Captioning** | Generate descriptive text about the damage | BLIP Transformer Model |

### 🎯 Key Objectives

- ✅ Automate aircraft damage classification with **85-90% accuracy**
- ✅ Provide **interpretable** image descriptions through natural language
- ✅ Reduce inspection time and **human error**
- ✅ Enable **scalable** damage assessment for aviation maintenance
- ✅ Demonstrate **transfer learning** with pre-trained models

---

## ✨ Key Features

| Feature | Description | Technology |
|---------|-------------|------------|
| 🔍 **Binary Classification** | Classifies images as "dent" or "crack" with high accuracy | VGG16 + Custom Dense Layers |
| 📝 **Image Captioning** | Generates descriptive captions for damage images | BLIP Transformer Model |
| 📊 **Image Summarization** | Produces detailed summaries of damage | BLIP with Custom Prompts |
| 🔄 **Data Augmentation** | Improves model generalization | Keras ImageDataGenerator |
| 📈 **Interactive Visualization** | Displays predictions with true/predicted labels | Matplotlib |
| 🧩 **Custom Keras Layer** | Seamless integration of BLIP model | TensorFlow Custom Layers |
| 🚀 **Transfer Learning** | Leverages pre-trained models for faster training | VGG16 (ImageNet) + BLIP |

---

## 🏗️ Technical Architecture

### 1. Classification Pipeline (VGG16)
|Input Image (224x224x3)
|VGC Base Model (Feature Extraction)
* Pre-trained on ImageNet
* Weight Frozen
* Include_top=False
|Flatten Layer |
|Dense Layer (512 units + RelU) |
|Dropout (0.3) |
|Dense Layer (512 units + ReLU) |
|Dropout(0.3) |
|Dense Layer (1 unit + Sigmoid) |
| Output: Binary Classification |
| 0: Dent | 1: Crack |

###2. Captioning Pipeline (BLIP)
| Input Image |
| BLIP Processor |
* Image Preprocessing(resize, normalize)
* Text Tokenization (Prompt preparation)

| BLIP Model (Conditional Generation) |
* Vision Transformer Encoder
* Language Model Decoder
* Cross-attention mechanisms

| Output Text |
|| Caption || Summary ||
|| A picture of a dent on an aircraft surface || This is a detailed photo showing a metallic dent on an aircraft fuselage with visible damage ||

---

## 📊 Dataset

### Source Information

| Attribute | Details |
|-----------|---------|
| **Original Source** | [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) |
| **License** | CC BY 4.0 |
| **Dataset Size** | Images of aircraft damage (dent & crack) |
| **Format** | JPG images |


### Data Preprocessing

python
# Image augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/Test (only rescaling)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

### Results
Classification Performance
• Test Accuracy: ~85-90% (varies by run)
• Loss: ~0.25-0.35
• Key Features:
• High accuracy in distinguishing dents from cracks
• Robust performance on unseen test images
• Quick inference time suitable for real-time applications
