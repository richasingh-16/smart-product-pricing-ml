# 🧠 Smart Product Pricing 

The goal is to develop a model that predicts optimal e-commerce product prices by analyzing both **textual descriptions** and **product images**.  
The model combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** techniques for multimodal learning.

---

## 📌 Problem Statement

In e-commerce, determining the optimal price point for products is crucial for profitability and customer satisfaction.  
The challenge requires building a model that predicts the price of a product using only the provided dataset — without any external data or price lookups.

Each product has:
- 🧾 `catalog_content` → title, description, and item quantity text  
- 🖼️ `image_link` → public product image URL  
- 💲 `price` → the target variable (for training only)

---

## 📊 Dataset Details

| File | Description |
|------|--------------|
| `train.csv` | 75,000 training products (with prices) |
| `test.csv` | 75,000 test products (without prices) |
| `sample_test.csv` | Sample test input |
| `sample_test_out.csv` | Example output format |
| `utils.py` | Helper for downloading images |
| `extract_text_features.py` | TF-IDF + SVD text embedding |
| `extract_image_features.py` | CNN (ResNet18) image embedding |
| `train_model.py` | Combines text + image features, trains LightGBM model |

---

## ⚙️ Pipeline Overview

1. **Text Feature Extraction (NLP)**  
   - TF-IDF vectorization on `catalog_content`  
   - Dimensionality reduction using Truncated SVD → 200 components  

2. **Image Feature Extraction (CV)**  
   - Pretrained ResNet18 (ImageNet weights)  
   - Extracts 2048-dimensional embeddings  
   - Features saved as `.pkl` for efficient re-use  

3. **Model Training (ML)**  
   - LightGBM Regressor with early stopping  
   - Trained on combined text + image feature vectors  

4. **Evaluation Metric**  
   - SMAPE (Symmetric Mean Absolute Percentage Error)

---

## 🧮 Evaluation Metric

The performance is measured using **SMAPE**:

\[
SMAPE = \frac{1}{n} \sum \frac{|pred - actual|}{(|pred| + |actual|)/2} \times 100
\]

Example:  
If actual price = \$100 and predicted = \$120,  
→ SMAPE = 18.18%

---

## 🧰 Tech Stack

- **Language:** Python 3.10  
- **Libraries:** pandas, numpy, sklearn, lightgbm, torch, torchvision, joblib, tqdm  
- **Frameworks:** PyTorch, LightGBM  
- **Environment:** Windows 11, AMD Ryzen 7 CPU, Radeon Integrated GPU

---
