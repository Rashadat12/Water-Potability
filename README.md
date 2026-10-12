# 💧 Water Potability Prediction

A Machine Learning project for predicting water quality and potability based on physicochemical metrics. This repository contains the complete exploratory data analysis, data preprocessing pipeline, class balancing, model training, and performance benchmarking comparing Support Vector Machines (SVM), Decision Trees (DT), and K-Nearest Neighbors (KNN).

---

## 📌 Project Overview

Access to safe drinking water is essential to health and a basic human right. This project evaluates various water quality attributes and predicts whether water is safe for human consumption (`Potability = 1`) or not safe (`Potability = 0`).

### 📊 Dataset Attributes

The dataset contains water quality metrics for 3,276 water bodies across 9 water quality indicators:

| Feature | Description | Reference Safe Limit |
|---|---|---|
| **pH** | Acid-base balance of water | 6.5 – 8.5 (WHO) |
| **Hardness** | Capacity of water to precipitate soap (calcium & magnesium) | < 200 mg/L |
| **Solids (TDS)** | Total dissolved solids and mineralization | 500 – 1000 mg/L |
| **Chloramines** | Disinfection agent derived from chlorine and ammonia | <= 4 mg/L |
| **Sulfate** | Naturally occurring minerals and industrial discharge | <= 250 mg/L |
| **Conductivity** | Electrical conductivity indicating ion concentration | <= 400 μS/cm |
| **Organic Carbon** | Total Organic Carbon (TOC) from decaying natural matter | <= 2 mg/L |
| **Trihalomethanes** | Disinfection by-products in chlorinated water | <= 80 μg/L |
| **Turbidity** | Measure of light-emitting properties / suspended matter | <= 5 NTU |
| **Potability** | **Target variable**: `0` = Not Potable, `1` = Potable | Binary Label |

---

## ⚙️ Methodology & Pipeline

1. **Exploratory Data Analysis (EDA)**:
   - Correlation heatmap analysis between chemical attributes.
   - Identifying missing data distributions.
2. **Missing Value Imputation**:
   - Features `ph`, `Sulfate`, and `Trihalomethanes` imputed using frequent value strategies.
3. **Class Balancing**:
   - Addressing class imbalance through minority class upsampling to prevent majority-class bias.
4. **Feature Scaling**:
   - Normalization using `MinMaxScaler` across all continuous dimensions.
5. **Model Training & Evaluation**:
   - **Support Vector Machine (SVM)** (RBF Kernel, C=1.0)
   - **Decision Tree Classifier**
   - **K-Nearest Neighbors (KNN)** ($k=3$)
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 📂 Repository Structure

```text
├── Water_Potability.ipynb          # Interactive Jupyter Notebook with Colab EDA & plots
├── main.py                         # Standalone modular ML pipeline script
├── water_potability.csv            # Water quality dataset
├── Water_Potability_Project_Report.pdf # Detailed project research report
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Rashadat12/Water-Potability-Prediction.git
cd Water-Potability-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
```bash
python main.py
```

Or open and run `Water_Potability.ipynb` in **Jupyter Notebook**, **VS Code**, or **Google Colab**.

---

## 👥 Author

- **Rahman Md. Rashadat Abdullah**
- GitHub: [@Rashadat12](https://github.com/Rashadat12)
