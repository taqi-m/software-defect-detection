# Software Defect Type Prediction

Multi-label classification system that predicts software defect types from bug report text.

## Features

- **Input**: Bug report description (text)
- **Output**: 7 defect types (Blocker, Regression, Bug, Documentation, Enhancement, Task, Dependency Upgrade)
- **Models**: Logistic Regression, SVM, Perceptron, Deep Neural Network
- **Metrics**: Hamming Loss, Micro-F1, Macro-F1, Precision@K

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
Run `AI_Assignment_3_Task_2.ipynb` to train and export models.

### 3. Run App
```bash
streamlit run app.py
```

## Files

- `app.py` - Streamlit application
- `AI_Assignment_3_Task_2.ipynb` - Training notebook
- `*.pkl` - Trained sklearn models
- `dnn_model.h5` - Deep learning model
- `tfidf_vectorizer.pkl` - Text vectorizer (required)
- `dataset.csv` - Training data

## Usage

1. Select a model from sidebar
2. Enter bug report text or upload CSV
3. Click "Predict Defect Types"
4. View results and download predictions

## Example

**Input**: "Critical bug in authentication causing system crash"

**Output**: Blocker ✓, Bug ✓
