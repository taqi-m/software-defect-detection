# Software Defect Type Prediction

Multi-label classification system that predicts software defect types from bug report text.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

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

### 2. Run App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

## Project Structure

```
software_defect/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── dataset.csv                     # Training data
├── logistic_regression_model.pkl   # Trained models (generated)
├── svm_model.pkl
├── perceptron_model.pkl
├── dnn_model.h5
└── tfidf_vectorizer.pkl           # Text vectorizer (required)
```

## Usage

1. **Select Model**: Choose from sidebar (Logistic Regression, SVM, Perceptron, or DNN)
2. **Enter Text**: Type bug report or upload CSV file
3. **Predict**: Click "Predict Defect Types" button
4. **Results**: View predictions, confidence scores, and download results

## Example

**Input**: "Critical bug in authentication causing system crash"

**Output**: 
- Blocker ✓ (92%)
- Bug ✓ (87%)
- Regression ✗ (23%)

## Models & Evaluation

All models use multi-label classification with the following evaluation metrics:
- **Hamming Loss**: Fraction of incorrect labels
- **Micro-F1**: Global F1 score
- **Macro-F1**: Average F1 per label
- **Precision@3/5**: Top-k precision

## Deployment

The app is deployed at: **[YOUR_STREAMLIT_URL_HERE](YOUR_STREAMLIT_URL_HERE)**

To deploy your own:
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

## License

Educational project for AI/ML learning purposes.
