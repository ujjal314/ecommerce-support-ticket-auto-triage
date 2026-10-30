# E-Commerce Support Ticket Auto-Triage System

An ML-powered support ticket classification and routing system that automatically categorizes incoming e-commerce customer queries, assigns them to operational queues, and escalates uncertain or sensitive cases for human review.

## 1. Project Overview

E-commerce support teams receive large volumes of customer queries related to refunds, cancellations, payments, delivery, account issues, and product availability.

This project builds an automated triage system using machine learning and rule-based escalation to:

- Classify customer support tickets into 11 operational categories.
- Assign tickets to the appropriate support queue.
- Use prediction confidence to identify uncertain cases.
- Escalate sensitive or high-risk tickets for human review.
- Provide an API and interactive dashboard for ticket processing.
- Capture reviewer corrections for future model improvement.

**Current implementation:** A classical NLP-based MVP using word-level and character-level TF-IDF features with Logistic Regression.

## 2. Architecture

```text
              Customer Support Ticket
                         |
                         v
                 Text Preprocessing
                         |
                         v
              TF-IDF Feature Extraction
                /                \
          Word TF-IDF       Character TF-IDF
                \                /
                 Feature Union
                         |
                         v
                Logistic Regression
                         |
                         v
                Predicted Category
                         |
                         v
              Confidence Evaluation
                         |
                 +-------+-------+
                 |               |
           High Confidence   Low Confidence
                 |               |
                 v               v
          Assigned Queue    Human Review
                 |
                 v
           API / Dashboard

       Rule-based safety escalation
       can independently trigger review.
```

## 3. Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP:** Word and character-level TF-IDF
- **Classifier:** Logistic Regression
- **Data Processing:** Pandas, NumPy
- **API:** FastAPI
- **Interactive Dashboard:** Streamlit
- **Testing:** Pytest
- **Model Persistence:** Joblib

## 4. Classification Categories

The system supports 11 routing categories:

1. refunds_cancellations
2. damaged_missing_items
3. shipping_logistics
4. account_technical
5. payments_billing
6. product_catalog_availability
7. order_management
8. store_information
9. customer_feedback
10. sales_promotions
11. human_triage_unknown

## 5. Machine Learning Approach

Two models were developed and evaluated.

### Baseline Model

- Word-level TF-IDF.
- Unigrams and bigrams.
- Logistic Regression.
- Balanced class weights.

### Robust Model

- Word-level TF-IDF features.
- Character-level TF-IDF features.
- Combined using Scikit-learn FeatureUnion.
- Logistic Regression with balanced class weights.

Character-level features help capture spelling variations, partial words, and other text variations.

### Test Set Results

Evaluation was performed on a held-out test set containing 6,702 examples.

| Metric   | Word-only baseline | Word + character model |
| -------- | -----------------: | ---------------------: |
| Accuracy |             99.42% |                 99.94% |
| Macro F1 |             0.9952 |                 0.9996 |

These results represent performance on the current random held-out test split. They should not be interpreted as guaranteed real-world production performance.

## 6. Intelligent Routing and Human Escalation

The routing service combines model predictions with additional decision logic.

Human review is triggered when:

- Prediction confidence falls below the configurable threshold.
- The model predicts the human-review category.
- Rule-based checks detect selected sensitive or high-risk situations.

Examples include suspicious transactions, potential account compromise, and certain ambiguous order-related requests.

The model prediction remains available for auditability, even when the ticket is routed to human review.

## 7. Application Features

### FastAPI

- `GET /health` — service health check.
- `POST /triage` — classify and route a support ticket.
- Input validation using Pydantic.
- Configurable confidence threshold.
- Appropriate validation and model-availability error responses.

### Streamlit Dashboard

- Interactive ticket submission.
- Configurable confidence threshold.
- Predicted category and confidence display.
- Assigned queue and human-review status.
- Reviewer correction and feedback capture in CSV format.

## 8. Project Structure

```text
ecommerce-support-ticket-auto-triage/
│
├── artifacts/
│   ├── robust_baseline_model.joblib
│   ├── baseline_metrics.json
│   ├── robust_baseline_test_metrics.json
│   └── robust_baseline_validation_metrics.json
│
├── data/
│   └── challenge/
│       └── challenge_tickets.csv
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── dashboard/
│   │   └── app.py
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── label_mapping.py
│   │   └── inspect_dataset.py
│   ├── explainability/
│   │   └── explain.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── robust_baseline.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── evaluate_robust.py
│   │   ├── evaluate_challenge.py
│   │   ├── error_analysis.py
│   │   └── robustness_evaluation.py
│   └── services/
│       └── triage.py
│
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Setup and Installation

Clone the repository:

```bash
git clone https://github.com/ujjal314/ecommerce-support-ticket-auto-triage.git
cd ecommerce-support-ticket-auto-triage
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The trained model is included in `artifacts/robust_baseline_model.joblib`, so model training is not required to run inference.

## 10. Running the Application

### Start the FastAPI server

```bash
uvicorn src.api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### Start the Streamlit dashboard

Open another terminal:

```bash
$env:PYTHONPATH = (Get-Location).Path
streamlit run src/dashboard/app.py
```

## 11. Training and Evaluation

The raw Bitext dataset is not included in the repository.

Place the dataset at:

```text
data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv
```

Generate the processed dataset and splits:

```bash
python -m src.data.preprocess
```

Train the final model:

```bash
python -m src.models.train
```

Evaluate on the test split:

```bash
python -m src.models.evaluate_robust
```

## 12. Testing

Run the automated test suite:

```bash
python -m pytest -v
```

## 13. Limitations

- The model is trained on a public retail e-commerce conversational dataset, not live company support tickets.
- The current evaluation uses a random stratified split; similar ticket templates may occur across splits.
- Confidence scores are model probabilities and are not calibrated estimates of real-world correctness.
- Rule-based escalation covers selected scenarios and is not a complete fraud or security detection system.
- Synthetic order value and customer tier fields are generated for demonstration and are not used by the current text classifier.
- Reviewer feedback is stored locally in CSV format; a complete active-learning pipeline is not implemented.
- The current API is synchronous and does not use a distributed task queue.

## 14. Future Improvements

- Metadata-aware classification and feature fusion.
- Confidence calibration and improved out-of-distribution detection.
- More comprehensive sensitive-case detection.
- Explainable predictions using LIME or SHAP.
- Active learning using reviewer corrections.
- Persistent feedback storage using a database.
- Asynchronous processing with Celery and Redis.
- Containerization and cloud deployment.
- Evaluation on real-world and independently collected support tickets.

---

**Project Status:** Functional ML-based support ticket auto-triage MVP.
