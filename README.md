# Student Name:-Amey Banarase
# Student ID:-bitsom_ba_2511968

# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Project Overview
This project builds and analyzes a feed-forward neural network to predict **customer churn** using structured tabular data. The goal is to demonstrate how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

## Dataset Source
**Dataset:** `customer_churn_nn.csv`  
**Source:** BITSoM Module 5 - Part 1 Shared Google Drive Folder  
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

## Dataset Description
| Feature | Type | Description |
|---|---|---|
| customer_id | Identifier | Unique customer ID (not used for training) |
| region | Categorical | Customer region (North/South/East/West/Central) |
| plan_type | Categorical | Subscription plan (Standard/Premium) |
| contract_type | Categorical | Contract duration type |
| payment_method | Categorical | Payment method used |
| tenure_months | Numerical | How long customer has been subscribed |
| monthly_charges_inr | Numerical | Monthly bill amount in INR |
| avg_login_days_per_month | Numerical | Average days logged in per month |
| support_tickets_last_90_days | Numerical | Number of support tickets raised |
| payment_delay_days | Numerical | Average delay in payment |
| data_usage_gb | Numerical | Data consumed per month in GB |
| satisfaction_score | Numerical | Customer satisfaction score (1-10) |
| last_complaint_days_ago | Numerical | Days since last complaint |
| discount_percent | Numerical | Discount given to customer |
| autopay_enabled | Binary | Whether autopay is on (1/0) |
| referral_count | Numerical | Number of referrals made |
| **churn** | **Target** | **1 = Churned, 0 = Retained** |

- **Total Records:** 2000
- **Total Features:** 16 input features
- **Task Type:** Binary Classification

## Repository Structure
```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    ├── evaluation_outputs.png
    ├── confusion_matrix.png
    └── target_distribution.png
```

## Tasks Completed

### Task 1: Dataset Understanding
- Loaded dataset with 2000 rows and 17 columns
- Identified 4 categorical and 12 numerical input features
- Target variable: `churn` (binary: 0 or 1)
- No missing values found in the dataset
- Explored class distribution (churn vs retained)

### Task 2: Data Preprocessing
- Dropped `customer_id` (identifier, not a feature)
- Applied **Label Encoding** for categorical columns: region, plan_type, contract_type, payment_method
- Applied **StandardScaler** for numerical feature normalization (zero mean, unit variance)
- **Train-Test Split:** 80% training (1600 samples), 20% testing (400 samples)
- Used `stratify=y` to preserve class distribution in both splits

### Task 3: Neural Network Architecture
Built a Feed-Forward Neural Network using TensorFlow/Keras:

```
Input Layer     → 16 features
Hidden Layer 1  → 64 neurons, ReLU activation, Dropout(0.3)
Hidden Layer 2  → 32 neurons, ReLU activation, Dropout(0.2)
Output Layer    → 1 neuron, Sigmoid activation
```

- **Loss Function:** Binary Crossentropy (suitable for binary classification)
- **Optimizer:** Adam (lr=0.001)
- **Metric:** Accuracy

### Task 4: Training and Evaluation
- Trained for 50 epochs with batch size 32
- Evaluated using accuracy, loss, confusion matrix, and classification report

### Task 5: Hyperparameter Experimentation

| Experiment | Neurons (L1, L2) | Learning Rate | Activation | Batch Size | Train Acc (%) | Test Acc (%) |
|---|---|---|---|---|---|---|
| Baseline | 64, 32 | 0.001 | ReLU | 32 | ~85% | ~84% |
| High LR | 64, 32 | 0.01 | ReLU | 32 | ~84% | ~83% |
| Larger Network | 128, 64 | 0.001 | ReLU | 32 | ~86% | ~85% |
| Tanh Activation | 64, 32 | 0.001 | Tanh | 64 | ~84% | ~83% |

**Key Observations:**
- Higher learning rate (0.01) caused slightly unstable training
- Larger network improved accuracy marginally but increased training time
- ReLU outperformed Tanh slightly for this dataset
- Baseline model offered the best balance of simplicity and accuracy

### Task 6: Final Reflection

**Role of Weights and Biases:**
Weights control how strongly each input feature influences the output. Biases allow the model to shift activation outputs independent of inputs, giving more flexibility. Both are updated during backpropagation to minimize the loss.

**Why Activation Functions are Required:**
Without activation functions, neural networks reduce to simple linear models regardless of depth. Activations like ReLU introduce non-linearity, enabling the model to learn complex patterns in real-world data.

**Effect of Learning Rate:**
- Too high → unstable training, loss oscillates or diverges
- Too low → very slow convergence, may get stuck
- Optimal (~0.001) → smooth convergence to good accuracy

**Underfitting vs Overfitting:**
The baseline model showed balanced train and test accuracy, indicating no major underfitting or overfitting. Dropout layers (0.3 and 0.2) acted as regularization to prevent overfitting.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebook.ipynb
```

## Requirements
See `requirements.txt`

## Results
- Test Accuracy: ~84%
- Best Configuration: Larger Network (128, 64 neurons) with Adam optimizer

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
