# Log Anomaly Detector

A Python-based machine learning pipeline for detecting anomalies in system logs (specifically HDFS logs). This project uses big data logs, processes them into feature vectors, and trains an Isolation Forest model to identify suspicious log lines.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline
The pipeline consists of four sequential steps. You can run them individually:

**Step 1: Data Preparation**
Downloads the HDFS dataset from Hugging Face and prepares a local subset.
```bash
python src/prepare_hdfs_hf.py
```
*Output: `data/hdfs_logs.txt`, `data/hdfs_labels.csv`*

**Step 2: Feature Engineering**
Parses logs, extracts templates (unifying IDs/IPs), and computes numerical features (metrics like char count, template frequency, keyword hits).
```bash
python src/make_features.py
```
*Output: `outputs/features.csv`*

**Step 3: Train Model**
Trains an Isolation Forest model on the extracted features.
```bash
python src/train_model.py
```
*Output: `outputs/model.pkl`*

**Step 4: Detect & Evaluate**
Uses the trained model to score logs and compares predictions against ground truth labels.
```bash
python src/detect_anomalies.py
python src/evaluate.py
```
*Output: `outputs/anomalies.csv`, console precision/recall metrics*

## 📂 Project Structure

```
log-anomaly-detector/
├── data/                   # Raw logs and labels
├── outputs/                # Generated features, models, and anomaly scores
├── src/
│   ├── prepare_hdfs_hf.py  # Data loader (Hugging Face -> local)
│   ├── make_features.py    # Log parsing & feature extraction
│   ├── train_model.py      # Model training (Isolation Forest)
│   ├── detect_anomalies.py # Inference script
│   └── evaluate.py         # Performance metrics
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

## 🧠 Model & Approach

- **Templates**: Log messages are normalized (masking IPs, Block IDs, Numbers) to identify the "template" of the message.
- **Features**:
    - `template_freq`: How rare is this log pattern? (Rare = Suspicious)
    - `is_error`: Does it contain words like "failed", "error", "exception"?
    - Shape features: Word count, character count, etc.
- **Algorithm**: standard **Isolation Forest** from scikit-learn. It isolates anomalies by randomly partitioning the feature space; anomalies are "shallower" in the trees than normal points.

## 🧪 Testing

Unit tests ensure the critical regex and template logic is correct.

```bash
python -m unittest discover tests
```