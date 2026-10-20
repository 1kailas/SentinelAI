# SentinelAI 🛡️
### Machine Learning-Based Insider Threat Detection

SentinelAI is a machine-learning-based insider-threat detection prototype designed to identify potentially suspicious employee behaviour by analysing activity data. It combines anomaly detection, sequential behaviour analysis, risk assessment, and a web-based dashboard to help administrators investigate unusual activity.

The project explores how machine learning can support security monitoring by highlighting patterns that may require further investigation.

## 🎯 Project Objectives

- Analyse employee activity data to identify unusual behaviour.
- Explore machine learning techniques for insider-threat detection.
- Use sequential analysis to study behavioural patterns over time.
- Assess suspicious activity through risk analysis.
- Present relevant information through a web-based dashboard.
- Evaluate model performance using machine learning metrics.

## ✨ Key Features

- **Anomaly Detection:** Uses Isolation Forest to identify potentially unusual activity.
- **Sequential Behaviour Analysis:** Includes an LSTM model for analysing sequences of activity.
- **Risk Assessment:** Provides a risk-analysis component to support the review of suspicious behaviour.
- **Web Interface:** Provides a frontend for interacting with the application.
- **Backend API:** Uses FastAPI to handle backend operations.
- **Local Database:** Uses SQLite for local data storage.
- **Model Evaluation:** Supports evaluation using metrics such as precision, recall, and ROC-AUC.

*Note: The Isolation Forest and LSTM components are separate model components; their presence does not necessarily mean they operate as a single combined model.*

## 🧠 How It Works

The project follows this general workflow:

1. **Activity Data:** Uses the CERT insider-threat dataset and, where applicable, simulated activity.
2. **Preprocessing:** Organises and prepares activity data for analysis.
3. **Machine Learning:** Applies anomaly detection and sequential analysis.
4. **Risk Analysis:** Processes relevant outputs to support suspicious-activity review.
5. **Dashboard:** Presents information through a web interface for further investigation.

Historical dataset analysis and simulated application activity should be treated as separate data sources unless they are explicitly connected in the implementation.

## 🛠️ Tech Stack

| Technology            | Purpose                                |
|-----------------------|----------------------------------------|
| Python                | Backend and machine learning           |
| FastAPI               | Backend API                            |
| HTML, CSS, JavaScript | Web interface                          |
| SQLite                | Local database                         |
| Pandas, NumPy         | Data processing                        |
| Scikit-learn          | Machine learning and anomaly detection |
| TensorFlow / Keras    | LSTM model                             |
| Joblib                | Saving and loading model artifacts     |

## 📁 Project Structure

```text
SentinelAI/
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── risk_engine.py
│   ├── device_monitor.py
│   ├── lstm_model.py
│   └── lstm_inference.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   ├── device.csv
│   ├── file.csv
│   ├── insiders.csv
│   └── logon.csv
├── outputs/
│   ├── sentinelai.db
│   ├── sentinelai_model.joblib
│   ├── sentinelai_lstm.keras
│   ├── sentinelai_lstm_metrics.json
│   └── sentinelai_lstm_scaler.npz
├── requirements.txt
├── README.md
└── SentinalAl.ipynb
```

*The structure above reflects the project's current components. Some generated files or datasets may be excluded from the public repository depending on size, licensing, and security considerations.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/GauriNair77/SentinelAI.git
cd SentinelAI
```

If the repository URL or capitalization differs, use the clone URL shown on your GitHub repository page.

### 2. Create a Virtual Environment

**Windows — PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If the project uses Supabase or other external services, create a local `.env` file in the project root and configure the required variables using the names expected by the application.

Example:

```env
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_key
```

Use the appropriate key type and variable names for your configuration. Never commit secret keys, database credentials, or other sensitive information to GitHub.

### 5. Run the Application

From the project root, run:

```bash
python -m uvicorn backend.main:app --reload
```

Open the application at:

- **Application:** http://127.0.0.1:8000/app/
- **API documentation:** http://127.0.0.1:8000/docs

These URLs apply when the server starts successfully with the default local configuration.

### 6. Run Unit Tests

Execute the automated test suite from the project root:

```bash
python -m unittest discover tests
```

## 📊 Model Evaluation

SentinelAI explores model evaluation using metrics such as:

- **Precision:** How many flagged cases are actually positive cases.
- **Recall:** How many actual positive cases are detected.
- **ROC-AUC:** How well the model distinguishes between positive and negative cases across classification thresholds.

Security applications need to consider both missed threats and false positives. A model's accuracy alone is not sufficient to establish that it is reliable in a real-world environment.

Any reported results should be interpreted in the context of the dataset, preprocessing, class imbalance, evaluation method, and test-set design. Historical dataset results do not automatically demonstrate real-world detection performance.

## 🔐 Security and Privacy

- Use only datasets and activity logs that you are authorised to access.
- Do not upload API keys, `.env` files, database credentials, or private user data.
- Review `.gitignore` before publishing the repository.
- Treat alerts as indicators for investigation, not proof of malicious intent.
- Apply appropriate access controls before connecting the project to any real organisation's systems.

## ⚠️ Limitations

SentinelAI is a student-developed prototype and is not a production-ready security platform.

- Detection performance depends on data quality and model evaluation.
- False positives and missed detections are possible.
- Historical datasets may not represent current enterprise environments.
- Simulated activity does not establish effectiveness against real attacks.
- External database connectivity does not automatically mean that data migration or synchronisation is implemented.

## 🔮 Future Improvements

- Integrate authorised enterprise audit logs through documented APIs.
- Improve alert explanations and risk-score interpretability.
- Compare models using consistent datasets and evaluation methods.
- Test performance on unseen data and different threat scenarios.
- Add appropriate authentication, authorisation, and audit logging.
- Complete and validate cloud database integration.
- Improve dashboard usability and investigation workflows.

## 🎓 Project Scope

SentinelAI is an educational project exploring the application of machine learning to insider-threat detection. Its contribution lies in implementing and integrating data processing, machine learning components, risk analysis, and a web interface into a working prototype.

It is intended to support learning and experimentation, rather than replace established enterprise security products.

## 👩‍💻 Author

**Gauri Nair**

B.Tech Computer Science and Engineering

## 📄 Disclaimer

This project is intended for educational and research purposes. It does not guarantee the detection of all insider threats and should not be used as the sole basis for security decisions.
