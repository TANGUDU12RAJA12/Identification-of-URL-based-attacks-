# URL-Based Cyber Attack Detection System

## Project Overview

The URL-Based Cyber Attack Detection System is a Machine Learning and Streamlit-based web application designed to detect malicious URL attacks using IPDR data, URL analysis, and payload-based feature extraction. The system performs both binary classification and multi-class classification to identify and categorize different types of cyber attacks.

The project focuses on improving cybersecurity by detecting harmful web traffic using advanced machine learning algorithms and interactive visualizations.

---

# Features

- Malicious URL Detection
- Binary Classification of Safe and Malicious URLs
- Multi-Class Attack Classification
- Machine Learning-Based Prediction
- Streamlit Web Application
- CSV and TXT File Upload Support
- Data Visualization and Analysis
- Feature Engineering for Better Accuracy
- Interactive Charts and Graphs
- Real-Time Prediction Interface
- User-Friendly Dashboard

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib / Pickle
- Jupyter Notebook
- Visual Studio Code

---

# Cybersecurity Tools Used

- SQLmap
- Burp Suite
- XSStrike

---

# Machine Learning Models

## Binary Classification

The system uses the **Random Forest Algorithm** to classify URLs as:

- Safe
- Malicious

---

## Multi-Class Classification

The system uses **XGBoost Algorithm** to classify different types of attacks with high accuracy.

---

# Dataset & Feature Engineering

The system uses structured datasets containing:

- URL Features
- Payload Features
- IPDR Data

Feature engineering techniques are applied to improve prediction accuracy and model performance.

---

# Evaluation Metrics

The performance of the system is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Feature Importance Analysis

The experimental results demonstrate that XGBoost performs better in multi-class attack classification compared to Random Forest.

---

# Streamlit Web Application

The project includes an interactive Streamlit web application that allows users to:

- Upload CSV or TXT files
- Analyze URL traffic
- Predict malicious attacks
- Visualize attack patterns
- View model predictions
- Generate interactive charts

---

# Project Structure

```text
URL_Attacks
│
├── app.py
├── ModelBuilding.ipynb
├── realistic_model_building.ipynb
├── models/
├── dataset/
├── requirements.txt
├── README.md
```

---

# How to Run the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/URL_Attacks.git
```

---

## Navigate to Project Folder

```bash
cd URL_Attacks
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# Experimental Results

The proposed system achieved high accuracy in detecting URL-based cyber attacks using machine learning techniques.

- Random Forest showed strong performance in binary classification.
- XGBoost achieved better accuracy in multi-class classification.
- Visualization techniques such as confusion matrices and feature importance graphs improved result interpretation.

---

# Advantages

- Detects malicious URL attacks efficiently
- Supports both binary and multi-class classification
- Interactive and user-friendly Streamlit interface
- Easy dataset upload and analysis
- Real-time prediction capability
- Extendable architecture for future improvements
- Enhances cybersecurity awareness

---

# Future Scope

Future enhancements may include:

- Deep Learning Integration
- Real-Time Network Monitoring
- PCAP File Processing
- Browser Extension Support
- Cloud Deployment
- Integration with Firewalls and IDS Systems
- Advanced Threat Detection
- Large-Scale Dataset Training

---

# Learning Outcomes

Through this project, practical experience was gained in:

- Machine Learning Model Building
- Cybersecurity Threat Detection
- Feature Engineering
- Data Visualization
- Streamlit Application Development
- Model Evaluation Techniques
- Python-Based Data Analysis
- Web-Based ML Deployment

---

# References

1. D. Sahoo, C. Liu, and S. C. H. Hoi, “Malicious URL Detection using Machine Learning: A Survey,” arXiv, 2017.
2. F. Türk and M. Kılıçaslan, “Malicious URL Detection with Advanced Machine Learning and Optimization-Supported Deep Learning Models,” Applied Sciences, 2025.
3. D. R. Patil and J. B. Patil, “Malicious URLs Detection using Machine Learning Techniques,” IEEE Conference, 2018.
4. S. Patil and H. A. Dinesha, “URL Redirection Attack Mitigation using Machine Learning Algorithm,” Indian Journal of Science and Technology, 2022.

---

# Author

Developed by Tangudu Raja
