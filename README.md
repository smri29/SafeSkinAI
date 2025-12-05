🛡️ SafeSkin AI v2.0

The Premium Standard for Cosmetic Safety Intelligence

SafeSkin AI is a cutting-edge computational toxicology application designed to screen cosmetic ingredients for hidden health risks. Leveraging a rigorously curated database of over 11,500 chemical structures, it utilizes advanced ensemble machine learning to predict Carcinogenicity and Reproductive Toxicity in real-time.

📸 App Interface

<p align="center">
<img src="SafeSkin%20AI%2001.png" width="45%" />
<img src="SafeSkin%20AI%2002.png" width="45%" />
</p>
<p align="center">
<img src="SafeSkin%20AI%2003.png" width="45%" />
<img src="SafeSkin%20AI%2004.png" width="45%" />
</p>

✨ v2.0 Features

💎 Premium Glassmorphism UI: A sophisticated "Deep Ocean" aesthetic featuring translucent glass cards, dynamic gradients, and smooth animations.

🧠 Advanced Analytics Dashboard:

Interactive Gauge Charts: Visualizing risk probabilities with dynamic color coding.

Chemical Property Radar: A 6-axis spider chart comparing physicochemical properties against cosmetic standards.

Molecular Structure Rendering: High-resolution 2D visualization of the analyzed ingredient.

🎲 Smart "Surprise Me" Engine: Integrated database of 50+ real-world cosmetic ingredients (Retinol, Parabens, Vitamin C, etc.) with context regarding their role and safety profile.

⚡ Dual-Endpoint Prediction:

Carcinogenicity: XGBoost classifier optimized for structural alerts.

Reproductive Toxicity: Random Forest model detecting endocrine disruption potential.

📄 Report Generation: Instantly download a text-based analysis report for any screened molecule.

📊 Project Intelligence Hub: Dynamic section displaying training metrics (ROC-AUC scores) and dataset statistics.

🚀 Installation & Setup

Follow these steps to run SafeSkin AI on your local machine.

Prerequisites

Python 3.8 or higher

VS Code (recommended)

1. Clone the Repository

git clone [https://github.com/smri29/SafeSkinAI.git](https://github.com/smri29/SafeSkinAI.git)
cd SafeSkinAI


2. Install Dependencies

pip install -r requirements.txt


3. Verify System Artifacts

Ensure the following model files are present in the root directory:

cancer_model.pkl

repro_model.pkl

scaler.pkl

app_metadata.json

model_stats.json

4. Launch the App

streamlit run app.py


The application will launch automatically in your browser at http://localhost:8501.

📂 Project Structure

safeskin-ai/
├── app.py                # Main v2.0 application source code
├── requirements.txt      # Python dependencies (Streamlit, RDKit, Plotly, etc.)
├── cancer_model.pkl      # Trained XGBoost model (Cancer Endpoint)
├── repro_model.pkl       # Trained Random Forest model (Repro Endpoint)
├── scaler.pkl            # StandardScaler object for feature normalization
├── app_metadata.json     # Feature mapping configuration
├── model_stats.json      # Validation metrics (AUC/Accuracy)
├── SafeSkin AI 01.png    # Screenshot
├── SafeSkin AI 02.png    # Screenshot
├── SafeSkin AI 03.png    # Screenshot
├── SafeSkin AI 04.png    # Screenshot
└── README.md             # Documentation


🧬 Scientific Methodology

SafeSkin AI represents a significant leap in in silico screening:

Data Curation: Aggregated 11,555 unique structures from high-confidence sources:

US EPA ToxCast (High-throughput screening data)

NIH Tox21 (Toxicology in the 21st Century)

PubChem (Structural resolution)

Feature Engineering: Extracts 2,069 molecular features per chemical, combining:

Morgan Fingerprints: 2048-bit structural vectors.

Physicochemical Descriptors: LogP, Molecular Weight, TPSA, Lipinski Violations, etc.

Imbalance Handling: Utilized ADASYN (Adaptive Synthetic Sampling) to oversample rare toxicity events, preventing model bias towards safe classes.

Validation: Models achieved rigorous validation scores (see "Project Intelligence" inside the app for live metrics).

⚠️ Disclaimer

SafeSkin AI is a predictive research tool. While it achieves high statistical accuracy based on historical data, in silico predictions should not replace standardized laboratory testing or regulatory safety assessments.

Developed by Shah Mohammad Rizvi | SafeSkin AI v2.0 | Powered by Streamlit, RDKit & Plotly