## Live Demo
👉 https://indian-ipo-intelligence-8adgxc95zuzyfid5j6kw34.streamlit.app/
# Indian IPO Intelligence Platform

An end-to-end data science project analyzing 259 Indian IPOs (2010–2021)
with ML prediction, deep learning, and interactive dashboards.

## Project Highlights
- Real-world messy data cleaning pipeline
- Pandera data validation with schema checks
- 10 interactive EDA visualizations using Plotly
- XGBoost model with SHAP explainability
- LSTM deep learning model with PyTorch
- MLflow experiment tracking
- 4-page Streamlit web application
- Power BI executive dashboard

## Tech Stack
Python · Pandas · XGBoost · PyTorch · SHAP · 
MLflow · FastAPI · Streamlit · Power BI · Pandera

## Project Structure
notebooks/   — 6 Jupyter notebooks (scraping → cleaning → EDA → validation → ML → DL)
src/         — Saved model files
app/         — Streamlit web application
api/         — FastAPI backend
data/        — Raw, cleaned, and validated datasets
powerbi/     — Power BI dashboard (.pbix)

## How to Run
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## Key Findings
- Total subscription is the strongest predictor of listing gains
- July is historically the best month for IPO listings (avg +29.1%)
- August is the worst month (avg -23.2%)
- Retail investor (RII) participation is the 2nd most important signal
- Market mood (Nifty 30d return) has moderate impact on IPO performance