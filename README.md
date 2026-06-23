# 🏏 IPL Stats Dashboard

## Project Overview
IPL Cricket Stats Dashboard built with Python, Pandas, Matplotlib, Scikit-learn and Streamlit.

## Features
- 🏠 Overview - Season wise stats
- 🔍 Data Quality - Data cleaning info
- 🏆 Team Stats - Win analysis
- 🏏 Batting Stats - Run scorers, Strike rates, Sixes
- 🎳 Bowling Stats - Wickets, Economy, Bowling SR
- 🤖 ML Prediction - Match winner prediction
- ⚔️ Head to Head - Team comparison

## Technologies Used
- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Dataset
- matches.csv - 1095 IPL matches (2007-2024)
- deliveries.csv - Ball by ball data

## How to Run
pip install pandas numpy matplotlib scikit-learn streamlit
streamlit run streamlit_app.py

## Project Structure
IPL_Project/
├── data/
│   ├── matches.csv
│   └── deliveries.csv
├── src/
│   ├── ipl_dashboard.py
│   ├── streamlit_app.py
├── models/
├── charts/
└── README.md