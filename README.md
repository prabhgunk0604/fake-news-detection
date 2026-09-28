# 🛡️ TruthCheck: Fake News Detection

A machine learning based Fake News Detection web app that classifies news articles as **Real** or **Fake** using NLP (TF-IDF + Logistic Regression), with a confidence score.

## Features
- Clean, modern Streamlit web interface
- Text preprocessing (removes URLs, punctuation, numbers, source tags)
- TF-IDF vectorization
- Real / Fake prediction with confidence score
- One-click example buttons to test quickly

## Project Structure
- `make_sample_data.py` : generates a small sample dataset
- `train_model.py` : trains the model and saves `model.pkl` and `vectorizer.pkl`
- `app.py` : Streamlit web app
- `requirements.txt` : required libraries

## Dataset
Fake and Real News Dataset (Kaggle): `True.csv` and `Fake.csv`.
Download it from Kaggle and place both files in the project folder.
(Alternatively, run `make_sample_data.py` to create a small sample dataset.)

## Installation
git clone https://github.com/prabhgunk0604/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt

## How to Run
1. Train the model:
   python train_model.py
2. Start the app:
   streamlit run app.py

## Tech Stack
Python, Scikit-learn, Pandas, Streamlit

## Author
Prabhgun

## License
MIT License
