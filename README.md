# 🎬 Movie Recommendation System

An AI-powered Movie Recommendation System built using **Machine Learning**, **Natural Language Processing (NLP)**, **FastAPI**, and **Streamlit**. The application recommends similar movies using a content-based filtering approach with **TF-IDF Vectorization** and provides movie posters, ratings, and details using the **TMDB API**.

## 🚀 Features

- 🔍 Search movies by title
- 🤖 AI-powered content-based recommendations
- 🧠 TF-IDF Vectorization for similarity matching
- 🎭 Genre-based recommendations
- 🖼️ Movie posters and details from TMDB API
- ⚡ FastAPI backend for high-performance APIs
- 🎨 Interactive Streamlit frontend
- ☁️ Backend deployed on Render
- 🌐 Frontend deployed on Streamlit Cloud

## 🛠️ Tech Stack

### Machine Learning & NLP
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### Backend
- FastAPI
- Uvicorn
- Python-dotenv

### Frontend
- Streamlit

### APIs
- TMDB API

## 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                  # Streamlit Frontend
├── main.py                 # FastAPI Backend
├── df.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
├── indices.pkl
├── requirements.txt
├── .env
└── README.md
```

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=YOUR_TMDB_API_KEY
```

## ▶️ Run Backend

```bash
uvicorn main:app --reload
```

## ▶️ Run Frontend

```bash
streamlit run app.py
```

## 📸 Screenshots

Add screenshots of your application here.

## 🌍 Deployment

- **Backend:** Render
- **Frontend:** Streamlit Community Cloud

## 📚 Future Improvements

- Collaborative Filtering
- Hybrid Recommendation System
- User Authentication
- Watchlist Feature
- Personalized Recommendations
- Movie Reviews & Ratings

## 👨‍💻 Author

**Uttam Kumar Singh**

If you found this project useful, consider giving it a ⭐ on GitHub!
