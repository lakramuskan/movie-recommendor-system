# 🎬 Movie Recommender System

A **Content-Based Movie Recommendation System** built using **Machine Learning** that suggests movies similar to a selected movie based on their features.

## 📌 Features

* Recommends movies based on **similarity score**
* Uses **cosine similarity**
* Built with **Python & Machine Learning**
* Interactive web interface (if using Streamlit / Flask)
* Efficient and scalable for large datasets


## 🧠 How It Works

1. Movie metadata is preprocessed
2. Important features are combined into a single representation
3. **Cosine Similarity** is calculated between movies
4. The system recommends top similar movies

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Pickle**
* **Streamlit / Flask** (if applicable)



## ⚠️ Important Note About `.pkl` Files

Large `.pkl` files (like `similarity.pkl`) are **not included** in the repository due to GitHub file size limits.

👉 You can generate them by running the training/preprocessing script:

```bash
python train.py
```

OR
Use **Git LFS** if you want to track large model files.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lakramuskan/movie-recommender-system.git
cd movie-recommender-system
```

---

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```bash
python app.py
```

OR (if Streamlit):

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

* Add collaborative filtering
* Improve UI/UX
* Deploy using AWS / Render / Streamlit Cloud
* Add user login & ratings

---

## 👩‍💻 Author

**Muskan**

* GitHub: [lakramuskan](https://github.com/lakramuskan)
* LinkedIn: [Muskan Lakra](https://www.linkedin.com/in/muskan-lakra-81872b281)

---

## ⭐ If you like this project

Don’t forget to **star ⭐ the repository**!


