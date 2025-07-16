# ⚽ FPL-team-optimizer

A machine learning-based Fantasy Premier League (FPL) optimizer that predicts player points for upcoming gameweeks and selects the best possible team using Integer Linear Programming (ILP).  
This project combines data science, machine learning, and optimization to help you win your mini-leagues!

---

## 🧠 Features

- 🔍 **Position-specific ML models** (GK, DEF, MID, FWD) trained on past seasons from 22-23 to 24-25 data 
- 📈 Predicts player points using stats up to the previous gameweek using machine learning 
- ⚙️ Selects the best 11-player team using ILP under FPL constraints (budget, formation, etc.)
- 📊 Supports captain/vice-captain assignment and multiple formations
- 🖥️ Modern **Streamlit UI** with team visualization and filtering
- 📦 Fully self-contained project with datasets and trained models

---
## Points Prediction 

## 📊 Model Details

| Position  | Model Used                      | Best Params / R² Score |
|-----------|----------------------------------|--------------------------|
| **GK**    | Lasso Regression                 | `alpha=0.1`, R² ≈ 0.40   |
| **DEF**   | HistGradientBoosting             | `max_depth=3`, `max_iter=150`, `min_samples_leaf=50`, `l2_regularization=0.1` |
| **MID**   | HistGradientBoosting             | same as DEF              |
| **FWD**   | Ridge Regression                 | default params           |


- Feature engineering was done **separately** for each position.
- Models were selected using R2 score as reference
- Goalkeepers had the best R2 score of 0.4004 
- Data was derived from [Vaastav’s GitHub](https://github.com/vaastav/Fantasy-Premier-League), covering **2022–23 to 2024–25** seasons.

---

## 🧮 Optimization

The final 11-player team is selected using:

- 💸 Budget constraint (max £100M)
- 📋 Squad rules (max 3 players per club, valid formation)
- 💯 Objective: **Maximize predicted points**

Implemented using **PuLP** (Linear Programming in Python).


## 🚀 Run Locally

### 1. Clone the repo and install dependencies
  pip install requirements.txt
  
### 2. Run streamlit app
  streamlit run app.py 
  or try
  python -m streamlit run app.py


<h2 align="center">⚽ App Screenshots</h2>

<p align="center">
  <img src="assets/Main%20output.png" alt="Main output" width="400"/>
  <img src="assets/Player%20cards.png" alt="Player cards" width="400"/>
</p>

<p align="center">
  <b>Main output</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Player cards</b>
</p>

<p align="center">
  <img src="assets/Team%20lineup.png" alt="Team lineup" width="400"/>
  <img src="assets/Team%20list.png" alt="Team list" width="400"/>
</p>

<p align="center">
  <b>Team lineup</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Team list</b>
</p>




🧑‍💻 Author

-Shaunak Paranjape

Feel free to connect or star ⭐ the repo if you found it useful!









