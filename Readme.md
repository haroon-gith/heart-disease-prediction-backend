---
title: Heart Disease Prediction
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
sdk_version: "3.10"
app_file: main.py
pinned: false
---

# Heart Disease Prediction API

This is a heart disease prediction model deployed on Hugging Face Spaces.

## How to use
Send a POST request to `/predict` with patient features.

## Model Info
- Trained with Scikit-learn KNN (K-Nearest Neighbors)
- Uses StandardScaler + One-Hot Encoding
- Returns prediction 0 (No Disease) or 1 (Heart Disease)
