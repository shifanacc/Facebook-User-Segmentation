#  Facebook User Segmentation using Unsupervised Machine Learning

## Overview

This project applies **Unsupervised Machine Learning** techniques to segment Facebook users based on demographic information and engagement behavior. Multiple clustering algorithms were implemented and compared to identify meaningful user groups that can support personalized recommendations, targeted marketing, and user engagement strategies.

The project also includes an interactive **Streamlit web application** that predicts the user segment for new Facebook users using a trained K-Means clustering pipeline.

---

## Project Objectives

- Understand the Facebook user dataset.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the data.
- Apply feature scaling.
- Build clustering models using:
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN
- Evaluate clustering performance.
- Visualize clusters using PCA.
- Profile and interpret each user segment.
- Build a machine learning pipeline.
- Develop an interactive Streamlit application.
- Deploy the application.

---

## Dataset Features

- Age
- Gender
- Tenure
- Friend Count
- Friendships Initiated
- Likes
- Likes Received
- Mobile Likes
- Mobile Likes Received
- Web Likes
- Web Likes Received

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## Machine Learning Algorithms

### K-Means Clustering
Used as the primary clustering algorithm for user segmentation.

### Hierarchical Clustering
Applied to analyze hierarchical relationships between users.

### DBSCAN
Used to identify density-based clusters and detect noise points.

---

## Evaluation Metrics

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- PCA Visualization

---

## User Segments

### Cluster 0 – Casual Users
- Low engagement
- Minimal social interaction
- Occasional platform usage

### Cluster 1 – Highly Engaged Users
- High likes and interactions
- Very active users
- Strong mobile engagement

### Cluster 2 – Senior Low-Activity Users
- Older users
- Long platform tenure
- Lower engagement

### Cluster 3 – Socially Active Users
- High friend count
- Active social participation
- Consistent engagement

---

## Streamlit Application

The application allows users to:

- Enter Facebook user details
- Predict the user segment
- View cluster description
- Receive business recommendations
- Review the input summary

---

## Project Structure

```
Facebook-User-Segmentation/
│
├── Face Book data.ipynb
├── app.py
├── facebook_segmentation_pipeline.pkl
├── facebook_data.csv
├── requirements.txt
└── README.md
```

---

## Run Locally

### Clone the repository

```bash
git clone https://github.com/shifanacc/Facebook-User-Segmentation.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run fb_app.py
```

---

## Live Demo

**Streamlit App:**  

Example:

https://facebook-user-segmentation-vdcdbuj3khqu2eon2hgsvt.streamlit.app/

---

## Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. K-Means Clustering
6. Hierarchical Clustering
7. DBSCAN
8. Cluster Evaluation
9. PCA Visualization
10. Cluster Profiling
11. Pipeline Development
12. Streamlit Application
13. Deployment

---

## Conclusion

This project successfully demonstrates an end-to-end unsupervised machine learning workflow for Facebook user segmentation. After comparing multiple clustering algorithms, K-Means produced the most meaningful and interpretable user groups. The developed Streamlit application enables interactive prediction of user segments, making the solution suitable for real-world user behavior analysis and personalized engagement strategies.

---
