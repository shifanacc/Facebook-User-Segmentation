import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Facebook User Segmentation",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
pipeline = joblib.load("facebook_segmentation_pipeline.pkl")

# --------------------------------------------------
# Cluster Information
# --------------------------------------------------
cluster_info = {
    0: {
        "name": "Casual Users",
        "description": "Users with low social interaction and minimal engagement.",
        "recommendation": "Use personalized notifications, friend suggestions, and content recommendations."
    },
    1: {
        "name": "Highly Engaged Users",
        "description": "Very active users with high likes, likes received, and platform interaction.",
        "recommendation": "Promote premium features, creator tools, and targeted advertisements."
    },
    2: {
        "name": "Senior Low-Activity Users",
        "description": "Older long-term users with lower engagement activity.",
        "recommendation": "Recommend family updates, memory features, and simplified content."
    },
    3: {
        "name": "Socially Active Users",
        "description": "Regular users with strong social connections and consistent platform activity.",
        "recommendation": "Suggest groups, events, communities, and personalized posts."
    }
}

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("📊 Facebook User Segmentation")
st.write(
    "This application predicts the user segment based on Facebook user demographic and engagement behavior."
)

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 13, 113, 25)

gender_label = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender_label == "Male" else 1

tenure = st.sidebar.number_input("Tenure (Days)", min_value=0, value=350)

friend_count = st.sidebar.number_input("Friend Count", min_value=0, value=500)
friendships_initiated = st.sidebar.number_input("Friendships Initiated", min_value=0, value=220)

likes = st.sidebar.number_input("Likes", min_value=0, value=150)
likes_received = st.sidebar.number_input("Likes Received", min_value=0, value=180)

mobile_likes = st.sidebar.number_input("Mobile Likes", min_value=0, value=120)
mobile_likes_received = st.sidebar.number_input("Mobile Likes Received", min_value=0, value=140)

www_likes = st.sidebar.number_input("Web Likes", min_value=0, value=30)
www_likes_received = st.sidebar.number_input("Web Likes Received", min_value=0, value=40)

# --------------------------------------------------
# Input DataFrame
# --------------------------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "tenure": [tenure],
    "friend_count": [friend_count],
    "friendships_initiated": [friendships_initiated],
    "likes": [likes],
    "likes_received": [likes_received],
    "mobile_likes": [mobile_likes],
    "mobile_likes_received": [mobile_likes_received],
    "www_likes": [www_likes],
    "www_likes_received": [www_likes_received]
})

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.sidebar.button("Predict User Segment"):
    cluster = pipeline.predict(input_data)[0]
    info = cluster_info[cluster]

    st.subheader("🎯 Prediction Result")

    st.success(f"Predicted Segment: {info['name']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cluster", cluster)

    with col2:
        st.metric("Age", age)

    with col3:
        st.metric("Friend Count", friend_count)

    st.markdown("### Segment Description")
    st.write(info["description"])

    st.markdown("### Recommended Business Action")
    st.write(info["recommendation"])

    st.markdown("### Input Summary")
    st.dataframe(input_data)