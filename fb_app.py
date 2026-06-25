import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Facebook User Segmentation",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS - Facebook White Theme
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #1C1E21;
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #1877F2;
    margin-top:-10px;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #666666;
    margin-bottom: 25px;
}

.card {
    background-color: #F0F2F5;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

.prediction-box {
    background: linear-gradient(135deg, #1877F2, #42A5F5);
    color: white;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 25px;
}

.section-title {
    color: #1877F2;
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}

section[data-testid="stSidebar"] {
    background-color: #F0F2F5;
}

.stButton > button {
    background-color: #1877F2;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background-color: #166FE5;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #F0F2F5;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.07);
}

hr {
    border: 1px solid #E4E6EB;
}
</style>
""", unsafe_allow_html=True)

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
# Header
# --------------------------------------------------

from PIL import Image

logo = Image.open("facebook_logo.png")

# Center the logo
col1, col2, col3 = st.columns([4, 1, 4])

with col2:
    st.image(logo, width=105)

# Title
st.markdown("""
<div class="main-title">
Facebook User Segmentation
</div>

<div class="sub-title">
Segment Facebook users based on demographic and engagement behavior using Machine Learning
</div>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("👤 Enter User Details")

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
# Main Layout
# --------------------------------------------------
st.markdown("""
<div class="card">
This application uses a trained K-Means clustering pipeline to classify Facebook users into meaningful user segments.
Enter user details from the sidebar and click the prediction button.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.sidebar.button("Predict User Segment"):
    cluster = pipeline.predict(input_data)[0]
    info = cluster_info[int(cluster)]

    st.markdown(
        f"""
        <div class="prediction-box">
            🎯 Predicted Segment<br>
            {info["name"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cluster ID", int(cluster))

    with col2:
        st.metric("Age", age)

    with col3:
        st.metric("Friend Count", friend_count)

    with col4:
        st.metric("Tenure", f"{tenure} Days")

    st.markdown("---")

    st.markdown('<div class="section-title">📋 Segment Description</div>', unsafe_allow_html=True)
    st.info(info["description"])

    st.markdown('<div class="section-title">💡 Recommended Business Action</div>', unsafe_allow_html=True)
    st.success(info["recommendation"])

    st.markdown('<div class="section-title">📄 Input Summary</div>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)

else:
    st.markdown("""
    <div class="card">
    <h3 style="color:#1877F2;">How to Use</h3>
    <p>1. Enter user details in the sidebar.</p>
    <p>2. Click <b>Predict User Segment</b>.</p>
    <p>3. View the predicted Facebook user segment and business recommendation.</p>
    </div>
    """, unsafe_allow_html=True)