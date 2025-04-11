import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for styling including recommendation boxes and shine effect
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
    }
    .stSubheader {
        color: #34495e;
        font-family: 'Arial', sans-serif;
        font-size: 20px;
    }
    .recommendation-box {
        background-color: #ffffff;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 90%;
    }
    .highlight-shine {
        position: relative;
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, transparent, #e74c3c, transparent);
        animation: shine 2s infinite;
        text-align: center;
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    @keyframes shine {
        0% { background-position: -200%; }
        100% { background-position: 200%; }
    }
    .score-highlight {
        color: #e74c3c;
        font-weight: bold;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Step 1: Load Real Data or Use Mock Data
try:
    mentors = pd.read_csv('real_mentors_data.csv')
    aspirants = pd.read_csv('real_aspirants_data.csv')
    for col in ['Subjects', 'Colleges']:
        mentors[col] = mentors[col].apply(eval)
        aspirants[col] = aspirants[col].apply(eval)
except FileNotFoundError:
    print("Real data files not found. Using mock data as fallback.")
    mentors = pd.DataFrame({
        'Mentor_ID': ['M1', 'M2', 'M3', 'M4', 'M5'],
        'Subjects': [['English', 'Legal'], ['Math', 'Logical'], ['English', 'GK'], ['Legal', 'Logical'], ['English', 'Math']],
        'Colleges': [['NLSIU'], ['NALSAR'], ['NLU Delhi'], ['NLSIU', 'NALSAR'], ['NLU Delhi']],
        'Prep_Level': [3, 2, 2, 3, 1],
        'Learning_Style': ['Visual', 'Practical', 'Theoretical', 'Visual', 'Practical']
    })
    aspirants = pd.DataFrame({
        'Aspirant_ID': ['A1', 'A2', 'A3'],
        'Subjects': [['English', 'Legal'], ['Math', 'Logical'], ['English', 'GK']],
        'Colleges': [['NLSIU'], ['NALSAR'], ['NLU Delhi']],
        'Prep_Level': [2, 1, 3],
        'Learning_Style': ['Visual', 'Practical', 'Theoretical']
    })

for df in [mentors, aspirants]:
    df['Subjects'] = df['Subjects'].apply(lambda x: x if isinstance(x, list) else [])
    df['Colleges'] = df['Colleges'].apply(lambda x: x if isinstance(x, list) else [])
    df['Prep_Level'] = pd.to_numeric(df['Prep_Level'], errors='coerce').fillna(2).clip(1, 3)
    df['Learning_Style'] = df['Learning_Style'].fillna('Visual')

print("Mentors DataFrame:")
print(mentors)
print("\nAspirants DataFrame:")
print(aspirants)

# Step 2: Preprocess Data
def preprocess_data(df):
    all_subjects = ['English', 'Legal', 'Math', 'Logical', 'GK']
    subject_matrix = np.zeros((len(df), len(all_subjects)))
    for i, subjects in enumerate(df['Subjects']):
        for subj in subjects:
            if subj in all_subjects:
                subject_matrix[i, all_subjects.index(subj)] = 1
    
    all_colleges = ['NLSIU', 'NALSAR', 'NLU Delhi']
    college_matrix = np.zeros((len(df), len(all_colleges)))
    for i, colleges in enumerate(df['Colleges']):
        for col in colleges:
            if col in all_colleges:
                college_matrix[i, all_colleges.index(col)] = 1
    
    prep_level = np.array(df['Prep_Level']).reshape(-1, 1) / 3.0
    
    all_styles = ['Visual', 'Practical', 'Theoretical']
    learning_style_df = pd.DataFrame(df['Learning_Style'], columns=['Learning_Style'])
    learning_style_encoded = pd.get_dummies(learning_style_df['Learning_Style']).astype(float).reindex(columns=all_styles, fill_value=0).values
    
    feature_matrix = np.hstack([subject_matrix, college_matrix, prep_level, learning_style_encoded])
    return feature_matrix

mentor_features = preprocess_data(mentors)

# Step 2: Collaborative Filtering with Ratings Matrix
ratings = pd.DataFrame({
    'Aspirant_ID': ['A1', 'A2', 'A3'],
    'M1': [5, 3, 2],
    'M2': [2, 4, 1],
    'M3': [1, 2, 5],
    'M4': [3, 1, 3],
    'M5': [2, 2, 3]
}).set_index('Aspirant_ID')

ratings_matrix = ratings.values.astype(float)
U, sigma, Vt = svds(ratings_matrix, k=min(2, min(ratings_matrix.shape)-1))
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
aspirant_similarity = cosine_similarity(predicted_ratings)

def collaborative_recommendations(aspirant_id, ratings, top_n=3):
    if aspirant_id not in ratings.index:
        return [], []
    aspirant_idx = ratings.index.get_loc(aspirant_id)
    similar_aspirants = np.argsort(aspirant_similarity[aspirant_idx])[::-1][1:top_n+1]
    if len(similar_aspirants) == 0:
        return [], []
    mentor_scores = np.mean(ratings.iloc[similar_aspirants], axis=0)
    top_mentors = ratings.columns[np.argsort(mentor_scores)[::-1][:top_n]]
    return list(top_mentors), mentor_scores[np.argsort(mentor_scores)[::-1][:top_n]]

# Step 4: Optimize Weights Using Gradient Descent
def optimize_weights(ratings_matrix, features, learning_rate=0.01, epochs=100):
    weights = np.ones(features.shape[1])
    n_features = 4  # Subjects (5), Colleges (3), Prep_Level (1), Learning_Style (3)
    
    losses = []
    for epoch in range(epochs):
        predicted = np.zeros_like(ratings_matrix)
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                mentor_features_row = features[j % len(features)]
                feature_subset = np.hstack([mentor_features_row[:5], mentor_features_row[5:8], 
                                          mentor_features_row[8], mentor_features_row[9:]])
                predicted[i, j] = np.dot(feature_subset[:n_features], weights[:n_features])
        
        loss = np.mean((ratings_matrix - predicted) ** 2)
        losses.append(loss)
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        error = predicted - ratings_matrix
        gradient = np.zeros(n_features)
        for f in range(n_features):
            gradient[f] = np.mean(error * features[:, f] if f < 9 else error * features[:, 9 + (f-2)])
        
        weights[:n_features] -= learning_rate * gradient
    
    return weights, losses

# Step 3: Streamlit App with Improved Interface
st.markdown('<div class="main">', unsafe_allow_html=True)

# Sidebar for Inputs
with st.sidebar:
    st.title("Aspirant Preferences")
    subjects = st.multiselect(
        "Select Subjects",
        ['English', 'Legal', 'Math', 'Logical', 'GK'],
        default=['English'],
        help="Choose the subjects you are interested in."
    )
    colleges = st.multiselect(
        "Select Colleges",
        ['NLSIU', 'NALSAR', 'NLU Delhi'],
        default=['NLSIU'],
        help="Select your preferred colleges."
    )
    prep_level = st.slider(
        "Preparation Level (1-3)",
        1, 3, 2,
        help="Indicate your current preparation level (1 = Beginner, 3 = Advanced)."
    )
    learning_style = st.selectbox(
        "Learning Style",
        ['Visual', 'Practical', 'Theoretical'],
        index=0,
        help="Select your preferred learning style."
    )
    if st.button("Reset Preferences"):
        st.session_state.clear()
        st.rerun()  # Replaced experimental_rerun with rerun

if 'asp_features' not in st.session_state:
    st.session_state.asp_features = None

# Tabs for Recommendations
tab1, tab2 = st.tabs(["Recommendations", "Optimization"])

with tab1:
    if st.button("Get Recommendations"):
        aspirant_df = pd.DataFrame({
            'Aspirant_ID': ['A_temp'],
            'Subjects': [subjects],
            'Colleges': [colleges],
            'Prep_Level': [prep_level],
            'Learning_Style': [learning_style]
        })
        st.session_state.asp_features = preprocess_data(aspirant_df)
        
        asp_features = st.session_state.asp_features
        
        # Content-based recommendations
        content_scores = cosine_similarity(asp_features, mentor_features)[0]
        content_indices = np.argsort(content_scores)[::-1][:3]
        content_mentors = mentors.iloc[content_indices]['Mentor_ID'].values
        content_top_scores = content_scores[content_indices]
        
        # Collaborative recommendations
        collab_mentors, collab_scores = collaborative_recommendations('A1', ratings)
        
        # Hybrid scores
        collab_dict = dict(zip(collab_mentors, collab_scores))
        hybrid_scores = 0.7 * content_scores + 0.3 * np.array([collab_dict.get(m, 0) for m in mentors['Mentor_ID']])
        hybrid_indices = np.argsort(hybrid_scores)[::-1][:3]
        hybrid_mentors = mentors.iloc[hybrid_indices]['Mentor_ID'].values
        hybrid_top_scores = hybrid_scores[hybrid_indices]
        
        st.markdown("<h3 style='color: #2c3e50;'>Recommendations</h3>", unsafe_allow_html=True)
        # Display top 3 recommendations as boxes
        for i, (mentor, score) in enumerate(zip(hybrid_mentors, hybrid_top_scores)):
            mentor_data = mentors[mentors['Mentor_ID'] == mentor].iloc[0]
            if i == 0:
                st.markdown(
                    "<div class='highlight-shine'>Highly Recommended</div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <div class='recommendation-box'>
                    <h4>{mentor_data['Mentor_ID']} Profile</h4>
                    <p><strong>Subjects:</strong> {', '.join(mentor_data['Subjects'])}</p>
                    <p><strong>Colleges:</strong> {', '.join(mentor_data['Colleges'])}</p>
                    <p><strong>Prep Level:</strong> {mentor_data['Prep_Level']}</p>
                    <p><strong>Learning Style:</strong> {mentor_data['Learning_Style']}</p>
                    <div class='score-highlight'>Score: {score:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with st.expander("Content-Based Recommendations"):
            for mentor, score in zip(content_mentors, content_top_scores):
                st.write(f"Mentor {mentor} with score: {score:.3f}")
            fig1, ax1 = plt.subplots()
            ax1.bar(content_mentors, content_top_scores, color='skyblue')
            ax1.set_title('Content-Based Similarity Scores')
            ax1.set_ylabel('Score')
            st.pyplot(fig1)
        
        with st.expander("Collaborative Recommendations (Based on A1)"):
            for mentor, score in zip(collab_mentors, collab_scores):
                st.write(f"Mentor {mentor} with score: {score:.3f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(collab_mentors, collab_scores, marker='o', color='green')
            ax2.set_title('Collaborative Scores for A1')
            ax2.set_ylabel('Average Rating')
            st.pyplot(fig2)
        
        with st.expander("Hybrid Recommendations (Detailed)"):
            for mentor, score in zip(hybrid_mentors, hybrid_top_scores):
                st.write(f"Mentor {mentor} with score: {score:.3f}")
            fig3, ax3 = plt.subplots()
            ax3.scatter(hybrid_mentors, hybrid_top_scores, color='red', s=100)
            ax3.set_title('Hybrid Recommendation Scores')
            ax3.set_ylabel('Score')
            st.pyplot(fig3)

with tab2:
    if st.button("Optimize Weights and Recommend"):
        if st.session_state.asp_features is None:
            st.error("Please get recommendations first to compute aspirant features.")
        else:
            asp_features = st.session_state.asp_features
            optimized_weights, losses = optimize_weights(ratings_matrix, mentor_features)
            st.markdown("<h3 style='color: #2c3e50;'>Optimization Results</h3>", unsafe_allow_html=True)
            st.write("Optimized Weights:", [round(w, 3) for w in optimized_weights[:4]])
            
            fig4, ax4 = plt.subplots()
            ax4.plot(range(len(losses)), losses, color='purple')
            ax4.set_title('Loss During Weight Optimization')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            st.pyplot(fig4)
            
            weighted_mentor_features = np.hstack([
                mentor_features[:, :5] * optimized_weights[0],  # Subjects
                mentor_features[:, 5:8] * optimized_weights[1],  # Colleges
                mentor_features[:, 8:9] * optimized_weights[2],  # Prep_Level
                mentor_features[:, 9:] * optimized_weights[3]    # Learning_Style
            ])
            
            asp_features_weighted = np.hstack([
                asp_features[:, :5] * optimized_weights[0],
                asp_features[:, 5:8] * optimized_weights[1],
                asp_features[:, 8:9] * optimized_weights[2],
                asp_features[:, 9:] * optimized_weights[3]
            ])
            
            opt_scores = cosine_similarity(asp_features_weighted, weighted_mentor_features)[0]
            opt_indices = np.argsort(opt_scores)[::-1][:3]
            opt_mentors = mentors.iloc[opt_indices]['Mentor_ID'].values
            opt_top_scores = opt_scores[opt_indices]
            
            with st.expander("Optimized Recommendations"):
                for mentor, score in zip(opt_mentors, opt_top_scores):
                    st.write(f"Mentor {mentor} with score: {score:.3f}")
            
            score_matrix = np.zeros((len(mentors), 1))
            for i, mentor in enumerate(mentors['Mentor_ID']):
                score_matrix[i] = opt_scores[i] if mentor in opt_mentors else 0
            fig5, ax5 = plt.subplots()
            sns.heatmap(score_matrix, annot=True, xticklabels=['Score'], yticklabels=mentors['Mentor_ID'], cmap='YlOrRd', ax=ax5)
            ax5.set_title('Optimized Similarity Scores Heatmap')
            st.pyplot(fig5)

st.markdown('</div>', unsafe_allow_html=True)

# Run with: streamlit run App.py
