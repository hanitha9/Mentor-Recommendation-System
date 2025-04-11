import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Mock Data
mentors = pd.DataFrame({
    'Mentor_ID': ['M1', 'M2', 'M3', 'M4', 'M5'],
    'Subjects': [['English', 'Legal'], ['Math', 'Logical'], ['English', 'GK'], ['Legal', 'Logical'], ['English', 'Math']],
    'Colleges': [['NLSIU'], ['NALSAR'], ['NLU Delhi'], ['NLSIU', 'NALSAR'], ['NLU Delhi']],
    'Prep_Level': [3, 2, 2, 3, 1],
    'Learning_Style': ['Visual', 'Practical', 'Theoretical', 'Visual', 'Practical']
})

aspirants = pd.DataFrame({
    'Aspirant_ID': ['A_temp'],
    'Subjects': [['English', 'Legal']],
    'Colleges': [['NLSIU']],
    'Prep_Level': [2],
    'Learning_Style': ['Visual']
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
asp_features = preprocess_data(aspirants)

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

collab_mentors, collab_scores = collaborative_recommendations('A1', ratings)

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
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        error = predicted - ratings_matrix
        gradient = np.zeros(n_features)
        for f in range(n_features):
            gradient[f] = np.mean(error * features[:, f] if f < 9 else error * features[:, 9 + (f-2)])
        
        weights[:n_features] -= learning_rate * gradient
    
    return weights, losses

optimized_weights, losses = optimize_weights(ratings_matrix, mentor_features)

# Content-based recommendations
content_scores = cosine_similarity(asp_features, mentor_features)[0]
content_indices = np.argsort(content_scores)[::-1][:3]
content_mentors = mentors.iloc[content_indices]['Mentor_ID'].values
content_top_scores = content_scores[content_indices]

# Hybrid scores
collab_dict = dict(zip(collab_mentors, collab_scores))
hybrid_scores = 0.7 * content_scores + 0.3 * np.array([collab_dict.get(m, 0) for m in mentors['Mentor_ID']])
hybrid_indices = np.argsort(hybrid_scores)[::-1][:3]
hybrid_mentors = mentors.iloc[hybrid_indices]['Mentor_ID'].values
hybrid_top_scores = hybrid_scores[hybrid_indices]

# Optimized recommendations
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

# Visualizations
# Content-Based Scores
plt.figure(figsize=(8, 5))
plt.bar(content_mentors, content_top_scores, color='skyblue')
plt.title('Content-Based Similarity Scores')
plt.ylabel('Score')
plt.show()

# Collaborative Scores
plt.figure(figsize=(8, 5))
plt.plot(collab_mentors, collab_scores, marker='o', color='green')
plt.title('Collaborative Scores for A1')
plt.ylabel('Average Rating')
plt.show()

# Hybrid Scores
plt.figure(figsize=(8, 5))
plt.scatter(hybrid_mentors, hybrid_top_scores, color='red', s=100)
plt.title('Hybrid Recommendation Scores')
plt.ylabel('Score')
plt.show()

# Loss During Optimization
plt.figure(figsize=(8, 5))
plt.plot(range(len(losses)), losses, color='purple')
plt.title('Loss During Weight Optimization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Optimized Similarity Scores Heatmap
score_matrix = np.zeros((len(mentors), 1))
for i, mentor in enumerate(mentors['Mentor_ID']):
    score_matrix[i] = opt_scores[i] if mentor in opt_mentors else 0
plt.figure(figsize=(8, 5))
sns.heatmap(score_matrix, annot=True, xticklabels=['Score'], yticklabels=mentors['Mentor_ID'], cmap='YlOrRd')
plt.title('Optimized Similarity Scores Heatmap')
plt.show()

# Print Results
print("\nContent-Based Recommendations:")
for mentor, score in zip(content_mentors, content_top_scores):
    print(f"Mentor {mentor} with score: {score:.3f}")

print("\nCollaborative Recommendations (Based on A1):")
for mentor, score in zip(collab_mentors, collab_scores):
    print(f"Mentor {mentor} with score: {score:.3f}")

print("\nHybrid Recommendations:")
for mentor, score in zip(hybrid_mentors, hybrid_top_scores):
    print(f"Mentor {mentor} with score: {score:.3f}")

print("\nOptimized Weights:", [round(w, 3) for w in optimized_weights[:4]])
print("\nOptimized Recommendations:")
for mentor, score in zip(opt_mentors, opt_top_scores):
    print(f"Mentor {mentor} with score: {score:.3f}")
