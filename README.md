# Mentor Recommendation System

This repository contains two implementations of a mentor recommendation system: a **Streamlit web application** (`app.py`) and a **standalone Python script** (`recommendations.py`). Both versions leverage machine learning techniques to recommend mentors to aspirants based on their preferences and compatibility. The system uses content-based filtering, collaborative filtering, hybrid recommendations, and weight optimization to provide personalized mentor suggestions.

## Table of Contents
- [Overview](#overview)
- [Important Concepts](#important-concepts)
- [How It Works](#how-it-works)
  - [Data Preprocessing](#data-preprocessing)
  - [Content-Based Filtering](#content-based-filtering)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Hybrid Recommendations](#hybrid-recommendations)
  - [Weight Optimization](#weight-optimization)
  - [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
  - [Streamlit App (`app.py`)](#streamlit-app-apppy)
  - [Standalone Script (`recommendations.py`)](#standalone-script-recommendationspy)
- [Files](#files)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
The Mentor Recommendation System is designed to match aspirants (students or learners) with mentors based on their subjects of interest, preferred colleges, preparation levels, and learning styles. The system provides three types of recommendations:
- **Content-Based**: Matches based on similarity in preferences.
- **Collaborative**: Uses ratings from similar aspirants.
- **Hybrid**: Combines content-based and collaborative approaches.
- **Optimized**: Enhances recommendations using optimized feature weights.

The Streamlit app offers an interactive interface with visualizations, while the standalone script provides the same functionality with console output and plot windows.

## Important Concepts
1. **Content-Based Filtering**:
   - Recommends mentors based on the similarity between an aspirant’s preferences and mentor profiles using cosine similarity.
   - Features include subjects, colleges, preparation level, and learning style.

2. **Collaborative Filtering**:
   - Utilizes a ratings matrix and Singular Value Decomposition (SVD) to identify patterns among aspirants and recommend mentors based on similar users’ preferences.
   - Employs cosine similarity to find similar aspirants.

3. **Hybrid Recommendations**:
   - Combines content-based and collaborative filtering with a weighted average (70% content, 30% collaborative) to leverage the strengths of both methods.

4. **Weight Optimization**:
   - Uses gradient descent to optimize weights for different feature categories (subjects, colleges, preparation level, learning style), minimizing the mean squared error between predicted and actual ratings.

5. **Data Preprocessing**:
   - Converts categorical and numerical data into a feature matrix using one-hot encoding and normalization.

6. **Visualizations**:
   - Bar charts, line plots, scatter plots, and heatmaps to represent recommendation scores and optimization loss.

## How It Works

### Data Preprocessing
- **Input Data**: The system uses mock data (or real data from CSV files if available) containing mentor and aspirant profiles with columns: `Mentor_ID`/`Aspirant_ID`, `Subjects`, `Colleges`, `Prep_Level`, and `Learning_Style`.
- **Feature Encoding**:
  - Subjects and colleges are one-hot encoded into binary matrices.
  - Preparation level is normalized (divided by 3 to scale between 0 and 1).
  - Learning style is one-hot encoded into three categories (Visual, Practical, Theoretical).
- **Feature Matrix**: Combines all encoded features into a single matrix for similarity calculations.

### Content-Based Filtering
- Computes cosine similarity between the aspirant’s feature vector and all mentor feature vectors.
- Returns the top 3 mentors with the highest similarity scores.
- Visualization: Bar chart showing similarity scores.

### Collaborative Filtering
- Uses a ratings matrix where aspirants rate mentors (mock data provided).
- Applies SVD to reduce dimensionality and predict ratings.
- Calculates cosine similarity between aspirant rating vectors to find similar users.
- Averages ratings from similar aspirants to recommend mentors.
- Visualization: Line plot of average ratings for top mentors.

### Hybrid Recommendations
- Combines content-based scores (70% weight) and collaborative scores (30% weight).
- Sorts mentors by hybrid scores and selects the top 3.
- Visualization: Scatter plot of hybrid scores.
- The Streamlit app displays these as styled boxes with a "Highly Recommended" shine effect for the top mentor.

### Weight Optimization
- Initializes equal weights for feature categories.
- Uses gradient descent to minimize the mean squared error between predicted ratings (based on weighted features) and actual ratings.
- Iterates over 100 epochs, updating weights to improve prediction accuracy.
- Visualization: Line plot of loss over epochs.

### Visualizations
- **Content-Based**: Bar chart of top 3 mentor scores.
- **Collaborative**: Line plot of average ratings.
- **Hybrid**: Scatter plot of top 3 scores.
- **Optimization**: Line plot of loss trend.
- **Optimized**: Heatmap of similarity scores for top mentors.
- In the Streamlit app, these are interactive; in the script, they are static plots.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Mentor-Recommendation-System.git
cd Mentor-Recommendation-System
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn scipy streamlit matplotlib seaborn
```

3. (Optional) If using real data, place `real_mentors_data.csv` and `real_aspirants_data.csv` in the repository root with the same structure as the mock data.

## Usage

### Streamlit App (`app.py`)
- Run the app:
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.

- **Interface**:
  - Sidebar: Select subjects, colleges, preparation level (1-3), and learning style.
  - "Reset Preferences": Clears selections and reruns the app.
  - "Get Recommendations": Displays three recommendation boxes with profiles and scores, plus expandable visualizations.
  - "Optimize Weights and Recommend": Shows optimized weights, loss plot, and optimized recommendations with a heatmap.

- **Features**:
  - Interactive input with styled boxes and a "Highly Recommended" shine effect.
  - Visualizations for all recommendation types and optimization.

### Standalone Script (`recommendations.py`)
- Run the script:
```bash
python recommendations.py
```

- **Output**:
  - Console prints top 3 recommendations for content-based, collaborative, hybrid, and optimized methods with scores.
  - Optimized weights are displayed.
  - Five plot windows open sequentially showing bar charts, line plots, scatter plots, loss trend, and heatmap.

- **Features**:
  - No interface; purely computational with static visualizations.
### video walkthrough
    https://drive.google.com/file/d/1i5gN80nmIV7bHZqJ_evZC4CrGfWUtq3o/view?usp=sharing
### live demo on streamlit
     you can access the page by this below link:
     https://mentor-recommendation-system-jc2mhim7hspuezbhxittaa.streamlit.app/
### screenshots
    streamlit version:
  ![Screenshot 2025-04-11 234924](https://github.com/user-attachments/assets/142629b0-e377-4163-88a9-1ab27c03875b)

  ![Screenshot 2025-04-11 235002](https://github.com/user-attachments/assets/84d658c3-ced1-49f5-a96a-88b367ba42c1)

  ![Screenshot 2025-04-11 235117](https://github.com/user-attachments/assets/2cbeb95e-7883-4dc8-99cb-9734c19b56a3)

  ![Screenshot 2025-04-11 235128](https://github.com/user-attachments/assets/437db79f-fca4-4268-8328-cbf47761f05f)

![Screenshot 2025-04-11 235209](https://github.com/user-attachments/assets/356ca947-5a57-4928-b4f5-3e3592cdd0b4)

![Screenshot 2025-04-11 235237](https://github.com/user-attachments/assets/4b6d506b-9c40-49a5-b5da-324f335c9b9b)

![Screenshot 2025-04-11 235250](https://github.com/user-attachments/assets/bea4271a-e7f3-4a0a-a4be-602f6f4b06e7)

![Screenshot 2025-04-11 235303](https://github.com/user-attachments/assets/845464dc-7ab1-412f-94f8-314a3c579a3e)

![Screenshot 2025-04-11 235314](https://github.com/user-attachments/assets/0f772d9d-d3f8-4dd7-8d8e-203dbe80f95d)

 "console look like this when app running":
 
![Screenshot 2025-04-12 000245](https://github.com/user-attachments/assets/37494508-68d2-4581-9dfa-405e300986f4)


 standard version:

 ![Screenshot 2025-04-11 234909](https://github.com/user-attachments/assets/1d9dc51f-28ad-424f-9460-ab06bdd354d5)




## Files
- **`app.py`**: Streamlit-based web application with interactive UI and visualizations.
- **`recommendations.py`**: Standalone Python script with console output and plot windows.
- **`README.md`**: This file, providing documentation.

## Contact
- **Author**: hanitha9
- **Email**: hanitharajeswari9@gmail.com

For questions or support, feel free to open an issue or contact me directly.
