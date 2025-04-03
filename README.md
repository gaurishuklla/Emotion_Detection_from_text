# Emotion_Detection_from_text
Emotion Detection from Text using Machine Learning 
This project is an AI-powered Emotion Detection Model that classifies text into different emotions using Natural Language Processing (NLP) and Machine Learning (ML).

##  Features
. Preprocessing & Cleaning: Removes URLs, punctuation, stopwords, and extra spaces
. TF-IDF Vectorization: Converts text into numerical format with 10,000 features & bigrams
. SMOTE Balancing: Handles imbalanced data to improve model performance
. RandomForest Classifier: Optimized with GridSearchCV for hyperparameter tuning
. Real-time Emotion Analysis: Enter a sentence and get the predicted emotion instantly

##  Dataset
The model is trained on the Tweet Emotions Dataset, which contains various emotion-labeled tweets.

##  Technologies Used
Python 

scikit-learn (ML Model & TF-IDF)

NeatText (Text Cleaning)

Seaborn & Matplotlib (Data Visualization)

SMOTE (imblearn) (Balancing Data)

Joblib (Model Saving & Loading)

##  How It Works?
1️. Load & Clean the Dataset
2️. Convert Text to Features (TF-IDF)
3️. Balance Dataset Using SMOTE
4️. Train & Optimize RandomForest Classifier
5️. Evaluate Performance & Test in Real-time

##  Performance
The optimized model achieves 80%+ accuracy by using feature engineering, hyperparameter tuning, and dataset balancing.

##  Usage
*  Install dependencies:

bash
Copy
Edit
pip install numpy pandas seaborn neattext scikit-learn imblearn joblib matplotlib
*  Run the script:

bash
Copy
Edit
python emotion_detection.py
*  Enter text input to get emotion predictions in real-time!

##  Future Improvements
. Deploy the model as an API or Web App
. Implement Deep Learning models like LSTMs/BERT for better accuracy
. Use a larger dataset for better generalization
