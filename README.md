# Movie-Review-Model
# Movie Review Sentiment Analysis using Bag of Words and Naïve Bayes

## **Project Overview**
This project aims to classify movie reviews as either **Positive** or **Negative** using the **Bag of Words (BoW)** approach and **Naïve Bayes** classification. The dataset consists of multiple movie reviews, which are converted into numerical representations before applying machine learning models.

## **Project Workflow**

1. **Data Collection**  
   - The dataset contains movie reviews with sentiment labels (Positive/Negative).  
   - Each review consists of text that needs preprocessing before model training.

2. **Data Preprocessing**  
   - Convert all text to lowercase.  
   - Remove punctuation, special characters, and stopwords.  
   - Tokenize the reviews into individual words.  
   - Create a vocabulary of unique words.

3. **Feature Engineering (Bag of Words Representation)**  
   - Construct a vocabulary of unique words across all reviews.  
   - Represent each review as a vector, indicating the presence (1) or absence (0) of each word.  
   - Store the transformed data as a **Bag of Words matrix**.

4. **Training a Probabilistic Model (Naïve Bayes)**  
   - Compute word frequencies for both **Positive** and **Negative** reviews.  
   - Apply **Laplace Smoothing** to avoid zero probabilities.  
   - Compute probabilities for each word in both sentiment classes.  
   - Use Bayes’ Theorem to classify new reviews.

5. **Model Evaluation**  
   - Split the dataset into training and testing sets.  
   - Train the Naïve Bayes classifier using the Bag of Words features.  
   - Evaluate the model using accuracy, precision, recall, and F1-score.

6. **Comparison with Other Models**  
   - Experiment with other models like **Logistic Regression, Decision Trees, and Support Vector Machines (SVM)**.  
   - Compare performance metrics to determine the best classifier for sentiment analysis.

## **How to Run the Project**

### **1. Install Required Libraries**
```bash
pip install numpy pandas scikit-learn nltk
```

### **2. Load and Preprocess Data**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('movie_reviews.csv')  # Ensure you have the dataset
```

### **3. Preprocess and Convert Text to Bag of Words**
```python
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']
```

### **4. Train and Evaluate Model**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
```

## **Results and Observations**
- **Bag of Words** effectively converts text into numerical format.
- **Naïve Bayes** performs well for text classification with relatively small datasets.
- **Limitations**: BoW ignores word order and context, leading to potential misclassifications.
- **Improvement**: Consider using **TF-IDF** or **word embeddings (Word2Vec, BERT)** for better performance.

## **Conclusion**
This project demonstrates how to apply the **Bag of Words model with Naïve Bayes** for sentiment classification of movie reviews. It provides an easy-to-understand approach to text classification and lays the foundation for more advanced NLP techniques.

## **Future Enhancements**
- Implement **TF-IDF** to weigh words based on importance.
- Use deep learning models like **LSTMs** or **Transformers**.
- Handle sarcasm and context better by incorporating more advanced NLP techniques.
