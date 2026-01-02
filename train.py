import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

data = 'Data/BBC News Train.csv'
model = 'Models/bbc_news_model.pkl'

def train():
    print("Beginning Training")

# Loading data from csv
    if not os.path.exists(data):
        print(f"Error: Data file can not be found at {data}")
        return
    
    print("Data loading")
    df = pd.read_csv(data)

#Spliting data into articles and categories
    X = df['Text']
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # creating pipeline which wil combine vectorizer and model together
    pipeline =  Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])

    #Training model
    print("Training model")
    pipeline.fit(X_train, y_train)

#Evaluating model
    print("Evaluating model")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy of model: {acc*100:.2f}%")
    print("\nClassification report: ")
    print(classification_report(y_test, preds))


    os.makedirs(os.path.dirname(model), exist_ok=True)

    print(f"Saving model @{model}")
    joblib.dump(pipeline, model)
    print("Training done")

if __name__ == "__main__":
    train()