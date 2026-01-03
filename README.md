# News Topic Classifier 

## Project Overview
This project implements a News Topic Classifier that uses Machine Learning to automatically analyze and categorise news articles into five distinct topics: Business, Sport, Technology, Politics, and Entertainment.

By using NLP techniques, the model identifies key patterns within the text to accurately determine the most relevant topic for each article. This project demonstrates how automated prediction pipelines can replace manual data entry processes.

---

## Technology Stack & Dataset

### Tech Stack
* **Language:** Python 3.13
* **Core Library:** Scikit-Learn (sklearn)
* **Data Processing:** Pandas
* **Model Serialization:** Joblib
* **Environment:** VS Code 

### The Dataset
* **Source:** [BBC News Classification Dataset](https://www.kaggle.com/c/learn-ai-bbc) (Kaggle).
* **Volume:** ~2,225 news articles.
* **Input:** Unstructured article text (Headline + Body).
* **Output:** 5 Categories (Business, Tech, Politics, Sport, Entertainment).

---

## ⚙️ Installation & Usage
1. Prerequisites:
- Ensure you have Python installed. It is recommended to use a virtual environment.

2. Install Dependencies:
- pip install -r requirements.txt

3. Training the Model:
- Run the training script to process the data and generate the model artifact (.pkl file).
- python train.py
- Output: You will see the accuracy score and a detailed classification report in the terminal.

4. Running the Application:
- Launch the interactive CLI tool to test predictions in real-time.
- python predict.py
- Usage: Type any news headline when prompted, and the system will return the predicted category and confidence score.

  ---

## Methodology: 
- Text vectorization is the process of converting textual data into numerical representations that machine-learning models can understand.TF-IDF (Term Frequency–Inverse Document Frequency) is a text vectorization technique that measures the importance of words based on how frequently they appear in a document relative to the entire dataset. It is well suited for this project because it down-weights common words such as “like”, “as”, “is”, and “and”, while emphasizing topic-specific terms that help distinguish between different news categories.

- For the model, the most suitable classifier chosen was Naive Bayes, specifically Multinomial Naive Bayes, as it is well suited for text classification tasks. This classifier assumes that all features are conditionally independent of one another, an assumption that works effectively for text data where the presence of specific words strongly correlates with particular topics.
  
- The pipeline was used to combine multiple stages of the machine-learning workflow into a single, unified process. In this project, it integrates the text vectorization stage, which is responsible for text preprocessing and feature extraction using TF-IDF, with the classification stage. The primary reason for using a pipeline is to prevent data leakage, which occurs when information from outside the training dataset influences the model during training. This can lead to overly optimistic performance metrics and reduced generalisation. Additionally, the pipeline ensures that all input data is processed through the same consistent preprocessing steps, guaranteeing that new and unseen text is transformed in exactly the same way as the training data.

---

## Future Applications 
This architecture is not limited to news articles but can also be expanded and adapted within other different domains, it can aid in improving decision making, in a quicker and more accurate manner.


