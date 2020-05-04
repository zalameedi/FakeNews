# Programmers : David Henshaw & Zeid Al-Ameedi
# Date : 05-01-2020
# Built to succeed the challenge found http://www.fakenewschallenge.org/
# Dataset courtesy of https://github.com/KaiDMML/FakeNewsNet
# Computer Science 315 - Introduction to Data Mining (Final Project)



# Hosted on Zeid Al-Ameedi's github
# Copyright © 2020 zalameedi


"""
Data science Modules.
Pandas : Fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
sklearn : Free software machine learning library.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

"""
The Fake News Class was constructed to be imported. The goal of the project is to deliver a module
that can be imported into existing scripts or applications. Once inserted, the module can be passed a large data set
(perhaps daily, weekly or monthly news?) and return validity upon each piece of news (binary label -real or fake).
"""
class FakeNewsClassifier:
    def __init__(self):
        pass

    """
    Run() takes no arguments. This function is meant to go through the process of solving the FN issue.
    The algorithm works by building labels, vectorizing the data, TF-IDF implementation to reduce common words.
    Next we remove stop words. Finally after splitting and transforming the data we build our classifier and predict.
    We'll return a score of accuracy on how well the classifier did with our data set.
    """
    def run(self):
        df = self.read_csv()
        labels = self.create_labels(df)
        x_test, x_train, y_test, y_train = self.traintest_split(df, labels)
        tfidf_vectorizer = self.tfid_vectorizer_fn()
        tfidf_test, tfidf_train = self.transform_helper(tfidf_vectorizer, x_test, x_train)
        pac = self.build_classifier2(tfidf_train, y_train)
        score = self.c_predict(pac, tfidf_test, y_test)
        self.report(score)

    def report(self, score):
        print(f'Accuracy: {round(score * 100, 2)}%')

    def c_predict(self, pac, tfidf_test, y_test):
        # predicting Data Set
        y_pred = pac.predict(tfidf_test)
        score = accuracy_score(y_test, y_pred)
        return score

    def build_classifier(self, tfidf_train, y_train):
        # Initializing Passive Aggresive Classifier
        pac = PassiveAggressiveClassifier(max_iter=1000)
        pac.fit(tfidf_train, y_train)
        return pac

    def build_classifier2(self, tfidf_train, y_train):
        # Initializing Perceptron Classifier
        per = Perceptron(tol=1e-3, random_state=0)
        per.fit(tfidf_train, y_train)
        return per

    def transform_helper(self, tfidf_vectorizer, x_test, x_train):
        # tranforing data set into test and train sets
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test)
        return tfidf_test, tfidf_train

    def tfid_vectorizer_fn(self):
        # Initialzing TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        return tfidf_vectorizer

    def traintest_split(self, df, labels):
        # Splitting of Data into test and train case.
        x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
        return x_test, x_train, y_test, y_train

    def create_labels(self, df):
        # Defining Labels
        labels = df.label
        labels.head()
        return labels

    def read_csv(self):
        # Importing the data
        df = pd.read_csv('news.csv')
        print(df.shape)
        print(df.head())
        return df


def main():
    fn = FakeNewsClassifier()
    fn.run()


if __name__ == '__main__':
    main()
