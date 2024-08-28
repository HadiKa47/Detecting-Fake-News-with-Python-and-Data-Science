import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib 
import seaborn as sns
import matplotlib.pyplot as plt  


df = pd.read_csv(r'C:\Users\Dell-\Desktop\Projects\Detecting Fake News with Python and Data Science/news.csv')
print(df.shape)
print(df.head())

labels = df['label']
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 3))
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

smote = SMOTE(random_state=7)
x_train_smote, y_train_smote = smote.fit_resample(tfidf_train, y_train)

parameters = {'max_iter': [50, 100, 200], 'C': [0.1, 1, 10]}
grid_search = GridSearchCV(PassiveAggressiveClassifier(), parameters, cv=5, n_jobs=-1)
grid_search.fit(tfidf_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')

pac = grid_search.best_estimator_
pac.fit(x_train_smote, y_train_smote)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
print('Classification Report:')
print(report)

model1 = PassiveAggressiveClassifier(max_iter=50)
model2 = LinearSVC()
model3 = MultinomialNB()

voting_clf = VotingClassifier(estimators=[('pac', model1), ('svc', model2), ('nb', model3)], voting='hard')
voting_clf.fit(x_train_smote, y_train_smote)

y_pred_ensemble = voting_clf.predict(tfidf_test)
score_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Accuracy: {round(score_ensemble * 100, 2)}%')

scores = cross_val_score(pac, tfidf_vectorizer.transform(df['text']), labels, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy scores: {scores}')
print(f'Mean accuracy: {scores.mean()}')

joblib.dump(pac, 'fake_news_model.pkl')
print('Model saved as fake_news_model.pkl')