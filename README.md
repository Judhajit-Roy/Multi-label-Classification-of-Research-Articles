# Problem Description
Multi-label classification of Research Articles using several machine learning and deep learning methods. Dataset consists of titles and abstracts of articles from various domain. The task is to predict the domains of these articles.

# Hypothesis
Given a dataset for multi-label classification with large spans of sequential text, an advanced deep neural network performs than machine learning techniques.

Reason being, in large spans of text it is important that the model understands the data sequentially. In machine learning techniques, the sequence in which the text exists is overlooked while training the model. Whereas, in advanced deep learning techniques like BiLSTM, the network learns the data sequentially thus it understands the input better and gives better results.

# Dataset Description
Dataset consists of 20k+ research articles of the following domains:
- Computer Science
- Mathematics
- Physics
- Statistics
- Quantitative Biology
- Quantitative Finance

# Methods Used
For the machine learning the techniques used are: LinearSVC, Logistic Regression and Multinomial Naive Bayes.
For deep learning the technique used is Bidirectional LSTM.

# Baseline 
 The Multinomial Naive Bayes model is the considered as the baseline model.
 Since, it is a Multi-label classification problem the metric used Macro-F1.
 For the Multinomial Naive Bayes model the classification report is the following:
 
               precision    recall  f1-score   support

           0       0.60      0.87      0.71      1183
           1       0.20      1.00      0.33       243
           2       0.10      0.99      0.18       114
           3       0.00      0.00      0.00         0
           4       0.00      0.00      0.00         0
           5       0.00      0.00      0.00         0
           
           
   micro avg       0.26      0.90      0.41      1540
   macro avg       0.15      0.48      0.20      1540
weighted avg       0.50      0.90      0.61      1540
 samples avg       0.29      0.33      0.30      1540





