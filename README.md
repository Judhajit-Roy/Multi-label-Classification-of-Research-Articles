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
 
![alt text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/dataset.JPG)

In a multi-label classification problem a sample can have more than 1 class assigned to it. (For Eg: The last sample in the image is to both Computer Science and Statistics)

**Distribution of samples as per classes**

![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/data%20plot.png)


# Methods Used
The machine learning the techniques used are: 

LinearSVC, Logistic Regression and Multinomial Naive Bayes.

- In the machine learning techniques, the title and abstract are concatenated. 
- The text obtained is preprocessed using regular expressions by replacing integers and decimals with a placeholder num, removing punctuation, replacing extra white spaces with   single space, stripping spaces from the start and end and converting the text to lower case.
- Further, the text is tokenized and converted into TFIDF features before training.

Deep learning technique used is Bidirectional LSTM.

- Similar to the machine learning approach the text is first concatenated and then preprocessed.
- After preprocessing Glove embedding is used to transform the text into embedded vectors of 200 dimensions before training.
- The model is trained for 20 epochs with the 'Adam' optimizer and Binary Cross Entropy loss is calculated for back propagation.

# Baseline and Evaluation metric
The Multinomial Naive Bayes model is the considered as the baseline model as it is simple to implement and has a chance of obtaining reasonable results.
Since, it is a Multi-label classification problem the metric used Macro-F1. Macro-F1 is average of the individual F1 scores of every class. Therefore, it gives equal importance to each class.
For the Multinomial Naive Bayes model the classification report is the following:
 
![alt text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/nb%20result.JPG)

Thus, the baseline Macro-f1 score is 0.20.

# Results

For Machine learning techniques:


**Logistic Regression**

![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/lr%20result.JPG)

**Linear SVC**

![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/svc%20result.JPG)

For Deep Learning technique:

**Bidirectional LSTM**

*Model Summary*


![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/BiLstm%20model%20summary.JPG)


*Results*


![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/bilstm%20metrics.JPG)


*Accuracy Plot*


![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/bilstm%20accuracy%20plot.JPG)


*Loss Plot*


![alt_text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/bilstm%20loss%20plot.JPG)

# Directions












