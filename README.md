# Problem Description
Multi-label classification of Research Articles using several machine learning and deep learning methods. Dataset consists of titles and abstracts of articles from various domain. The task is to predict the domains of these articles.

# Hypothesis
Given a dataset for multi-label classification with large spans of sequential text, an advanced deep neural network performs better than machine learning techniques.

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

For the Multinomial Naive Bayes model the classification report is the following:
 
![alt text](https://github.com/Judhajit-Roy/Multi-label-Classification-of-Research-Articles/blob/main/Images/nb%20result.JPG)


The classification report provides several metrics to analyze the performance of the model like precision, recall, F1-score and support. 
Since, it is a Multi-label classification problem the main evaluation metric to compare the models is Macro-F1. Macro-F1 is the average of the individual F1 scores of every class. Therefore, it gives equal importance to each class.
Thus, the baseline Macro-f1 score is **0.20**.

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

# Results Summary and Takeways

Results show that the BiLSTM performs considerably better with a Macro-F1 score of 0.72. In the machine learning models Linear SVC is the best model with a Macro-F1 score 0.63. The BiLSTM was trained for 20 epochs. Around the epochs 9-10 it starts to get overfit on the training data based on the accuracy and loss plot.

So the key takeaways are that with a Advanced Deep learning approach like BiLSTM we obtain a higher Macro-F1 score than machine learning models. The drawback is that BiLSTM takes a lot of time to train whereas Linear SVC is trained within a minute and provides decent results. 
In order to improve the results further more BiLSTM stacked on top another to obtain a higher score. Also, in this experiment only 200 dimensional glove embedding was used. Results for 300d can be explored as well as other approaches using Transformers, BERT and BioBert are some other advanced techniques to look at.

# Steps to run code:

The code is in a Ipython notebook format.
Dataset is provided.












