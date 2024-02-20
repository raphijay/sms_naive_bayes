import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Naive():
 
 
    def __init__(self):
        self.actual = None
        self.classes = []
        self.predictions = []
        self.count = []

    def fit(self, x, y):
        self.actual = y
        self.classes = y.unique()
        for c in self.classes:
            indices = [i for i, value in enumerate(y) if value == c]
            words = []
            for i in indices:
                message = x.iloc[i]
                trimmessage = re.findall(r'\b\w\w+\b', message.lower())
                words += trimmessage
            count = Counter(words)
            # Probability is calculated based on unigram probability and add one smoothing
            probs = {key: (value + 1) / (sum(count.values()) + len(count)) for key, value in count.items()}
            self.predictions.append(probs)
            self.count.append(count)
        return self
    
    def predict(self, x):
        predictions = []
        for message in x:
            trimmessage = re.findall(r'\b\w\w+\b', message.lower())
            #print(trimmessage)
            probs = {}
            for c in self.classes:
                counts = self.actual.value_counts()
                prob = counts[c] / len(self.actual)
                #break
                for word in trimmessage:
                    if word in self.predictions[c]:
                        prob *= self.predictions[c][word]
                    else: # word not found in the wordcount
                        prob *= 1 / (sum(self.count[c].values()) + len(self.count[c]))
                probs[c] = prob
            # Predicts whether ham (0) or spam)
            # Boolean statement converts into an integer.
            predictions.append(int(probs[1] > probs[0]))

        return predictions
                

def main():

    # Reading the dataset
    data = pd.read_csv('data/SMSSpamCollection.txt', sep = '\t', names = ['ham/spam', 'message'])
    # Replacing ham and spam with 0 and 1 respectively
    data['ham/spam'] = data['ham/spam'].replace(to_replace=['ham', 'spam'], value=[0, 1])
    X_train, X_test, Y_train, Y_test = train_test_split(data['message'], data['ham/spam'], test_size=0.20, random_state = 25)

    # Statistics
    # Size of the dataframe
    print(len(data))
    # Total count of the dataframe
    print(data['ham/spam'].value_counts())
    

    # Fitting the Naive Bayes model
    nb = Naive().fit(X_train, Y_train)

    # Train dataset evaluation
    #y_pred = nb.predict(X_train)
    #accuracy = [Y_train == y_pred]
    #print("Training Accuracy:", sum(Y_train == y_pred) / len(y_pred))


    # Test dataset evaluation
    y_pred = nb.predict(X_test)
    accuracy = [Y_test == y_pred]
    #print("Test Accuracy", sum(Y_test == y_pred) / len(y_pred))

    # Confusion Matrix
    print("Confusion Matrix Format:")
    # True Positive
    tp = sum((y_pred[i] == 1) and (Y_test.iloc[i] == 1) for i in range(len(y_pred)))
    # False Positive
    fp = sum((y_pred[i] == 1) and (Y_test.iloc[i] == 0) for i in range(len(y_pred)))
    # True Negative
    tn = sum((y_pred[i] == 0) and (Y_test.iloc[i] == 0) for i in range(len(y_pred)))
    # False Negative
    fn = sum((y_pred[i] == 0) and (Y_test.iloc[i] == 1) for i in range(len(y_pred)))
    print(" TP  |  FP")
    print("------------")
    print(" FN  |  TN");
    print("Confusion Matrix Evaluation:")
    print("{:^5d} | {:^5d}".format(tp, fp))
    print("------------")
    print("{:^5d} | {:^5d}".format(fn, tn))
    print("----------------------------")
    # Evaluation
    print("True Positive:", tp)
    print("False Positive:", fp)
    print("True Negative:", tn)
    print("False Negative:", fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    


if __name__ == '__main__':
    main()