import pickle
import numpy as np
import math


class AdaBoostClassifier(object):
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.__weak_classifier = weak_classifier
        self.__n_weakers_limit = n_weakers_limit
        self.__a_list = []
        self.__h_list = []

        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        n_samples, n_features = X.shape
        w = []
        for i in range(n_samples):
            w.append(1 / n_samples)

        for m in range(self.__n_weakers_limit):
            h = self.__weak_classifier(max_depth=1, random_state=6, max_features=165600)
            h.fit(X, y, sample_weight=w)
            predict = h.predict(X)
            # print(type(predict))
            # print(predict.shape)
            result = (predict != y) + 0
            error = (w * result).sum()
            if error > 0.5:
                break
            else:
                a = math.log((1-error) / error) / 2
            w = w * np.exp(-y * a * predict)
            w = w / w.sum()
            self.__h_list.append(h)
            self.__a_list.append(a)


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of different samples, which shape should be (n_samples,1).
        '''
        test1 = np.array([h.predict(X) for h in self.__h_list])
        test2 = np.array(self.__a_list)
        # print(test1.shape)
        # print(test2.shape)
        # print(test1)
        # print(test2)
        # print(type(test1))
        # print(type(test2))
        test3 = test2.reshape(-1, 1)
        return test1 * test3

    def predict(self, X, threshold=0):

        '''Predict the categories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        result_list = self.predict_scores(X).sum(axis=0)
        for i in range(len(result_list)):
            if result_list[i] > 0:
                result_list[i] = 1
            else:
                result_list[i] = -1
        return result_list

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
