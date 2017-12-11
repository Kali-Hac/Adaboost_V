# -*- coding: utf-8 -*-
# Adaboost类的构建
import cPickle as pickle
from sklearn.metrics import classification_report
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit=5):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.base_classifier = weak_classifier
        self.n_limit = n_weakers_limit
        # 存储多个weakers分类器的模型和各个分类器所占的重要程度权重a
        self.weakers_models = []
        self.weaker_a = np.ones(self.n_limit)
        self.pred_result = []
        pass
    def is_good_enough(self, y_true):
        '''Optional'''
        target_names = ['Positive', 'Negative']
        labels = [1, -1]
        return classification_report(y_true, self.pred_result, labels=labels, target_names=target_names, digits=3)

    def fit(self, X_train, y_train):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        x_size = X_train.shape[0]
        w = [1.0 / X_train.shape[0] for t in range(x_size)]
        w = np.array(w)
        cnt = 0
        while cnt < self.n_limit:
            model = self.base_classifier.fit(X_train, y_train, sample_weight=w)
            y_predict = model.predict(X_train)
            error = 0.0
            for i in range(x_size):
                if not y_predict[i] == y_train[i]:
                    error += w[i]
            self.weakers_models.append(model)
            # sklearn的决策树对简单数据的分类准确率可以达到100% 本实验的数据比较简单
            if error != 0.0:
                self.weaker_a[cnt] = 0.5 * math.log((1-error)/error)
                # print self.weaker_a[cnt]
                w *= np.exp(-self.weaker_a[cnt] * y_train * y_predict)
                # print w
                w = w / w.sum()
            cnt += 1


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        all_predict_y = []
        scores = np.zeros(X.shape[0])
        for base_classifier in self.weakers_models:
            y_predict = base_classifier.predict(X)
            all_predict_y.append(y_predict)
        for i in range(len(all_predict_y)):
            scores += all_predict_y[i] * self.weaker_a[i]
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        scores = self.predict_scores(X)
        self.pred_result = np.sign(scores)
        return self.pred_result

    @staticmethod
    def save(adaboost_model, filename):
        with open(filename, "wb") as f:
            pickle.dump(adaboost_model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
