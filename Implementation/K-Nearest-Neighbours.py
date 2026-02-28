import numpy as np
import pandas as pd

class KNNClassifier:

    def __init__(self,k):
        self.k = k

    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
        
    def _distance(self,x,y):
        return np.sqrt(np.sum(x-y)**2)
    
    def _predict_one(self,X):
        for i in self.X_train:
            distances = [self._distance(X,X_train) for X_train in self.X_train]

        knn = np.argsort(distances)[:self.k]
        knn_classes = [self.Y_train[i] for i in knn]

        majority = np.bincount(knn_classes)
        return np.argmax(majority)
    

    def predict(self,X):
        Y_pred = [self._predict_one(x) for x in X]
        return Y_pred
    
def main():
    X_train = np.array([
        [1,2],
        [2,3],
        [3,3],
        [6,5],
        [7,7]
    ])
    Y_train= np.array([0,0,0,1,1])

    X_test = np.array([
        [2,2],
        [6,6]
    ])

    model = KNNClassifier(k=3)
    model.fit(X_train,Y_train)

    print(model.predict(X_test))


if(__name__ == '__main__'):
    main()