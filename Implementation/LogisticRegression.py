import numpy as np
import pandas as pd

class LogisticRegressor:
    def __init__(self,learning_rate = 0.01,n_iter =10000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
    
    def _sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def fit(self,X,Y):
        samples,features = X.shape

        self.bias = 0
        self.weights = np.zeros(features)

        for _ in range(self.n_iter):
            z = self.bias + np.dot(X,self.weights)
            Y_pred = self._sigmoid(z)
            db = 1/samples * (Y_pred-Y)
            dw = 1/samples * np.dot(X.T,(Y_pred-Y))

            self.bias -= db
            self.weights -=dw
    
    def get_probability(self,X):
        z = self.bias  + np.dot(X,self.weights)
        prob = self._sigmoid(z)
        return prob
    
    def predict(self,X,threshold = 0.5):
        prob = self.get_probability(X)
        
        Y_pred = (prob>=threshold).astype(int)
        return Y_pred
    
def main():
    X = np.array([[1,2],[2,3],[3,4],[4,5]])
    Y = np.array([0,0,1,1])
    
    model = LogisticRegressor()
    model.fit(X,Y)
    
    T = np.array([[1,4],[2,4],[4,6],[14,12]])

    print(model.predict(T))

if(__name__ == '__main__'):
    main()