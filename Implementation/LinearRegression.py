import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self,learning_rate = 0.01,n_iter = 10000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.bias = None
        self.weights = None

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame):
        samples,features = X.shape
        #Step 1
        self.bias = 0
        self.weights = np.zeros(shape=features)

    
        for _ in range(self.n_iter):

            #Step 2
            Y_pred = self.bias + np.dot(X,self.weights)

            #Step 3
            db = (1/samples)*np.sum(Y_pred-Y)
            dw = (1/samples)*np.dot(X.T,Y_pred-Y)

            #Step 4
            self.bias = self.bias - self.lr*db
            self.weights = self.weights - self.lr*dw
    def predict(self,X):
        return self.bias + np.dot(X,self.weights)
    
def main():
    X = np.array([[1],[2],[3],[4],[5],[6]])
    Y = np.array([3,6,9,12,15,18])
    model = LinearRegression()

    model.fit(X,Y)
    print(model.predict([[10],[20],[30],[40],[50]]))
if(__name__ == '__main__'):
    main()
    