import numpy as np
class myLogisticRegression:
    def __init__(self,n_features,lr=0.01,epochs=10):
        #n features
        self.n = n_features

        #learning rate
        self.lr = lr

        #epochs
        self.epochs = epochs

        #weights
        self.W = np.random.randn(self.n,1)
        self.b = 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-1*x))
    
    def accuracy(self,y_pred,y):

        y_threshold = y_pred>=0.5
        total = (y_threshold==y)

        return np.sum(total)/len(y)

    def cross_entropy_loss(self,y_pred,y):
        loss = -1*np.mean((y*np.log(y_pred)) + ((1-y)*np.log(1-y_pred)))
        return loss

    def my_fit(self,X,y):
        
        # nxm
        # mx1
        X = np.array(X)
        y = np.array(y)

        _,m = X.shape

        for epoch in range(self.epochs):

            #ypred
            y_pred = np.dot(X.T,self.W) + self.b
            y_pred = self.sigmoid(y_pred)

            #loss
            if epoch%10==0:
                loss = self.cross_entropy_loss(y_pred,y)
                accuracy = self.accuracy(y_pred,y)
                print(f"The loss for epoch {epoch} is {loss} and accuracy is {accuracy}")

            #gradient descent
            #nx1
            dw = (1/(2*m))*(np.dot(X,y_pred-y))
            db = (1/(2*m))*(np.sum(y_pred-y))

            #update
            self.W = self.W - self.lr*dw
            self.b = self.b - self.lr*db
    
    def test(self,X,y):

        #ypred
        y_pred = np.dot(X.T,self.W) + self.b
        y_pred = self.sigmoid(y_pred)

        print(f"Accuracy is {self.accuracy(y_pred,y)}")





