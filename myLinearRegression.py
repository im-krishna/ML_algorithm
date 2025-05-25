import numpy as np
class myLinearRegression:
    def __init__(self,n_features,epochs = 10,lr = 0.001):

        #hyper-params
        self.lr = lr
        self.epochs = epochs 
        self.n = n_features

        #dimension
        #X -> nxm
        #y -> mx1
        #weights -> nx1
        #intialise weights as a numpy array of 
        # random values taken out of standard distribution 
        self.W = np.random.randn(self.n,1)
        self.b = 0


    def my_fit(self,X,y):
        
        X = np.array(X)
        y = np.array(y)

        m,_ = X.shape
        
        prev_loss = 0
        for epoch in range(self.epochs):
            #model output
            y_pred = np.dot(X.T,self.W) + self.b
            #mx1

            #loss calulation
            loss = np.mean((y_pred-y)**2)
            print(f"Loss for epoch {epoch}: {loss}")

            #gradient descent
            dw = (1/(2*m))*np.dot(X,(y_pred-y))
            db = (1/(2*m))*np.sum(y_pred-y)

            #update
            self.W = self.W-self.lr*dw
            self.b = self.b-self.lr*db


            #termination condition
            if abs(loss-prev_loss)<10e-8:
                break
            prev_loss = loss
    
    def test(self,X):

        X= np.array(X)
        return np.dot(X.T,self.W) + self.b
        





