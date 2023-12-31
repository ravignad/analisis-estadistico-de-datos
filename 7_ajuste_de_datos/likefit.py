import numpy as np
from scipy.optimize import minimize


class LeastSquares:  


    def __init__(self, x, y, ysigma, model_function):  
        self.x = x
        self.y = y
        self.ysigma = ysigma
        self.model_function = model_function
            

    def cost_function(self, theta): 
        mu = self.model_function(self.x, theta)        
        residuals = (self.y - mu) / self.ysigma
        cost = np.sum(residuals**2)
        return cost


    def fit(self, seed):
        fit_result = minimize(self.cost_function, x0=seed)
        return fit_result
        


# class Poisson(object):  

#    def __call__(self, theta) :  
#        mu = self.fit_model(theta)
#        cost_array = 2 * (mu - self.k) - 2 * self.k * np.log(mu / self.k)
#        return cost_array.sum()




    

