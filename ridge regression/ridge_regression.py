# Importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ridge Regression

class RidgeRegression() :
	
	def __init__( self, learning_rate, iterations, l2_penality ) :
		
		self.learning_rate = learning_rate		
		self.iterations = iterations		
		self.l2_penality = l2_penality
		
	# Function for model training			
	def fit( self, X, Y ) :
		
		# no_of_training_examples, no_of_features		
		self.m, self.n = X.shape
		
		# weight initialization		
		self.W = np.zeros( self.n )
		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		# gradient descent learning
				
		for i in range( self.iterations ) :			
			self.update_weights()			
		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :		
		Y_pred = self.predict( self.X )
		
		# calculate gradients	
		dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +			
			( 2 * self.l2_penality * self.W ) ) / self.m	
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights	
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db		
		return self
	
	# Hypothetical function h( x )
	def predict( self, X ) :	
		return X.dot( self.W ) + self.b
	

def make_linear(w=1, b=0.8, size=100, noise=1.0): #w = weight b = bias
    x = np.random.rand(size).reshape(size,1)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = (y + noise).reshape(size,1)
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='r', label=f'y = {w}*x + {b}')
    plt.scatter(x, yy, label='data')
    plt.legend(fontsize=20)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy

np.random.seed(42)
X = np.random.rand(100,1)
y = 5*((X)**(2)) + np.random.rand(100,1)
