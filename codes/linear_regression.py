"""
The main point of this practice is using 
the linear model function "LinearRegression().fit(x, y)"
to fit a linear model
"""
from sklearn import datasets, linear_model
import numpy as np 
import matplotlib.pyplot as plt   
data = np.array([[152,51],[156,53],[160,54],[164,55],
                 [168,57],[172,60],[176,62],[180,65],
                 [184,69],[188,72]]) # input a dataset

print(data.shape) # print the dimension of the dataset

# TODO 1. seperate x and y from the dataset
x, y = data[:,0].reshape(-1,1), data[:,1]
# TODO 2. find a linear model on x and y, then save and regress it
reg = linear_model.LinearRegression().fit(x, y)
# TODO 3. plot the model
plt.plot(x, reg.predict(x), color='blue')
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')
plt.show()

# use the model to predict
print ("Standard weight for person with 163 is %.2f"% reg.predict([[163]]))