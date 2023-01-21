#random forest regression and  the concept of ensemble learning  is when you take multiple algorithms or same algorithm multiple times to make it more powerful. 
#Step 1 Pick at random k data points from the training set
#Step 2 build a decision tree associated to the K data points
#Step 3 choode the number Ntree of trees you want to build and repeat step 1 and 2
#Step 4 for new data poits, make each one of your Ntrees predict the value of Y for the data point in the question and assign the new data points the average across all the predicted values of Y
import numpy 
import matplotlib.pyplot as graph
import pandas 
dataset = pandas.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#n_estimators = Number of trees that we want to create
regressor.fit(x,y)
regressor.predict([[6.5]])
x_grid = numpy.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid),1))
graph.scatter(x,y, color = 'red')
graph.plot(x_grid, regressor.predict(x_grid), color = 'blue')
graph.title('(Random Forest Regression)')
graph.xlabel('Position level')
graph.ylabel('Salary')
graph.show()
