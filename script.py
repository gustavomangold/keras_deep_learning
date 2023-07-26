import pandas as pd
import numpy as np
import keras
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential as seq
from keras.layers import Dense as den

data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
data_columns = data.columns

#define the predictors and the target column
predictors = data[data_columns[data_columns != 'Strength']] #here is the target
target = data['Strength']

#normalize the predictors
predictors_normalized = (predictors - predictors.mean())/ predictors.std()

#this defines the predictors (X) and target (y) for training and testing
X_train, X_test, y_train, y_test =  train_test_split(predictors, target, test_size = 0.3, random_state = 42)

#predictors_stdr = (predictors - predictors.mean()) / predictors.std() #here we standardize the predictors

n_columns = predictors.shape[1]
n_classes = y_test.shape[0]

#with this funciton we define the model and add the layers to it
#with posterior use of the function compile, which makes the regression possible
def regression_model():
    model = seq()
    model.add(den(10, activation = 'relu', input_shape = (n_columns,)))
    model.add(den(10, activation = 'relu'))
    model.add(den(10, activation = 'relu'))
    model.add(den(n_classes))
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    return model
  
#this builds the model
model = regression_model()

mse_list = []
#this trains and predicts
for i in range(1, 50):
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, batch_size = 200, verbose = 1)
    y_pred = model.predict(X_test)
    mserr = mean_squared_error(y_test, y_pred[:,0])
    mse_list.append(mserr)  
 
#and this evaluates it
mean, std_dev = np.mean(mse_list), np.std(mse_list)
print(mean, std_dev)
#must complete printing the model graph or smth
