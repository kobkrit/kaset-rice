# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
r = requests.get('http://kaset-prediction.iapp.co.th/api/breed')
print(r.text)

exit()

#RNN Layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
regressor = Sequential()


# Reading CSV file from train set
training_set = pd.read_csv('train-rice.csv')
training_set.head()

#Selecting the second column [for prediction]
training_set = training_set.iloc[0:417,1:2]
training_set.head()

# Converting into 2D array
training_set = training_set.values
raw_training_set = training_set

# Scaling of Data [Normalization]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Length of the Data set
len(training_set)

X_train = training_set[0:-1,:]
Y_train = training_set[1:,:]
print ("X_train")
print (X_train)
print (X_train.shape)

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train,(416, 1, 1))
print(X_train)

#-------------------------Need to be have Keras and TensorFlow backend--------------------------- 

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 8, activation = 'sigmoid', input_shape = (None, 1)))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the Recurrent Neural Network [epoches is a kindoff number of iteration]
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 400)

# Reading CSV file from test set
test_set = pd.read_csv('test-rice.csv')
test_set.head()

#selecting the second column from test data 
real_rice_price = test_set.iloc[:,1:2]         

print("real_rice_price")
print(real_rice_price)
# Coverting into 2D array
real_rice_price = real_rice_price.values      

#getting the predicted BTC value of the first week of Dec 2017  
inputs = real_rice_price			
inputs = sc.transform(inputs)
inputs = inputs[0:1]

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]

predicted_rice_price = []

for i in range(7):
    # print("before inputs")
    # print(i, inputs)

    inputs = np.reshape(inputs, (1, 1, 1))

    print("after inputs")
    print(i, inputs)

    result = regressor.predict(inputs)

    print("result")
    print(i, result)
    
    predicted_rice_price.append(result[0])
    inputs = result[0]

print("predicted")
print(predicted_rice_price)

predicted_rice_price = sc.inverse_transform(predicted_rice_price)
print("reverse predicted")
print(predicted_rice_price)

#Graphs for predicted values
plt.plot(np.concatenate((raw_training_set, real_rice_price[1:]), axis=0), color = 'red', label = 'Real Rice Value')
plt.plot(np.concatenate((raw_training_set, predicted_rice_price), axis=0), color = 'blue', label = 'Predicted Rice Value')
plt.title('Rice Value Prediction')
plt.xlabel('Months')
plt.ylabel('Rice Value')
plt.legend()
plt.show()

