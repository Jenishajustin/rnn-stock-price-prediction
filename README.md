# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset.

<b>Dataset: </b>The dataset consists of two CSV files:

<b>Trainset.csv:</b> This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock.
<br>

<img src="https://github.com/Jenishajustin/rnn-stock-price-prediction/assets/119405070/ff74d152-da01-46a3-8d0e-155003ee7c90" height=200 width=700>

<b>Testset.csv:</b> This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock.
<br>

<img src="https://github.com/Jenishajustin/rnn-stock-price-prediction/assets/119405070/ccc6ef9d-9d46-486e-b1b5-e98e74dd8b27" height=200 width=700>


The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data.

## Design Steps

### Step 1:
Import the necessary libraries

### Step 2:
Read and preprocess training data, including scaling and sequence creation.
### Step 3:
Create a Sequential model with SimpleRNN and Dense layers.

### Step 4:
Compile the created model with loss as MSE and optimizer as Adam.

### Step 5:
Train the model on the prepared training data.
### Step 6:
Preprocess test data, predict using the trained model, and visualize the results.

## Program
#### Name: J.JENISHA
#### Register Number: 212222230056

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense

# Train dataset
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values  #[:,1:2] -> : = for row and 1:2 = for col
train_set
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []

for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1
model = Sequential([
    SimpleRNN(50, input_shape = (length, n_features)),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse')
print("J.JENISHA \n212222230056")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
print("J.JENISHA \n212222230056")
model.summary()

# Test dataset
dataset_test = pd.read_csv('testset.csv')
dataset_test
test_set = dataset_test.iloc[:,1:2].values
test_set
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: J.JENISHA \nRegister Number: 212222230056    ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img src="https://github.com/Jenishajustin/rnn-stock-price-prediction/assets/119405070/87bcd2ec-a03a-416b-8971-1e230a0315c1" height=400 width=700>


### Mean Square Error
<img src="https://github.com/Jenishajustin/rnn-stock-price-prediction/assets/119405070/fee89bed-eedf-4de2-a468-8465d65a20d6" height=400 width=700>

<img src="https://github.com/Jenishajustin/rnn-stock-price-prediction/assets/119405070/c8b2ce85-ddb5-46aa-8bfc-7fab83c5343b" height=400 width=700>


## Result
Thus a Recurrent Neural Network model for stock price prediction is created and executed successfully.
