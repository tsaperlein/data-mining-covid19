import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Remove warnings from the console
pd.options.mode.chained_assignment = None

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load the CSV file
df = pd.read_csv("modified_dataframe.csv")
# Keep only the data for Greece
df_gr = df[df["Entity"] == "Greece"]


# --- Positivity Rate -----------------------------------------------------------------------------------------------
# Calculate Positivity Rate 
df_gr["Positivity Rate"] = df_gr["Cases"].diff() / df_gr["Daily tests"]
# For the first day, is Cases/Daily tests
df_gr["Positivity Rate"].iloc[0] = df_gr["Cases"].iloc[0] / df_gr["Daily tests"].iloc[0]
# --------------------------------------------------------------------------------------------------------------------


# --- Current/Future Sets --------------------------------------------------------------------------------------------
# Divide the data into training and testing sets, before and after 1/1/2021
current_set = df_gr[df_gr["Date"] < "2021-01-01"].reset_index(drop=True)
future_set = df_gr[df_gr["Date"] >= "2021-01-01"].reset_index(drop=True)
# --------------------------------------------------------------------------------------------------------------------


# --- Train/Test Plot ------------------------------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(current_set["Date"], current_set["Positivity Rate"], label="Current Set", color="blue")
plt.plot(future_set["Date"], future_set["Positivity Rate"], label="Future Set", color="orange")
plt.axvline(x="2021-01-01", color="red", linestyle="--")
plt.legend()
plt.xticks(np.arange(0, len(df_gr), 7), rotation=45)
plt.savefig("positivity_rate_train_test.png")
plt.close()
# --------------------------------------------------------------------------------------------------------------------

# --- Train/Test Split ----------------------------------------------------------------------------------------------
# Create a new column with the positivity rate 3 days ahead, so that we can use it as a target variable
current_set["Positivity Rate 3 days ahead"] = current_set["Positivity Rate"].shift(-3)

# Drop the last 3 rows, since we don't have the positivity rate 3 days ahead for them
current_set.dropna(subset=["Positivity Rate 3 days ahead"], inplace=True)

# Split the data into X (features) and y (target)
X = current_set[["Daily tests", "Cases"]]
y = current_set["Positivity Rate 3 days ahead"]

# # Scale the data
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y.values.reshape(-1, 1))

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
# --------------------------------------------------------------------------------------------------------------------


# ----------------------------------- Train the Models ---------------------------------------------------------------
# --- RNN Model ------------------------------------------------------------------------------------------------------
RNNmodel = Sequential()
RNNmodel.add(LSTM(units=10, input_shape=(trainX.shape[1], 1)))
RNNmodel.add(Dense(1))

RNNmodel.compile(optimizer="adam", loss="mean_squared_error")

RNNmodel.fit(trainX, trainY, epochs=20, batch_size=1)

y_pred = RNNmodel.predict(testX)
mse_rnn = mean_squared_error(testY, y_pred)
print("RNN-MSE: ", mse_rnn)
# --------------------------------------------------------------------------------------------------------------------


# --- SVM Model ------------------------------------------------------------------------------------------------------
SVMmodel = svm.SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)

SVMmodel.fit(trainX, trainY)

y_pred = SVMmodel.predict(testX)
mse_svm = mean_squared_error(testY, y_pred)
print("SVM-MSE: ", mse_svm)
# --------------------------------------------------------------------------------------------------------------------


# --- Predictions ----------------------------------------------------------------------------------------------------
# Create a new column with the positivity rate 3 days ahead, so that we can use it as a target variable
future_set["Positivity Rate 3 days ahead"] = future_set["Positivity Rate"].shift(-3)

future_set.dropna(subset=["Positivity Rate 3 days ahead"], inplace=True)

# Split the data into X (features) and y (target)
X = future_set[["Daily tests", "Cases"]]
y = future_set["Positivity Rate 3 days ahead"]

# # Scale the data
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y.values.reshape(-1, 1))

y_pred_svm = SVMmodel.predict(X)
y_pred_rnn = RNNmodel.predict(X)

# Visualize the predicted values alongside the actual values
plt.figure(figsize=(14, 5))
plt.plot(future_set["Date"], y, label="Actual", color="black")
plt.plot(future_set["Date"], y_pred_rnn, label="RNN Predictions", color="orange")
plt.plot(future_set["Date"], y_pred_svm, label="SVM Predictions", color="green")
plt.legend()
plt.xticks(np.arange(0, len(future_set), 7), rotation=45)
plt.savefig("predictions.png", bbox_inches="tight")
plt.show()