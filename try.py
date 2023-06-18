import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Lambda, Input
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

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

# Keep only two columns for the analysis: Date, Positivity Rate
df_gr = df_gr[["Date", "Positivity Rate"]]
# --------------------------------------------------------------------------------------------------------------------


# --- Train-Test Split ----------------------------------------------------------------------------------------------
# Divide the data into training and testing sets, before and after 1/1/2021
train_set = df_gr[df_gr["Date"] < "2021-01-01"].reset_index(drop=True)
train_dates = pd.to_datetime(train_set["Date"])
test_set = df_gr[df_gr["Date"] >= "2021-01-01"].reset_index(drop=True)
test_dates = pd.to_datetime(test_set["Date"])
# --------------------------------------------------------------------------------------------------------------------


# --- Train/Test Plot ------------------------------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(train_set["Date"], train_set["Positivity Rate"], label="Train Set", color="blue")
plt.plot(test_set["Date"], test_set["Positivity Rate"], label="Test Set", color="orange")
plt.axvline(x="2021-01-01", color="red", linestyle="--")
plt.legend()
plt.xticks(np.arange(0, len(df_gr), 7), rotation=45)
plt.savefig("positivity_rate_train_test.png")
plt.close()
# --------------------------------------------------------------------------------------------------------------------


# --- Real Positivity Rate Dataframe ---------------------------------------------------------------------------------
original = df_gr[['Date', 'Positivity Rate']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2021-01-01']
# --------------------------------------------------------------------------------------------------------------------


# --- Data Preprocessing ---------------------------------------------------------------------------------------------
# Scale the data
scaler = MinMaxScaler()

# Fit and transform only the positivity rate column
scaler.fit(train_set[["Positivity Rate"]])
scaled_train = scaler.transform(train_set[["Positivity Rate"]])
scaled_test = scaler.transform(test_set[["Positivity Rate"]])
# --------------------------------------------------------------------------------------------------------------------

# Callback for early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

# --- SECOND METHOD --------------------------------------------------------------------------------------------------
n_input = 5    # Number of past days we want to use to predict the future
n_features = 1  # Number of features we want to predict

training_sequence = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, sampling_rate=1, batch_size=16)
validation_sequence = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, sampling_rate=1, stride=5, batch_size=16)

# --- RNN Model ---
RNNmodel = Sequential()
RNNmodel.add(LSTM(128, activation='leaky_relu', input_shape=(n_input, n_features), return_sequences=True))
RNNmodel.add(LSTM(64, activation='leaky_relu', return_sequences=True))
RNNmodel.add(LSTM(32, activation='leaky_relu', return_sequences=False))
RNNmodel.add(Dropout(0.2))
RNNmodel.add(Dense(1))

# Compile the model
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
RNNmodel.compile(optimizer=sgd, loss='mean_squared_error')

print("\nTraining RNN")
        
history = RNNmodel.fit(training_sequence, epochs=20, validation_data=validation_sequence, verbose=1, callbacks=[early_stopping])
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig("RNN2_lossPerEpoch.png")
plt.close()

# Predict the positivity rate 3 days ahead
predictions = RNNmodel.predict(scaled_test)
original['RNN'] = predictions

# --- SVM Model ---
SVMmodel = svm.LinearSVR(C=1, epsilon=0.001, loss='squared_epsilon_insensitive')

print("\nTraining SVM")

SVMmodel.fit(scaled_train, scaled_train)

predictions = SVMmodel.predict(scaled_test)
original['SVM'] = predictions
# --------------------------------------------------------------------------------------------------------------------


# --- Save the predictions to a csv file ----------------------------------------------------------------------------
original = original.reset_index(drop=True)
original.to_csv("predictions.csv")
# -------------------------------------------------------------------------------------------------------------------


# --- Plot the results ----------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(original['Date'], original['Positivity Rate'], label='Actual Positivity Rate', color='black')

# First method
plt.plot(original['Date'], original['RNN'], label='RNN Predictions', color='red')
plt.plot(original['Date'], original['SVM'], label='SVM Predictions', color='green')

plt.title('Positivity Rate Predictions')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig("predictions.png")
# --------------------------------------------------------------------------------------------------------------------