import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
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


# --- FIRST METHOD ---------------------------------------------------------------------------------------------------
# --- Create the Training Data ----
trainX = []
trainY = []

n_future = 3    # Number of days we want to predict into the future
n_past = 20     # Number of past days we want to use to predict the future
n_features = 1  # Number of features (in this case, only the positivity rate)

for i in range(n_past, len(scaled_train) - n_future +1):
    trainX.append(scaled_train[i - n_past:i, 0])
    trainY.append(scaled_train[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# --- RNN Model (1) ---
RNN1model = Sequential()
RNN1model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], n_features), return_sequences=True))
RNN1model.add(Dropout(0.2))
RNN1model.add(Dense(1))

# Compile the model
sgd = optimizers.SGD(learning_rate=0.02)
RNN1model.compile(loss='mean_squared_error', optimizer=sgd)

# --- SVM Model (1) ---
SVM1model = svm.LinearSVR(C=1, epsilon=0.1, max_iter=10000, tol=0.001)

# Predict the positivity rate for 3 days after each day in the test set
n_future = len(test_set)
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

# --- For each model (RNN1, SVM1) train and predict the positivity rate ---
for model in [RNN1model, SVM1model]:
    
    # Fit the model
    if model == SVM1model:
        print("\nTraining SVM1")
        
        model.fit(trainX, trainY.ravel())
    else:
        print("\nTraining RNN1")
        
        history = model.fit(trainX, trainY, epochs=20, batch_size=16, verbose=1, validation_split=0.2)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.savefig("RNN1_lossPerEpoch.png")
        plt.close()
        
    forecast = model.predict(trainX[-n_future:])
    forecast_copies = np.repeat(forecast, n_features, axis=-1)
    forecast_copies = forecast_copies.reshape(-1, n_features)  # Reshape to 2D array
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Positivity Rate': y_pred_future[:len(forecast_dates)]})
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    
    if model == RNN1model:
        original['RNN_1'] = df_forecast['Positivity Rate'].values
    else:
        original['SVM_1'] = df_forecast['Positivity Rate'].values
# --------------------------------------------------------------------------------------------------------------------



# --- SECOND METHOD --------------------------------------------------------------------------------------------------
n_input = 20    # Number of past days we want to use to predict the future

# Create the generator, but we use 80% of the data for training and 20% for validation
training_sequence = TimeseriesGenerator(scaled_train[:int(len(scaled_train)*0.8)], scaled_train[:int(len(scaled_train)*0.8)], length=n_input, batch_size=16)
validation_sequence = TimeseriesGenerator(scaled_train[int(len(scaled_train)*0.8):], scaled_train[int(len(scaled_train)*0.8):], length=n_input, batch_size=16)

# --- RNN Model (2) ---
RNN2model = Sequential()
RNN2model.add(LSTM(64, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
RNN2model.add(Dropout(0.2))
RNN2model.add(Dense(1))

# Compile the model
sgd = optimizers.SGD(learning_rate=0.02)
RNN2model.compile(loss='mean_squared_error', optimizer=sgd)

print("\nTraining RNN2")
        
history = RNN2model.fit(training_sequence, epochs=20, validation_data=validation_sequence, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig("RNN2_lossPerEpoch.png")
plt.close()

# Generate predictions using the trained model
current_batch = scaled_train[-n_input:].reshape((1, n_input, n_features))
predictions = []

for i in range(len(test_dates)):
    current_pred = RNN2model.predict(current_batch)[0][0]
    predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
rnn_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
original['RNN_2'] = rnn_predictions

# --- SVM Model (2) ---
SVM2model = svm.LinearSVR(C=1, epsilon=0.01, loss='squared_epsilon_insensitive')

print("\nTraining SVM2")

SVM2model.fit(scaled_train[:, 0].reshape(-1, 1), scaled_train[:, 0])

# Take the last n_input values from the training set as the initial input for prediction
current_batch = scaled_train[-n_input:]

# Number of future values to predict
n_predictions = len(test_dates)
svm_predictions = []

# Fix this error: ValueError: X has 20 features, but LinearSVR is expecting 1 features as input.
for i in range(n_predictions):
    current_pred = SVM2model.predict(current_batch.reshape(-1, 1))[0]
    svm_predictions.append(current_pred)
    current_batch = np.append(current_batch[1:], [current_pred])

original['SVM_2'] = svm_predictions
# --------------------------------------------------------------------------------------------------------------------


# --- Save the predictions to a csv file ----------------------------------------------------------------------------
original = original.reset_index(drop=True)
original.to_csv("predictions.csv")
# -------------------------------------------------------------------------------------------------------------------


# --- Plot the results ----------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(original['Date'], original['Positivity Rate'], label='Actual Positivity Rate', color='black')

# First method
plt.plot(original['Date'], original['RNN_1'], label='RNN Predictions (1)', color='red')
plt.plot(original['Date'], original['SVM_1'], label='SVM Predictions (1)', color='green')

# Second method
plt.plot(original['Date'], original['RNN_2'], label='RNN Predictions (2)', color='orange')
plt.plot(original['Date'], original['SVM_2'], label='SVM Predictions (2)', color='pink')

plt.title('Positivity Rate Predictions')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig("predictions.png")
# --------------------------------------------------------------------------------------------------------------------


# --- Calculate all the evaluation metrics at once -------------------------------------------------------------------
# Create an empty DataFrame for metrics
metrics = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2'])

# Loop through the models
for model in ['RNN_1', 'SVM_1', 'RNN_2', 'SVM_2']:
    
    # Get the metrics
    mae = mean_absolute_error(original['Positivity Rate'], original[model])
    mse = mean_squared_error(original['Positivity Rate'], original[model])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((original['Positivity Rate'] - original[model]) / original['Positivity Rate'])) * 100
    r2 = r2_score(original['Positivity Rate'], original[model])
    
    # Append the metrics to the DataFrame
    metrics = pd.concat([metrics, pd.DataFrame({'Model': [model], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape], 'R2': [r2]})])
    
metrics = metrics.reset_index(drop=True)
# Save the metrics to a csv file
metrics.to_csv("metrics.csv")
# --------------------------------------------------------------------------------------------------------------------