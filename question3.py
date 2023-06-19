import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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
plt.savefig("question3/positivity_rate_current_future.png", bbox_inches="tight")
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

# Scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)
# --------------------------------------------------------------------------------------------------------------------


# ----------------------------------- Train the Models ---------------------------------------------------------------
# --- RNN Model ------------------------------------------------------------------------------------------------------
print("\n--- Training the RNN Model ---")
RNNmodel = Sequential()
RNNmodel.add(LSTM(20, input_shape=(trainX.shape[1], 1)))
RNNmodel.add(Dense(1))

# Compile the model
RNNmodel.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
RNNmodel.fit(trainX, trainY, epochs=10, batch_size=1)

y_pred = RNNmodel.predict(testX)
mse_rnn = mean_squared_error(testY, y_pred)
print("\nRNN-MSE: ", mse_rnn)
# --------------------------------------------------------------------------------------------------------------------


# --- SVM Model ------------------------------------------------------------------------------------------------------
print("\n--- Training the SVM Model ---")
SVMmodel = svm.SVR(kernel="rbf", C=400, gamma='scale', epsilon=0.15)

# Fit the model
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

# Scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

y_pred_svm = SVMmodel.predict(X)
y_pred_rnn = RNNmodel.predict(X)

# Create a new dataframe with the actual values and the predicted values
predictions = pd.DataFrame({"Date": future_set["Date"], "Actual": y.flatten(), "RNN": y_pred_rnn.flatten(), "SVM": y_pred_svm.flatten()})
# Save the dataframe to a CSV file
predictions.to_csv("question3/predictions.csv", index=False)

# Print the MSE for the predictions
mse_rnn = mean_squared_error(y, y_pred_rnn)
mse_svm = mean_squared_error(y, y_pred_svm)
print("\nRNN-MSE (future): ", mse_rnn)
print("SVM-MSE (future): ", mse_svm)

# Visualize the predicted values alongside the actual values
plt.figure(figsize=(14, 5))
plt.plot(predictions["Date"], predictions["Actual"], label="Actual", color="black")
plt.plot(predictions["Date"], predictions["RNN"], label="RNN", color="orange")
plt.plot(predictions["Date"], predictions["SVM"], label="SVM", color="green")
plt.legend(loc=2)
plt.xticks(np.arange(0, len(future_set), 7), rotation=45)
plt.savefig("question3/predictions.png", bbox_inches="tight")
plt.close()
# --------------------------------------------------------------------------------------------------------------------

# --- Calculate all the evaluation metrics at once -------------------------------------------------------------------
# Create an empty DataFrame for metrics
metrics = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'R2'])

# Loop through the models
for model in ['RNN', 'SVM']:
    
    # Get the metrics
    mae = mean_absolute_error(predictions['Actual'], predictions[model])
    mse = mean_squared_error(predictions['Actual'], predictions[model])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(predictions['Actual'], predictions[model])
    r2 = r2_score(predictions['Actual'], predictions[model])
    
    # Append the metrics to the DataFrame
    metrics = pd.concat([metrics, pd.DataFrame({'Model': [model], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape], 'R2': [r2]})])
    
metrics = metrics.reset_index(drop=True)
# Save the metrics to a csv file
metrics.to_csv("question3/metrics.csv")
# --------------------------------------------------------------------------------------------------------------------