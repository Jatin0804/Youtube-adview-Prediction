# Step 1: Import the datasets and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
df = pd.read_csv('train.csv')

# Step 2: Check shape and datatype
print(df.shape)
print(df.dtypes)

# Step 3: Visualize the dataset
# Heatmap to check correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plot distributions
df.hist(bins=30, figsize=(15, 10))
plt.show()

# Step 4: Clean the dataset
# Handle missing values
df = df.dropna()

# Convert 'duration' to numerical (seconds)
def convert_duration(duration):
    h, m, s = 0, 0, 0
    duration = duration.replace('PT', '')
    if 'H' in duration:
        h, duration = duration.split('H')
    if 'M' in duration:
        m, duration = duration.split('M')
    if 'S' in duration:
        s = duration.split('S')[0]
    return int(h) * 3600 + int(m) * 60 + int(s)

df['duration'] = df['duration'].apply(convert_duration)

# Convert 'published' to datetime and extract useful features
df['published'] = pd.to_datetime(df['published'])
df['publish_year'] = df['published'].dt.year
df['publish_month'] = df['published'].dt.month
df['publish_day'] = df['published'].dt.day
df = df.drop('published', axis=1)

# Encode 'category' using LabelEncoder
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])

# Step 5: Normalize the data and split into training, validation, and test sets
X = df.drop(['vidid', 'adview'], axis=1)
y = df['adview']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train models and calculate errors
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print("SVR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_svr)))

# Decision Tree Regressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dtr)))

# Random Forest Regressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rfr)))

# Step 7: Build and train an Artificial Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Step 8: Evaluate the ANN
y_pred_ann = model.predict(X_test)
print("ANN RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ann)))

# Step 9: Save the best model
# Assuming Random Forest performed the best
import joblib
joblib.dump(rfr, 'best_model.pkl')

# Step 10: Predict on the test set
y_pred_final = rfr.predict(X_test)
print("Final RMSE on test set:", np.sqrt(mean_squared_error(y_test, y_pred_final)))
