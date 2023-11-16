import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Food-Truck-LineReg.csv')
X = df.iloc[: , :-1]
y = df.iloc[: , -1]
# Scatter plot
plt.scatter(X, y)
plt.title('Scatter Plot of X vs Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Build a correlation matrix
correlation_matrix = df.corr()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,train_size=0.7, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Compute regression parameters
intercept = model.intercept_
slope = model.coef_[0]

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate SSE, SSR, SST
SSE = np.sum((y_pred - y_test) ** 2)
SSR = np.sum((y_pred - np.mean(y_test)) ** 2)
SST = np.sum((y_test - np.mean(y_test)) ** 2)

# Print results
#print(f'Intercept: {intercept}')
#print(f'Slope: {slope}')
print(f'Cost: {mse}')
print(f'R-squared (R2): {r2}')
print(f'Sum of Squared Errors (SSE): {SSE}')
print(f'Sum of Squared Residuals (SSR): {SSR}')
print(f'Total Sum of Squares (SST): {SST}')

# Display correlation matrix
print('\nCorrelation Matrix:')
print(correlation_matrix)
