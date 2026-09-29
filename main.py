# ==============================================================================
# Google Colab Simple Machine Learning Workflow
# Task: Predict house prices based on square footage using Linear Regression
# ==============================================================================

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Data into a Pandas DataFrame
data = {
    'Square_Feet': [
        750,
        800,
        850,
        900,
        950,
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        1900,
        2000,
    ],
    'Price_USD': [
        150000,
        158000,
        168000,
        182000,
        191000,
        200000,
        218000,
        242000,
        258000,
        279000,
        295000,
        312000,
        335000,
        348000,
        372000,
        390000,
    ],
}
df = pd.DataFrame(data)

# 2. Separate Features (X) and Target Variable (y)
X = df[['Square_Feet']]  # 2D array/DataFrame expected by scikit-learn
y = df['Price_USD']  # 1D Series

# 3. Split Dataset into Training (80%) and Testing (20%) Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions on Test Data
y_pred = model.predict(X_test)

# 6. Evaluate Model Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Model Coefficient (Slope): ${model.coef_[0]:,.2f} per sq ft')
print(f'Model Intercept: ${model.intercept_:,.2f}')
print(f'R-squared Score: {r2:.4f}')
print(f'Mean Squared Error: {mse:,.2f}')

# 7. Visualize Results using Matplotlib
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(
    X_train, y_train, color='blue', alpha=0.7, label='Training Data'
)

# Plot testing data (actuals)
plt.scatter(
    X_test,
    y_test,
    color='green',
    s=100,
    marker='^',
    label='Actual Test Data',
)

# Plot testing data (predictions)
plt.scatter(
    X_test,
    y_pred,
    color='orange',
    s=100,
    marker='x',
    label='Predicted Test Data',
)

# Plot the regression line across all feature values
plt.plot(
    X,
    model.predict(X),
    color='red',
    linestyle='--',
    linewidth=2,
    label='Regression Line',
)

# Customize plot titles and labels
plt.title(
    'House Price Prediction using Linear Regression',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Square Footage (sq ft)', fontsize=12)
plt.ylabel('Price (USD $)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Display plot directly inside Google Colab
plt.show()