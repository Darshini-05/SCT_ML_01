import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/P.DARSHINI/Downloads/synthetic_house_prices.csv")



print("Here's what the data looks like:")
print(df.head())


X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("Mean Squared Error:", round(mse, 2))
print("R² Score (Accuracy):", round(r2, 4))


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("📈 Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
