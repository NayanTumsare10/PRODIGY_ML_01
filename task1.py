import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("train.csv")

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = data[features]
y = data['SalePrice']

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nHouse Price Prediction")

area = float(input("Enter square footage: "))
bed = int(input("Enter number of bedrooms: "))
bath = int(input("Enter number of bathrooms: "))

new_data = pd.DataFrame([[area, bed, bath]], columns=features)

price = model.predict(new_data)

print("Estimated Price:", round(price[0], 2))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()