import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("salary_data.csv")

X = df[["CGPA", "Internships", "Projects", "Certifications"]]
y = df["Salary"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train improved model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# User input
cgpa = float(input("Enter CGPA: "))
internships = int(input("Enter internships: "))
projects = int(input("Enter projects: "))
certifications = int(input("Enter certifications: "))

user_data = scaler.transform([[cgpa, internships, projects, certifications]])
salary = model.predict(user_data)

print(f"Predicted Salary: {salary[0]:.2f} LPA")

# Feature importance
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")

# Visualization
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.tight_layout()
plt.show()
