import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Daten laden
data = load_diabetes()
X, y = data.data, data.target

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersagen
y_pred = model.predict(X_test)

# MSE berechnen
mse = mean_squared_error(y_test, y_pred)
print(f"LinearRegression Test MSE: {mse:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='LinearRegression Vorhersagen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfekte Vorhersage (y = x)')
plt.xlabel('Echte Werte')
plt.ylabel('Vorhergesagte Werte')
plt.title('LinearRegression: Modell vs. Realität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()