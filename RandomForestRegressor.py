from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Daten laden und skalieren
data = load_diabetes()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modell erstellen und trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen
y_pred = model.predict(X_test)

# MSE berechnen
mse = mean_squared_error(y_test, y_pred)
print(f"RandomForestRegressor Test MSE: {mse:.2f}")

# Visualisierung
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6, label='RandomForest Vorhersagen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfekte Vorhersage (y = x)')
plt.xlabel('Echte Werte')
plt.ylabel('Vorhergesagte Werte')
plt.title('RandomForestRegressor: Modell vs. Realität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()