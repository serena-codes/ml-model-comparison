# Daten laden mit Scikit-learn
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
data = load_diabetes()
X, y = data.data, data.target

# Daten skalieren
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test-Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# random_state=42 – weil 42 bekanntlich die Antwort auf alles ist 😉

# PyTorch-Modell bauen
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = RegressionModel()

# Training vorbereiten
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training starten
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Vorhersage & Bewertung
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse:.2f}")

# Echte Werte vs. Vorhersagen
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.6, label='Vorhersagen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfekte Vorhersage (y = x)')
plt.xlabel('Echte Werte')
plt.ylabel('Vorhergesagte Werte')
plt.title('Diabetes-Vorhersage: Modell vs. Realität')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()