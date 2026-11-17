import numpy as np
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, classification_report

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

mlp_xor = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    solver='lbfgs',
    max_iter=1000,
    random_state=47
)
mlp_xor.fit(X_xor, y_xor) 
y_xor_pred = mlp_xor.predict(X_xor) 

print("=== CZĘŚĆ 1: PROBLEM XOR ===")
for wejscie, cel, predykcja in zip(X_xor, y_xor, y_xor_pred):
    print(f"Wejście: {wejscie} | Oczekiwane: {cel} | Predykcja sieci: {predykcja}")
print(f"Dokładność XOR: {accuracy_score(y_xor, y_xor_pred) * 100:.2f}%\n")

data = load_breast_cancer()
X = data.data   
y = data.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, random_state=47)
mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print(f"Dokładność modelu: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
