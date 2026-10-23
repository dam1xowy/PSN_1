from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tabela import save_t

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
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy * 100:.2f}%\n")

report_dict = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)

df_report = pd.DataFrame(report_dict).transpose().round(2)

save_t(df_report, 'tabela_wynikow.png')