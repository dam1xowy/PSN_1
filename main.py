import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(47)
np.random.seed(47)

FOLDER = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(FOLDER, 'wykresy_mlp_keras')
os.makedirs(PLOTS_DIR, exist_ok=True)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
y_train_f = y_train.astype(np.float32)
y_test_f = y_test.astype(np.float32)

class DiagnosticsCallback(callbacks.Callback):
    def __init__(self, X_tr, y_tr, X_te, y_te, target_mse=0.0055):
        super().__init__()
        self.X_tr = tf.constant(X_tr)
        self.y_tr = tf.constant(y_tr.reshape(-1, 1))
        self.X_te = tf.constant(X_te)
        self.y_te = y_te
        self.target_mse = target_mse

        self.history = {
            'mse_out_train': [],     
            'mse_hid_train': [],    
            'mse_out_sample': [],    
            'mse_hid_sample': [],    
            'class_err_train': [],   
            'class_err_test': [],    
            'W1_history': [],        
            'W2_history': [],        
            'lr_history': []        
        }

    def _compute_layer_mse(self, X_batch, y_batch):
        hidden_layer = self.model.get_layer('warstwa_ukryta')
        output_layer = self.model.get_layer('warstwa_wyjsciowa')

        with tf.GradientTape() as tape:
            h_out = hidden_layer(X_batch)
            tape.watch(h_out)
            y_pred = output_layer(h_out)
            mse_out = tf.reduce_mean(tf.square(y_pred - y_batch))

        grad_hidden = tape.gradient(mse_out, h_out)
        batch_len = tf.cast(tf.shape(X_batch)[0], tf.float32)
        mse_hid = tf.reduce_mean(tf.square(grad_hidden * batch_len))
        return float(mse_out.numpy()), float(mse_hid.numpy()), y_pred.numpy()

    def on_epoch_end(self, epoch, logs=None):
        mse_out_all, mse_hid_all, pred_tr = self._compute_layer_mse(self.X_tr, self.y_tr)

        mse_out_s, mse_hid_s, _ = self._compute_layer_mse(self.X_tr[0:1], self.y_tr[0:1])

        class_err_tr = np.mean((pred_tr.flatten() >= 0.5).astype(int) != self.y_tr.numpy().flatten())
        pred_te = self.model.predict(self.X_te, verbose=0).flatten()
        class_err_te = np.mean((pred_te >= 0.5).astype(int) != self.y_te)

        W1 = self.model.get_layer('warstwa_ukryta').get_weights()[0]
        W2 = self.model.get_layer('warstwa_wyjsciowa').get_weights()[0]

        lr_obj = self.model.optimizer.learning_rate
        current_lr = float(lr_obj.numpy() if hasattr(lr_obj, 'numpy') else lr_obj)

        self.history['mse_out_train'].append(mse_out_all)
        self.history['mse_hid_train'].append(mse_hid_all)
        self.history['mse_out_sample'].append(mse_out_s)
        self.history['mse_hid_sample'].append(mse_hid_s)
        self.history['class_err_train'].append(class_err_tr)
        self.history['class_err_test'].append(class_err_te)
        self.history['W1_history'].append(W1[:5, 0].copy())
        self.history['W2_history'].append(W2[:5, 0].copy())
        self.history['lr_history'].append(current_lr)

        if mse_out_all <= self.target_mse:
            print(f"\n[Early Stop] Osiągnięto zadany błąd MSE <= {self.target_mse} w epoce {epoch + 1}!")
            self.model.stop_training = True


def build_mlp(lr=0.05, momentum=0.9):
    model = models.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(30, activation='relu', name='warstwa_ukryta'),
        layers.Dense(1, activation='sigmoid', name='warstwa_wyjsciowa')
    ])
    opt = optimizers.SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model


mlp = build_mlp(lr=0.05, momentum=0.9)

diag_cb = DiagnosticsCallback(X_train_scaled, y_train_f, X_test_scaled, y_test, target_mse=0.004)
adaptive_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.7,
    patience=2,
    min_lr=1e-4,
    verbose=0
)

mlp.fit(
    X_train_scaled, y_train_f,
    epochs=300,
    batch_size=32,
    callbacks=[diag_cb, adaptive_lr_cb],
    verbose=0
)

y_pred = (mlp.predict(X_test_scaled, verbose=0).flatten() >= 0.5).astype(int)

print(f"Dokładność modelu (Test): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=data.target_names))


hist = diag_cb.history
epochs = range(1, len(hist['mse_out_train']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, hist['mse_out_train'], label='MSE Warstwa 2 (Wyjściowa) - Cały zbiór', lw=2)
axes[0].plot(epochs, hist['mse_out_sample'], label='MSE Warstwa 2 (Wyjściowa) - Przykład uczący', linestyle='--', alpha=0.7)
axes[0].set_title('Błąd MSE w warstwie wyjściowej')
axes[0].set_xlabel('Epoka')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, hist['mse_hid_train'], label='MSE Warstwa 1 (Ukryta) - Cały zbiór', color='orange', lw=2)
axes[1].plot(epochs, hist['mse_hid_sample'], label='MSE Warstwa 1 (Ukryta) - Przykład uczący', color='red', linestyle='--', alpha=0.6)
axes[1].set_title('Błąd MSE propagowany w warstwie ukrytej')
axes[1].set_xlabel('Epoka')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '1_mse_warstwy.png'), dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, hist['class_err_train'], label='Błąd klasyfikacji 0-1 (Trening)', lw=2)
axes[0].plot(epochs, hist['class_err_test'], label='Błąd klasyfikacji 0-1 (Test)', lw=2)
axes[0].set_title('Błąd klasyfikacji (0-1) na przestrzeni epok')
axes[0].set_xlabel('Epoka')
axes[0].set_ylabel('Odsetek błędnych klasyfikacji')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, hist['lr_history'], color='purple', lw=2)
axes[1].set_title('Adaptacyjny współczynnik uczenia (ReduceLROnPlateau)')
axes[1].set_xlabel('Epoka')
axes[1].set_ylabel('Wartość LR')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '2_blad_klasyfikacji_i_lr.png'), dpi=300)
plt.close()

W1_arr = np.array(hist['W1_history'])
W2_arr = np.array(hist['W2_history'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i in range(W1_arr.shape[1]):
    axes[0].plot(epochs, W1_arr[:, i], label=f'Waga W1[{i},0]')
axes[0].set_title('Ewolucja wag w warstwie 1 (ukrytej)')
axes[0].set_xlabel('Epoka')
axes[0].set_ylabel('Wartość wagi')
axes[0].legend()
axes[0].grid(True)

for i in range(W2_arr.shape[1]):
    axes[1].plot(epochs, W2_arr[:, i], label=f'Waga W2[{i},0]')
axes[1].set_title('Ewolucja wag w warstwie 2 (wyjściowej)')
axes[1].set_xlabel('Epoka')
axes[1].set_ylabel('Wartość wagi')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '3_wagi_w_epokach.png'), dpi=300)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Macierz pomyłek - Breast Cancer')
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '4_macierz_pomylek.png'), dpi=300)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for mom in [0.0, 0.5, 0.9]:
    m = build_mlp(lr=0.05, momentum=mom)
    h = m.fit(X_train_scaled, y_train_f, epochs=100, batch_size=32, verbose=0)
    axes[0].plot(h.history['loss'], label=f'Momentum = {mom}')
axes[0].set_title('Wpływ Momentum na spadek MSE')
axes[0].set_xlabel('Epoka')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True)

for bs in [1, 32, len(X_train_scaled)]:
    label_bs = f'Batch = {bs}' if bs != len(X_train_scaled) else 'Full Batch (455)'
    m = build_mlp(lr=0.05, momentum=0.9)
    h = m.fit(X_train_scaled, y_train_f, epochs=100, batch_size=bs, verbose=0)
    axes[1].plot(h.history['loss'], label=label_bs)
axes[1].set_title('Wpływ rozmiaru Mini-Batch na uczenie')
axes[1].set_xlabel('Epoka')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].grid(True)

for lr_val in [0.005, 0.05, 0.3]:
    m = build_mlp(lr=lr_val, momentum=0.5)
    h = m.fit(X_train_scaled, y_train_f, epochs=100, batch_size=32, verbose=0)
    axes[2].plot(h.history['loss'], label=f'LR = {lr_val}')
axes[2].set_title('Wpływ Learning Rate na zbieżność')
axes[2].set_xlabel('Epoka')
axes[2].set_ylabel('MSE')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '5_zaleznosc_od_parametrow.png'), dpi=300)
plt.close()

report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
df = pd.DataFrame(report).transpose().round(2).astype(object)
df.loc['accuracy', ['precision', 'recall']] = '-'
df['support'] = [44, 70, 114, 114, 114]

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
table.scale(1, 1.8)
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.savefig(os.path.join(PLOTS_DIR, '6_tabela_wynikow.png'), bbox_inches='tight', dpi=300)
plt.close()
