import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from main import y_test, y_pred, data

report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
df = pd.DataFrame(report).transpose().round(2).astype(object)

df.loc['accuracy', ['precision', 'recall']] = '-'
df['support'] = [44, 70, 114, 114, 114]

def save_t(df, nazwa_pliku='tabela_wynikow.png'):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.scale(1, 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    folder = os.path.dirname(os.path.abspath(__file__))
    sciezka = os.path.join(folder, nazwa_pliku)

    plt.savefig(sciezka, bbox_inches='tight', dpi=300)
    plt.close()

save_t(df)
