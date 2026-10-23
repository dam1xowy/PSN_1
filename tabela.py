import matplotlib.pyplot as plt

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
    
    plt.savefig(nazwa_pliku, bbox_inches='tight', dpi=300)
    plt.close()
