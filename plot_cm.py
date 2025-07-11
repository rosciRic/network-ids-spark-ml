import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Imposta lo stile per grafici più leggibili
plt.style.use('default')
sns.set_palette("husl")


def create_confusion_matrix_1():
    """Prima confusion matrix: Benign vs Malicious"""

    # Tutte le possibili label ordinate alfabeticamente
    all_labels = ['Benign', 'Malicious']

    # Dati originali
    data = {
        'Malicious': {'Benign': 0, 'Malicious': 1450015},
        'Benign': {'Benign': 217, 'Malicious': 40}
    }

    # Crea matrice con compensazione per label mancanti
    matrix = np.zeros((len(all_labels), len(all_labels)))

    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            if true_label in data and pred_label in data[true_label]:
                matrix[i, j] = data[true_label][pred_label]

    return matrix, all_labels


def create_confusion_matrix_2():
    """Seconda confusion matrix: Multi-classe (8 classi)"""

    # Tutte le possibili label ordinate alfabeticamente
    all_labels = sorted(['Audio', 'Background', 'Bruteforce', 'DoS', 'Information Gathering', 'Mirai', 'Text', 'Video'])

    # Dati originali
    data = {
        'Audio': {'Audio': 31, 'Bruteforce': 0, 'DoS': 5, 'Information Gathering': 0, 'Mirai': 2, 'Text': 0,
                  'Video': 1},
        'Bruteforce': {'Audio': 0, 'Bruteforce': 6709, 'DoS': 106, 'Information Gathering': 0, 'Mirai': 0, 'Text': 0,
                       'Video': 0},
        'DoS': {'Audio': 0, 'Bruteforce': 2, 'DoS': 1276914, 'Information Gathering': 0, 'Mirai': 5, 'Text': 0,
                'Video': 0},
        'Information Gathering': {'Audio': 0, 'Bruteforce': 1, 'DoS': 10, 'Information Gathering': 148729, 'Mirai': 14,
                                  'Text': 0, 'Video': 0},
        'Mirai': {'Audio': 0, 'Bruteforce': 25, 'DoS': 390, 'Information Gathering': 274, 'Mirai': 16858, 'Text': 0,
                  'Video': 0},
        'Text': {'Audio': 0, 'Bruteforce': 0, 'DoS': 6, 'Information Gathering': 0, 'Mirai': 0, 'Text': 33, 'Video': 3},
        'Video': {'Audio': 0, 'Bruteforce': 1, 'DoS': 18, 'Information Gathering': 9, 'Mirai': 0, 'Text': 0,
                  'Video': 152},
        'Background': {'Audio': 0, 'Bruteforce': 0, 'DoS': 0, 'Information Gathering': 0, 'Mirai': 0, 'Text': 0,
                       'Video': 7}
    }

    # Crea matrice con compensazione per label mancanti
    matrix = np.zeros((len(all_labels), len(all_labels)))

    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            if true_label in data and pred_label in data[true_label]:
                matrix[i, j] = data[true_label][pred_label]

    return matrix, all_labels


def create_confusion_matrix_3():
    """Terza confusion matrix: Multi-classe dettagliata (28+ classi)"""

    # Tutte le possibili label ordinate alfabeticamente
    base_labels = [
        'Audio', 'Background', 'Bruteforce DNS', 'Bruteforce FTP', 'Bruteforce HTTP',
        'Bruteforce SSH', 'Bruteforce Telnet', 'DoS ACK', 'DoS CWR', 'DoS ECN',
        'DoS FIN', 'DoS HTTP', 'DoS MAC', 'DoS PSH', 'DoS RST', 'DoS SYN',
        'DoS UDP', 'DoS URG', 'Information Gathering', 'Mirai DDoS ACK',
        'Mirai DDoS DNS', 'Mirai DDoS GREETH', 'Mirai DDoS GREIP', 'Mirai DDoS HTTP',
        'Mirai DDoS SYN', 'Mirai DDoS UDP', 'Mirai Scan Bruteforce', 'Text',
        'Video HTTP', 'Video RTP', 'Video UDP'
    ]

    all_labels = sorted(base_labels)

    # Dati originali (mapping nome -> valori per ogni colonna)
    raw_data = {
        'Audio': [33, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
        'Background': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
        'Bruteforce DNS': [0, 0, 4255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bruteforce FTP': [0, 0, 0, 699, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bruteforce HTTP': [0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bruteforce SSH': [0, 0, 0, 0, 0, 723, 0, 0, 0, 0, 3, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Bruteforce Telnet': [0, 0, 0, 0, 0, 0, 907, 11, 0, 0, 1, 61, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0],
        'DoS ACK': [0, 0, 0, 0, 0, 0, 0, 152411, 0, 0, 4719, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS CWR': [0, 0, 0, 0, 0, 0, 0, 0, 141663, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS ECN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 141492, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS FIN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 130856, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS HTTP': [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 124, 15293, 0, 31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
        'DoS MAC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS PSH': [0, 0, 0, 0, 0, 0, 0, 5441, 0, 0, 0, 2, 0, 149370, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS RST': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3764, 189821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS SYN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 4, 0, 0, 0, 146125, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        'DoS UDP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 51087, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'DoS URG': [0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 4, 0, 0, 0, 0, 0, 144927, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Information Gathering': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 148920, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0],
        'Mirai DDoS ACK': [0, 0, 0, 0, 0, 0, 0, 724, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
        'Mirai DDoS DNS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 11120, 0, 0, 0, 0, 0, 0, 0, 0,
                           0],
        'Mirai DDoS GREETH': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Mirai DDoS GREIP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Mirai DDoS HTTP': [0, 0, 0, 0, 0, 0, 0, 53, 0, 0, 2, 204, 0, 18, 0, 0, 1, 0, 375, 0, 8, 0, 971, 3, 1, 0, 0, 0,
                            0, 0],
        'Mirai DDoS SYN': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 1, 2795, 0, 0, 0, 0, 0,
                           0],
        'Mirai DDoS UDP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Mirai Scan Bruteforce': [2, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 146, 3, 1, 0, 0, 1, 0, 41, 0, 21, 0, 2, 0, 898, 0, 0,
                                  0, 0, 0],
        'Text': [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0],
        'Video HTTP': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 52, 8, 0],
        'Video RTP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        'Video UDP': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 0]
    }

    # Lista originale delle colonne (ordine nei dati)
    original_cols = [
        'Audio', 'Background', 'Bruteforce DNS', 'Bruteforce FTP', 'Bruteforce HTTP',
        'Bruteforce SSH', 'Bruteforce Telnet', 'DoS ACK', 'DoS CWR', 'DoS ECN',
        'DoS FIN', 'DoS HTTP', 'DoS MAC', 'DoS PSH', 'DoS RST', 'DoS SYN',
        'DoS UDP', 'DoS URG', 'Information Gathering', 'Mirai DDoS ACK',
        'Mirai DDoS DNS', 'Mirai DDoS GREETH', 'Mirai DDoS GREIP', 'Mirai DDoS HTTP',
        'Mirai DDoS SYN', 'Mirai Scan Bruteforce', 'Text', 'Video HTTP', 'Video RTP', 'Video UDP'
    ]

    # Crea matrice con compensazione per label mancanti
    matrix = np.zeros((len(all_labels), len(all_labels)))

    for i, true_label in enumerate(all_labels):
        if true_label in raw_data:
            for j, pred_label in enumerate(all_labels):
                if pred_label in original_cols:
                    col_idx = original_cols.index(pred_label)
                    if col_idx < len(raw_data[true_label]):
                        matrix[i, j] = raw_data[true_label][col_idx]

    return matrix, all_labels


def plot_confusion_matrix(matrix, labels, title, figsize=(12, 10)):
    """Plotta una confusion matrix con seaborn"""

    plt.figure(figsize=figsize)

    # Converti in DataFrame per seaborn
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)

    # Crea matrice di annotazioni personalizzate per evitare notazione scientifica
    annotations = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            val = int(matrix[i, j])
            if val == 0:
                row.append('0')
            elif val >= 1000000:
                # Per valori >= 1M, usa formato con separatori delle migliaia
                row.append(f'{val:,}'.replace(',', '.'))
            elif val >= 1000:
                # Per valori >= 1K, usa formato con separatori delle migliaia
                row.append(f'{val:,}'.replace(',', '.'))
            else:
                row.append(str(val))
        annotations.append(row)

    # Crea la heatmap con annotazioni personalizzate
    ax = sns.heatmap(df_cm,
                     annot=annotations,
                     fmt='',  # Formato vuoto perché usiamo annotazioni personalizzate
                     cmap='Blues',
                     square=True,
                     linewidths=0.5,
                     cbar_kws={"shrink": .8},
                     annot_kws={'size': 6 if len(labels) > 20 else 8 if len(labels) > 10 else 10})

    # Personalizza il grafico
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')

    # Ruota le etichette per migliorare la leggibilità
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Aggiusta il layout
    plt.tight_layout()

    # Mostra statistiche base
    total_samples = np.sum(matrix)
    accuracy = np.trace(matrix) / total_samples if total_samples > 0 else 0

    print(f"\n=== {title} ===")
    print(f"Dimensioni matrice: {matrix.shape[0]}x{matrix.shape[1]}")
    print(f"Totale campioni: {int(total_samples):,}")
    print(f"Accuracy globale: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Classi ordinate alfabeticamente: {len(labels)}")
    print("-" * 60)

    return ax

def main():
    """Funzione principale per generare e visualizzare tutte le confusion matrices"""

    print("Generazione Confusion Matrices con Seaborn")
    print("=" * 60)

    # Prima matrice: Benign vs Malicious
    matrix1, labels1 = create_confusion_matrix_1()
    plot_confusion_matrix(matrix1, labels1,
                          "Traffic Labels",
                          figsize=(8, 6))

    # Seconda matrice: Multi-classe (8 classi)
    matrix2, labels2 = create_confusion_matrix_2()
    plot_confusion_matrix(matrix2, labels2,
                          "Traffic Types",
                          figsize=(10, 8))

    # Terza matrice: Multi-classe dettagliata
    matrix3, labels3 = create_confusion_matrix_3()
    plot_confusion_matrix(matrix3, labels3,
                          "Traffic Subtypes",
                          figsize=(16, 14))

    # Mostra tutti i grafici
    plt.show()

    print("\nNote:")
    print("- Tutte le matrici sono ordinate alfabeticamente")
    print("- Le label mancanti sono compensate con vettori colonna/riga di zeri")
    print("- I colori più scuri indicano valori più alti")
    print("- L'accuracy è calcolata come (somma diagonale) / (totale campioni)")


if __name__ == "__main__":
    main()