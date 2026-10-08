"""     ANALISI STATISTICA DATI TELEMETRIA
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             """

import pandas as pd                                             # Libreria per l'analisi di dataframe
import os                                                       # Libreria necessaria per creare cartelle direttamente da Python
import sys                                                      # Libreria necessaria per gestire interruzioni di codice
import pickle                                                   # Libreria per la gestione del formato di salvataggio pickle
from Analisi_dati import PreProcessing                          # Classe contenente tutte le funzioni necessarie al pre-processing
from Analisi_statistica import StatisticalAnalysis              # Raccolta funzioni necessarie all'analisi dei dati e alla generazione del file Excel di output

''' Si ricevono in input i dati di telemetria grezzi contenuti in file .csv suddivisi per la combinazione (traiettoria, combinazione condizioni ambientali/operative) che definisce la singola missione. 
    Le combinazioni sono estratte dal file di testo opportuno e usate per generare le chiavi del dizionario riassuntivo oltre che per accedere ai file nella cartella Telemetrie.
    I file sono, quindi, analizzati per estrarre e salvare i DataFrame grezzi nel dizionario opportuno e elaborati con uno o più metodi differenti per produrre dei DataFrame regolarizzati da usare per le analisi.
    L'utente può scegliere se procedere da 0 con le due fasi (se ci sono stati cambiamenti nei file o se ci sono stati cambiamenti nel codice per l'elaborazione). In caso può anche scegliere di fare un solo tipo di processing se l'altro non viene intaccato.
    Successivamente si procede con l'analisi statistica. Anche qui l'utente può selezionare il tipo di analisi da effettuare e con quali Datasets farla. Al termine di questa fase si producono dei file di testo contenenti
    i coefficienti di correlazione di ogni singola missione e uno con i valori medi di quelle più significative, oltre che delle heatmap che mostrano graficamente questi legami.
    Infine, si procede con un'analisi PCA.
    Per cambiare i metodi usati per il processing dei DataFrame o per l'analisi statistica occorre cambiare le liste presenti nella fase di inizializzazione delle variabili'''


# Funzione per la creazione della tabella riassuntiva con tutti i coefficienti medi per la tesi --> li salva successivamente in un file di testo
def tabella_tesi(storico_coppie_medie):

    print(f'\nGenerazione e salvataggio della tabella per Latex in corso:')

    percorso_file = os.path.join('Risultati_analisi', 'Tabella_riassuntiva_Latex.txt')

    # Si apre il file in modalità append e si genera l'intestazione della sezione dei valori medi
    with open(percorso_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"CODICE LATEX COMPLETO PER LA GENERAZIONE DI UNA TABELLA RIASSUNTIVA\n")
        f.write("=" * 60 + "\n\n")

        righe_tabella = []

        # 1. Identificazione coppie uniche
        tutte_le_coppie = set()
        for gestione in storico_coppie_medie.values():
            for statistica in gestione.values():
                tutte_le_coppie.update(statistica.keys())

        tutte_le_coppie = sorted(list(tutte_le_coppie))

        # 2. Costruzione righe
        for coppia in tutte_le_coppie:
            riga = {
                'Relazione': coppia,
                'P_Res': storico_coppie_medie.get('resampling', {}).get('pearson', {}).get(coppia, 0),
                'S_Res': storico_coppie_medie.get('resampling', {}).get('spearman', {}).get(coppia, 0),
                'P_Int': storico_coppie_medie.get('interpolazione', {}).get('pearson', {}).get(coppia, 0),
                'S_Int': storico_coppie_medie.get('interpolazione', {}).get('spearman', {}).get(coppia, 0)
            }
            righe_tabella.append(riga)

        data_frame_finale = pd.DataFrame(righe_tabella)

        # Protezione caratteri speciali LaTeX
        data_frame_finale['Relazione'] = data_frame_finale['Relazione'].str.replace('%', r'\%', regex=False)
        data_frame_finale['Relazione'] = data_frame_finale['Relazione'].str.replace('_', r'\_', regex=False)
        data_frame_finale['Relazione'] = data_frame_finale['Relazione'].str.replace('<->', r'$\longleftrightarrow$', regex=False)
        data_frame_finale['Relazione'] = data_frame_finale['Relazione'].str.replace('ω', r'$\omega$', regex=False)

        # 3. COSTRUZIONE MANUALE DEL CODICE LATEX
        f.write(r"\begingroup" + "\n")
        f.write(r"\footnotesize" + "\n")
        f.write(r"\keepXColumns" + "\n")
        f.write(r"\begin{tabularx}{\textwidth}{|X | cc | cc |}" + "\n")
        f.write(r"\caption{Analisi comparativa dei risultati medi}" + "\n")
        f.write(r"\label{tab:analisi_correlazione} \\")
        f.write(r"\hline" + "\n")
        f.write(r"\textbf{Relazione} & \multicolumn{2}{c|}{\textbf{Resampling}} & \multicolumn{2}{c|}{\textbf{Interpolazione}} \\" + "\n")
        f.write(r"\cline{2-5}" + "\n")
        f.write(r"& \textbf{Pearson} & \textbf{Spearman} & \textbf{Pearson} & \textbf{Spearman} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\endfirsthead" + "\n")

        # Gestione pagine successive
        f.write(r"\multicolumn{5}{|c|}{{Continua dalla pagina precedente}} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"& \textbf{Pearson} & \textbf{Spearman} & \textbf{Pearson} & \textbf{Spearman} \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\endhead" + "\n")

        f.write(r"\hline \multicolumn{5}{|r|}{{Continua nella pagina successiva}} \\ \hline" + "\n")
        f.write(r"\endfoot" + "\n")
        f.write(r"\hline \endlastfoot" + "\n")

        # Aggiunta dei dati con \hline dopo ogni riga
        for _, row in data_frame_finale.iterrows():
            linea_dati = (fr"{row['Relazione']} & {row['P_Res']:.3f} & {row['S_Res']:.3f} & "
                          fr"{row['P_Int']:.3f} & {row['S_Int']:.3f} \\" + "\n")
            f.write(linea_dati)

        f.write(r"\end{tabularx}" + "\n")
        f.write(r"\endgroup")

    print(f'Generazione e salvataggio tabella completato.')

    return data_frame_finale


# ======================================================
#              INIZIALIZZAZIONE VARIABILI
# ======================================================

sensors_frequencies = {'IMU': 5, 'DVL': 10, 'Depth': 5, 'DepthVel': 5, 'MOT': 5, 'V_ref': 5, 'GPS': 1}  # Frequenze associate ai sensori

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                                 # Directory in cui si trova lo script python
ROOT_DIR : str = os.path.dirname(SCRIPT_DIR)                                            # Directory esterna a quella dello script --> in questa verranno generate le cartelle degli output
DF_DICT_DIR = os.path.join(ROOT_DIR, 'Dizionari_datasets')                              # Directory in cui verranno salvati i file pickle associati a ogni dizionario prodotto nell'esecuzione
os.makedirs(DF_DICT_DIR, exist_ok=True)                                                 # Si genera la cartella di destinazione dei file di salvataggio pickle con i vari DataFrame (se non presente)
file_datasets_raw = os.path.join(DF_DICT_DIR, 'Dataset_globale.pkl')                    # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_datasets_resampled = os.path.join(DF_DICT_DIR, 'Dataset_resampled.pkl')            # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_datasets_interpolated = os.path.join(DF_DICT_DIR, 'Dataset_interpolated.pkl')      # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_config = os.path.join(ROOT_DIR, "Telemetrie/Matrice_configurazioni.txt")           # File di testo di input contenente un elenco delle combinazioni condizione ambientale/operativa per ogni traiettoria simulata

# Si inizializzano i metodi prescelti e i dizionari per l'immagazzinamento associati
possible_processing_methods = ['resampling', 'interpolazione']              # Lista di metodi per il trattamento dei dati di telemetria pura
possible_correlation_methods = ['pearson', 'spearman']                      # Lista di metodi usati per l'analisi di correlazione
processed_datasets_dict = {}                                                # Dizionario che conterrà i dizionari processati con i vari metodi prescelti --> nella forma {'metodo 1': {}, 'metodo 2': {}, ...}

files_processed_datasets = {'resampling': file_datasets_resampled, 'interpolazione': file_datasets_interpolated}

#correlation_dict = {}                                             # Dizionario necessario per accumulare i valori di correlazione calcolati --> forma {metodo: {coppia_valori}...}
#mean_correlation_dict = {}                                        # Dizionario necessario per accumulare i valori medi su tutte le missioni per i coefficienti --> forma {'resampling': {'Pearson': {}, 'Spearman': {}}, 'interpolazione': {...}}

# Si inizializzano le variabili --> per evitare soft warnings nel seguito
processing_methods_choice = ''
processing_choice = ''

# ======================================================
#             PRE-PROCESSING TELEMETRIA
# ======================================================

print('\n\n===================================================')
print('============ AVVIO PRE-PROCESSING DATI ============')
print('===================================================')

# Si richiama la classe associata al pre-processing
pre_processing = PreProcessing()
print(f"\nSelezionare i parametri legati all'estrazione e all'analisi dei dati di telemetria:")

# -----------------------------------------------------
#         ESTRAZIONE DATAFRAMES GREZZI DA CSV
# -----------------------------------------------------

while True:

    generate_new_dataset = input(f"\n  Si vuole generare un nuovo Dataset contenente i dati di telemetria (S/N/EXIT)?").upper()

    if generate_new_dataset == 'S':

        pre_processing.csv_analysis(file_config, ROOT_DIR)
        pre_processing.save_raw_datasets_dict('pickle', DF_DICT_DIR)
        raw_dataset = pre_processing.raw_database

        while True:

            save_raw_dataset = input(f"\n    Si vuole salvare in formato Excel il Dataset prodotto (S/N/EXIT)?").upper()

            if save_raw_dataset == 'S':

                pre_processing.save_raw_datasets_dict('excel', DF_DICT_DIR)
                break

            elif save_raw_dataset == 'N':
                print(f"      Prosecuzione analisi senza salvare i Dataset grezzi prodotti.")
                break

            elif save_raw_dataset == 'EXIT':
                print(f"      Chiusura del programma forzata da utente in corso...")
                sys.exit()

            else:
                print(f"      Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")

        break

    elif generate_new_dataset == 'N':

        # Controllo esistenza del file pickle
        if os.path.isfile(file_datasets_raw) and not os.stat(file_datasets_raw).st_size == 0:
            print(f"    Utilizzo del Dataset precedentemente generato e salvato in {file_datasets_raw}.")
            with open(file_datasets_raw, 'rb') as f:
                raw_dataset = pickle.load(f)
            break
        else:
            print(f"    Il file {file_datasets_raw} risulta non presente o vuoto. Verificare la presenza o generare un nuovo dataset.")

    elif generate_new_dataset == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")


# -----------------------------------------------------
#           SELEZIONE METODI PER PROCESSING
# -----------------------------------------------------

while True:

    all_processing_methods = input("\n  Si vuole effettuare il processing dei dati con tutti i metodi disponibili (S/N/EXIT)?").upper()

    if all_processing_methods == 'S':
        processing_methods = possible_processing_methods
        break

    elif all_processing_methods == 'N':

        while True:

            processing_methods_choice = input(f"\n    Selezionare il/i metodi di processing dei dati da utilizzare ({[m for m in possible_processing_methods]}) separati da una virgola:").lower()

            processing_methods = [m.strip() for m in processing_methods_choice.split(',') if m.strip()]

            if any(m not in possible_processing_methods for m in processing_methods):
                print(f"      Il/i metodi selezionati non sono stati inseriti correttamente. Ripetere la scelta")

            else:
                break

        break

    elif all_processing_methods == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")


# -----------------------------------------------------
#         GESTIONE SCELTA SU NUOVA ELABORAZIONE
# -----------------------------------------------------

while True:

    process_dataset = input(f"\n  Si vuole procedere con una nuova elaborazione dei dati di telemetria (S/N/EXIT)?").upper()

    # Se si è generato un nuovo Dataset grezzo è inevitabile rieseguire anche l'elaborazione dello stesso
    if generate_new_dataset == 'S':
        process_dataset = 'S'

    if process_dataset == 'S':

        # Si itera su ogni metodo selezionato in precedenza --> per ognuno si effettua l'analisi tramite apposita funzione
        for processing_method in processing_methods:

            if processing_method == 'resampling':
                pre_processing.raw_dataframes_processing(generate_new_dataset, file_datasets_raw, sensors_frequencies)
                pre_processing.save_processed_dataset_dict('pickle', DF_DICT_DIR, 'resampling')
                processed_datasets_dict[processing_method] = pre_processing.resampled_database

            elif processing_method == 'interpolazione':
                pre_processing.dataframe_interpolation(generate_new_dataset, file_datasets_raw)
                pre_processing.save_processed_dataset_dict('pickle', DF_DICT_DIR, 'interpolazione')
                processed_datasets_dict[processing_method] = pre_processing.interpolated_database

            # Gestione eventuale salvataggio in formato Excel
            while True:

                save_processed_dataset = input(f"\n    Si desidera salvare il Dataset trattato con {processing_method.upper()} in formato Excel (S/N/EXIT)?").upper()

                if save_processed_dataset == 'S':

                    pre_processing.save_processed_dataset_dict('excel', DF_DICT_DIR, processing_method)
                    break

                elif save_processed_dataset == 'N':
                    print(f"      Prosecuzione analisi senza salvare i Dataset elaborati prodotti.")
                    break

                elif save_processed_dataset == 'EXIT':
                    print(f"      Chiusura del programma forzata da utente in corso...")
                    sys.exit()

                else:
                    print(f"      Input inserito non valido. Ripetere la scelta con s, n o exit.")


    elif process_dataset == 'N':

        file_exists = True
        for processing_method in processing_methods:

            dict_file_path = files_processed_datasets[processing_method]
            if os.path.isfile(dict_file_path) and os.stat(dict_file_path).st_size > 0:

                with open(dict_file_path, 'rb') as f:
                    processed_datasets_dict[processing_method] = pickle.load(f)

            else:
                print(f"    Il file {dict_file_path} risulta non presente o vuoto. Ricontrollare o rieseguire l'analisi dei dati grezzi. ")
                file_exists = False

        # Solo se tutti i file pickle di interesse sono stati letti correttamente si esce dal ciclo
        if file_exists:
            break

    elif process_dataset == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")



# ======================================================
#           ANALISI STATISTICA DI CORRELAZIONE
# ======================================================

print('\n\n=====================================================')
print('========== AVVIO FASE DI ANALISI STATISTICA =========')
print('=====================================================')

analysis = StatisticalAnalysis()

# -----------------------------------------------------
#          CARICAMENTO DATABASE PROCESSATO
# -----------------------------------------------------

# Se uno dei due dataframe risulta vuoto ci sono stati errori non previsti a monte e si interrompe qui il codice
print("\nVerifica correttezza Dataset per i metodi di elaborazione selezionati:")
if not all(d for d in processed_datasets_dict.values()):
    print(f"  [ERRORE] Almeno uno dei due dataset elaborati non risulta caricato correttamente (è vuoto).")
    sys.exit()
else:
    print("  I Dataset sono stati generati/caricati correttamente.")

# -----------------------------------------------------
#          GESTIONE ANALISI DI CORRELAZIONE
# -----------------------------------------------------

print(f"\nSelezionare i parametri legati all'analisi di correlazione da effettuare:")

while True:

    all_correlation_methods = input(f"\n  Si vogliono eseguire tutti i tipi di analisi indicati (S/N/EXIT)?").upper()

    if all_correlation_methods == 'S':

        correlation_methods = possible_correlation_methods
        analysis.correlation_analysis(correlation_methods, processed_datasets_dict, processing_methods, ROOT_DIR)
        analysis.mean_correlation_analysis(correlation_methods, processing_methods, ROOT_DIR)

        break

    elif all_correlation_methods == 'N':

        while True:

            correlation_methods_choice = input(f"\n    Quale analisi si vuole effettuare ({[m for m in possible_correlation_methods]})?").lower()

            correlation_methods = [m.strip() for m in correlation_methods_choice.split(',') if m.strip()]

            valid_input = True
            for correlation_method in correlation_methods:

                if correlation_method in possible_correlation_methods:

                    analysis.correlation_analysis(correlation_method, processed_datasets_dict, processing_methods, ROOT_DIR)
                    analysis.mean_correlation_analysis(correlation_method, processing_methods, ROOT_DIR)

                else:
                    valid_input = False

            if valid_input:
                break
            else:
                print(f"      Inserito un input non valido. Ripetere la scelta selezionando uno o più metodi corretti.")

        break

    elif all_correlation_methods == 'EXIT':
        print(f"      Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:

        print(f'    Input non valido. Ripetere la scelta selezionando s, n o exit.')





# Condizione di verifica per l'ingresso nel ciclo --> la lista deve contenere elementi e il metodo di gestione dati deve essere interpolazione (altrimenti avrei troppi NaN)
#if tutte_le_tabelle and processing_method == 'interpolazione':

    # Unione di tutti i dati per l'analisi globale
    #tabella_globale = pd.concat(tutte_le_tabelle, axis=0, ignore_index=True)

    # Chiamata alla nuova funzione di analisi aggregata
    #Funzioni_analisi_correlazione.analisi_pca_globale(processing_method, tabella_globale)

# Si richiama la funzione che permette di creare una tabella riassuntiva dei coefficienti medi per i vari metodi di correlazione e interpolazione introdotti --> utile solo ai fini della tesi
#tabella_tesi(mean_correlation_dict)

