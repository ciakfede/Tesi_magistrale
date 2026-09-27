'''     ANALISI STATISTICA DATI TELEMETRIA
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             '''

import pandas as pd                                             # Libreria per l'analisi di dataframe
import os                                                       # Libreria necessaria per creare cartelle direttamente da Python
import sys                                                      # Libreria necessaria per gestire interruzioni di codice
import pickle                                                   # Libreria per la gestione del formato di salvataggio pickle
from Analisi_dati import PreProcessing                          # Classe contenente tutte le funzioni necessarie al pre-processing
from Analisi_statistica import StatisticalAnalysis              # Raccolta funzioni necessarie all'analisi dei dati e alla generazione del file Excel di output

''' In input prende i dati di telemetria (caricati nella cartella Telemetria) e li elabora, generando una tabella contenente tutti i dati dei sensori in ordine di timestamp (tramite interpolazione o resampling) e convertendola in formato Excel.
    In seguito, si esegue un'analisi di correlazione tramite i metodi indicati all'inizio --> si generano delle heatmap, dei file .txt riassuntivi con i valori numerici e un dizionario contenente tutti i valori dei coefficienti suddivisi per metodo.
    L'analisi si conclude generando un file .txt con i soli valori medi rilevanti dei coefficienti di correlazione'''


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

sensors_frequencies = {'IMU': 5, 'DVL': 10, 'Depth': 5, 'Depth_rate': 5, 'MOT': 5, 'V_ref': 5, 'GPS': 1}  # Frequenze associate ai sensori

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DF_DICT_DIR = os.path.join(ROOT_DIR, 'Dizionari_dataframes')
os.makedirs(DF_DICT_DIR, exist_ok=True)                                                 # Si genera la cartella di destinazione dei file di salvataggio pickle con i vari dataframe
file_database_raw = os.path.join(DF_DICT_DIR, 'Dataframe_globale.pkl')                  # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_database_resampled = os.path.join(DF_DICT_DIR, 'Dataframe_resampled.pkl')          # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_database_interpolated = os.path.join(DF_DICT_DIR, 'Dataframe_interpolated.pkl')    # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_config = os.path.join(ROOT_DIR, "Telemetrie/Matrice_configurazioni.txt")           # File di testo di input contenente un elenco delle combinazioni condizione ambientale/operativa per ogni traiettoria simulata

# Si inizializzano i metodi prescelti e i dizionari per l'immagazzinamento associati
processing_methods = ['resampling', 'interpolazione']             # Lista di metodi per il trattamento dei dati di telemetria pura
correlation_methods = ['pearson', 'spearman']                     # Lista di metodi usati per l'analisi di correlazione
correlation_dict = {}                                             # Dizionario necessario per accumulare i valori di correlazione calcolati --> forma {metodo: {coppia_valori}...}
mean_correlation_dict = {}                                        # Dizionario necessario per accumulare i valori medi su tutte le missioni per i coefficienti --> forma {'resampling': {'Pearson': {}, 'Spearman': {}}, 'interpolazione': {...}}

# Si inizializzano le variabili
processing_methods_choice = ''
processing_choice = ''

# ======================================================
#             PRE-PROCESSING TELEMETRIA
# ======================================================

# Si richiama la classe legata al Pre-processing dei dati
pre_processing = PreProcessing()

# -----------------------------------------------------
#         ESTRAZIONE DATAFRAMES GREZZI DA CSV
# -----------------------------------------------------

print('\n\n========== CARICAMENTO DATI DI TELEMETRIA ==========')
while True:

    # Gestione dati di telemetria di origine --> si può generare nuovamente il dataframe o utilizzarne uno già creato
    generate_dataframe = input(f'\nSi vuole generare un nuovo dataframe contenente i dati di telemetria (S/N)?').upper()

    if generate_dataframe == 'S':

        # Si analizzano i file .csv contenenti i dati di telemetria di ogni missione --> si genera un dizionario globale e si salva in un apposito file pickle
        pre_processing.csv_analysis(file_config, ROOT_DIR)
        pre_processing.save_raw_dataframes('pickle', DF_DICT_DIR)

        # Gestione del salvataggio del file in formato Excel del dataframe
        while True:
            save_excel = input(f'\nSi vuole salvare il dataframe anche in formato Excel (S/N)?').upper()
            if save_excel == 'S':
                pre_processing.save_raw_dataframes('excel', ROOT_DIR)
                break
            elif save_excel == 'N':
                print(f'  Prosecuzione operazioni pre-processing dati senza salvataggio Excel.')
                break
            else:
                print(f'  Inserito input non valido. Ripetere la scelta.')

        break

    elif generate_dataframe == 'N':

        if os.path.isfile(file_database_raw):
            # Si continua con l'analisi usando i dati raccolti in un file pickle in iterazione precedente
            print(f'  Caricamento del database dal file {file_database_raw}')
            break
        else:
            print(f'  File contenente il database non trovato. Verificare o procedere con la generazione del database.')

    else:

        # Si fa ripetere il ciclo poiché l'input utente non è valido
        print(f'  Inserito un input non valido. Ripetere la scelta.')

# -----------------------------------------------------
#           PROCESSING DEI DATAFRAMES GREZZI
# -----------------------------------------------------

print('\n\n=========== ELABORAZIONE DATAFRAME ==========')
while True:

    process_dataframe = input(f"\nSi vuole generare da zero il dataframe elaborato (S/N)?").upper()

    if process_dataframe == 'S':

        while True:

            processing_methods_choice = input(f'\n  Si vogliono trattare i dati con tutti i metodi disponibili (S/N)?').upper()

            if processing_methods_choice == 'S':

                # Si itera su tutti i metodi presenti all'interno della lista di possibilità
                for processing_method in processing_methods:
                    print(f'\n  -- Gestione dati di telemetria con {processing_method} in corso:')

                    # Si effettua il resampling del dataframe
                    if processing_method == 'resampling':
                        pre_processing.raw_dataframes_processing(generate_dataframe, file_database_raw, sensors_frequencies)
                        pre_processing.save_dataframe_resampled('pickle', DF_DICT_DIR)

                    # Si effettua l'interpolazione del dataframe
                    elif processing_method == 'interpolazione':
                        pre_processing.dataframe_interpolation(generate_dataframe, file_database_raw)
                        pre_processing.save_dataframe_interpolated('pickle', DF_DICT_DIR)

                # Gestione del salvataggio dei dataframe così generati
                while True:

                    save_processed_dataframe = input(f'\n  Si vogliono salvare i dataframe generati in formato Excel (S/N)?').upper()

                    if save_processed_dataframe == 'S':
                        pre_processing.save_dataframe_resampled('excel', ROOT_DIR)
                        pre_processing.save_dataframe_interpolated('excel', ROOT_DIR)
                        break

                    elif save_processed_dataframe == 'N':

                        print(f'    Prosecuzione analisi senza salvataggio Excel dei dataframe elaborati.')
                        break

                    else:

                        print(f'    Inserito input non valido. Ripetere la scelta.')

                break

            elif processing_methods_choice == 'N':

                # Selezione del metodo di gestione dati da usare
                while True:

                    processing_choice = input(f"\n  Quale metodo si vuole utilizzare per trattare i dati (resampling/interpolazione)?").lower()

                    if processing_choice not in processing_methods:

                        print(f"  Il metodo selezionato non è presente nella lista o la formattazione dell'input è errata. Reinserire il metodo.")

                    else:

                        # Si genera il dataframe trattato opportunamente per ogni missione --> salvato in apposito dizionario
                        print(f'\n  -- Gestione dati di telemetria con {processing_choice} in corso:')

                        # Si effettua il resampling del dataframe
                        if processing_choice == 'resampling':
                            pre_processing.raw_dataframes_processing(generate_dataframe, file_database_raw, sensors_frequencies)
                            pre_processing.save_dataframe_resampled('pickle', DF_DICT_DIR)

                        # Si effettua l'interpolazione del dataframe
                        elif processing_choice == 'interpolazione':
                            pre_processing.dataframe_interpolation(generate_dataframe, file_database_raw)
                            pre_processing.save_dataframe_interpolated('pickle', DF_DICT_DIR)

                        # Gestione del salvataggio dei dataframe così generati
                        while True:

                            save_processed_dataframe = input(f'\n  Si vuole salvare il dataframe generato tramite {processing_choice} in formato Excel (S/N)?').upper()

                            if save_processed_dataframe == 'S':

                                # Si effettua il resampling del dataframe
                                if processing_choice == 'resampling':
                                    pre_processing.save_dataframe_resampled('excel', ROOT_DIR)

                                # Si effettua l'interpolazione del dataframe
                                elif processing_choice == 'interpolazione':
                                    pre_processing.save_dataframe_interpolated('excel', ROOT_DIR)
                                break

                            elif save_processed_dataframe == 'N':

                                print(f'    Prosecuzione analisi senza salvataggio Excel dei dataframe elaborati.')
                                break

                            else:

                                print(f'    Inserito input non valido. Ripetere la scelta.')

                        break
                break

            else:

                print(f'    Inserito un input non valido. Ripetere la scelta.')
        break

    elif process_dataframe == 'N':

        if os.path.isfile(file_database_resampled) and os.path.isfile(file_database_interpolated):
            print(f'  Caricamento del dataframe resampled e interpolated dai file {file_database_resampled} e {file_database_interpolated}')
            break
        else:
            print(f'  File contenenti i database non trovati. Verificare o procedere con la generazione dei database.')

    else:

        # Si fa ripetere il ciclo poiché l'input utente non è valido
        print(f'  Inserito un input non valido. Ripetere la scelta.')


# ======================================================
#           ANALISI STATISTICA DI CORRELAZIONE
# ======================================================

print('\n\n========== ANALISI STATISTICA ==========')

analysis = StatisticalAnalysis()

# -----------------------------------------------------
#          CARICAMENTO DATABASE PROCESSATO
# -----------------------------------------------------

# Si gestiscono le varie condizioni --> quale metodo è stato scelto e se si è scelto di caricarlo da zero o usare il dataframe salvato
resampled_dataframes_dict = {}
interpolated_dataframes_dict = {}
if process_dataframe == 'S':

    if processing_methods_choice == 'S':

        resampled_dataframes_dict = pre_processing.resampled_trajectories_dataframes.copy()
        interpolated_dataframes_dict = pre_processing.interpolated_trajectories_dataframes.copy()

    elif processing_methods_choice == 'N':

        # Se il metodo è resampling si carica il solo dizionario associato e si aggiorna la lista per il ciclo for di aggregazione --> se il dizionario dovesse essere vuoto si esce dal programma
        if processing_methods_choice == 'resampling':

            resampled_dataframes_dict = pre_processing.resampled_trajectories_dataframes.copy()
            processing_methods = ['resampling']
            if not resampled_dataframes_dict:
                print(f"  [ERRORE] Il dataset elaborato con resampling non risulta caricato correttamente (è vuoto).")
                sys.exit()

        # Se il metodo è interpolazione si carica il solo dizionario associato e si aggiorna la lista per il ciclo for di aggregazione --> se il dizionario dovesse essere vuoto si esce dal programma
        elif processing_choice == 'interpolazione':

            interpolated_dataframes_dict = pre_processing.interpolated_trajectories_dataframes.copy()
            processing_methods = ['interpolazione']
            if not interpolated_dataframes_dict:
                print(f"  [ERRORE] Il dataset elaborato con interpolazione non risulta caricato correttamente (è vuoto).")
                sys.exit()

elif process_dataframe == 'N':

    try:
        with open(file_database_resampled, 'rb') as f:
            resampled_dataframes_dict = pickle.load(f)

        with open(file_database_interpolated, 'rb') as f:
            interpolated_dataframes_dict = pickle.load(f)

    except Exception as e:
        print(f"  [WARNING] Errore nell'apertura del file pickle dei dataframe elaborati --> {e}")

# Se uno dei due dataframe risulta vuoto ci sono stati errori non previsti a monte e si interrompe qui il codice
if processing_methods_choice == 'S' and (not resampled_dataframes_dict or not interpolated_dataframes_dict):
    print(f"  [ERRORE] Almeno uno dei due dataset elaborati non risulta caricato correttamente (è vuoto).")
    sys.exit()

# Si uniscono i dataframe, usando come chiavi i metodi utilizzati nel pre-processing
processed_dataframes_dict = {}
for processing_method in processing_methods:

    if processing_method not in processed_dataframes_dict:
        processed_dataframes_dict[processing_method] = {}

    if processing_method == 'resampling':
        processed_dataframes_dict[processing_method] = resampled_dataframes_dict
    elif processing_method == 'interpolazione':
        processed_dataframes_dict[processing_method] = interpolated_dataframes_dict

# -----------------------------------------------------
#          GESTIONE ANALISI DI CORRELAZIONE
# -----------------------------------------------------

while True:

    both_methods = input(f"\nSi vogliono eseguire tutti i tipi di analisi indicati (S/N)?").upper()

    if both_methods == 'S':

        # Si effettua il calcolo dei coefficienti con entrambi i metodi e per ognuno dei due dataset (resampling e interpolazione)
        analysis.correlation_analysis(correlation_methods, processed_dataframes_dict, processing_methods, ROOT_DIR)

        # Si effettua il calcolo dei valori medi sui dizionari generati
        analysis.mean_correlation_analysis(correlation_methods, processing_methods, ROOT_DIR)

        break

    elif both_methods == 'N':

        while True:
            method_choice = input(f"\n  Quale analisi si vuole effettuare?").lower()

            if method_choice == 'pearson':

                # Si effettua il calcolo dei coefficienti con il metodo prescelto e per ognuno dei due dataset (resampling e interpolazione)
                analysis.correlation_analysis(correlation_methods[0], processed_dataframes_dict, processing_methods, ROOT_DIR)

                # Si effettua il calcolo dei valori medi sui dizionari generati
                analysis.mean_correlation_analysis(correlation_methods[0], processing_methods, ROOT_DIR)

                break

            elif method_choice == 'spearman':

                # Si effettua il calcolo dei coefficienti con il metodo prescelto e per ognuno dei due dataset (resampling e interpolazione)
                analysis.correlation_analysis(correlation_methods[1], processed_dataframes_dict, processing_methods, ROOT_DIR)

                # Si effettua il calcolo dei valori medi sui dizionari generati
                analysis.mean_correlation_analysis(correlation_methods[1], processing_methods, ROOT_DIR)

                break

            else:
                print(f"    Inserito un input non valido. Ripetere la scelta.")

        break

    else:

        print(f'  Input non valido. Ripetere la scelta.')





# Condizione di verifica per l'ingresso nel ciclo --> la lista deve contenere elementi e il metodo di gestione dati deve essere interpolazione (altrimenti avrei troppi NaN)
#if tutte_le_tabelle and processing_method == 'interpolazione':

    # Unione di tutti i dati per l'analisi globale
    #tabella_globale = pd.concat(tutte_le_tabelle, axis=0, ignore_index=True)

    # Chiamata alla nuova funzione di analisi aggregata
    #Funzioni_analisi_correlazione.analisi_pca_globale(processing_method, tabella_globale)

# Si richiama la funzione che permette di creare una tabella riassuntiva dei coefficienti medi per i vari metodi di correlazione e interpolazione introdotti --> utile solo ai fini della tesi
#tabella_tesi(mean_correlation_dict)

