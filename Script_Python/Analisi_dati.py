"""       PRE-PROCESSING E ANALISI DATI
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             """

''' Il pre-processing produce delle liste di liste di DataFrame (una lista contenente più dataframe, per ogni missione e per ogni blocco di sensori) --> fatto su più 
    griglie di riferimento con passo uniforme (0.100s per DVL, 0.200s per IMU e 1s per GPS) --> misurazioni asincrone spostate al decimo di secondo più vicino all'istante 
    reale --> inizio e fine del dataframe per ogni sensore (su singola missione) sono coincidenti per evitare sfasamenti nella valutazione della rete --> i dataframe
    sono poi convertiti in tensori pytorch secondo la medesima struttura di divisioni dei DataFrame

    La classe Post-Processing presenta varie funzioni utili all'analisi dei risultati dell'esecuzione della rete neurale, come la rappresentazione dei grafici e il calcolo 
    dell'errore medio (RMSE) --> è, inoltre, presente una funzione opzionale che permette di separare automaticamente le singole ripetizioni di traiettoria all'interno di 
    una missione (con criterio )
    '''

import numpy as np                              # Libreria per i calcoli matematici
import torch                                    # Libreria per i calcoli matematici
import pandas as pd                             # Libreria per la gestione dei dataframe
import warnings                                 # Libreria per la gestione dei warnings come exception
from pandas.errors import DtypeWarning          # Libreria per la gestione dei warnings di tipo misto nella lettura di un dataframe pandas
import pickle                                   # Libreria per la gestione del formato di salvataggio pickle
import os                                       # Libreria per la gestione dell'interfaccia tra sistema operativo e script
from sklearn.metrics import mean_squared_error  # Libreria per l'utilizzo della funzione per il calcolo dell'errore quadratico medio
import matplotlib.pyplot as plt                 # Libreria per la gestione dell'ambiente grafico
import utm                                      # Libreria per la gestione della conversione di coordinate espresse in gradi (WGS84)
import sys                                      # Libreria che permette di interagire con il Runtime di Python

pd.set_option('display.max_columns', None)              # Visualizza tutte le colonne (nessun limite al numero)
pd.set_option('display.max_colwidth', None)             # Visualizza tutto il contenuto della cella (evita di troncare stringhe lunghe)
pd.set_option('display.expand_frame_repr', False)       # Evita che le colonne vadano a capo su più righe nel terminale
pd.options.display.max_rows = 100                             # Imposta il numero massimo di righe visualizzabili a schermo

warnings.filterwarnings('error', category=pd.errors.DtypeWarning)           # Forza i DtypeWarning a comportarsi come eccezioni


class PreProcessing:

    # Si inizializza la struttura comune degli elementi appartenenti alla classe
    def __init__(self, reference_frequency='100ms'):

        self.freq = reference_frequency                 # Si assegna alla variabile interna la frequenza base della griglia

        self.config_matrix = {}                         # Si inizializza un dizionario che conterrà, per ogni traiettoria, una lista di stringhe che individuano la combinazione scenario ambientale/scenario operativo associata a ogni missione --> avrà forma {'traiettoria0': ['comb0', 'comb1', ...], 'traiettoria1': [], ...}

        self.raw_database = {}                          # Si inizializza un dizionario che una volta riempito avrà forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe...}, 'traiettoria1'; {}, ...}

        self.sensors_divided_database = {}              # Si inizializza un dizionario che conterrà per ogni missione i dataframe dei singoli sensori su griglia temporale comune --> ha forma {'traiettoria0': {'missione1': {'IMU'; dataframe, 'DVL':dataframe...} , 'missione2': {'IMU'; dataframe, 'DVL':dataframe...}, ...}, 'traiettoria1': {}, ...}

        self.resampled_database = {}                    # Si inizializza un dizionario che conterrà per ogni missione il dataframe globale trattato con resampling --> ha forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe, ...}, 'traiettoria1': {}, ...}

        self.interpolated_database = {}                 # Si inizializza un dizionario che conterrà per ogni missione il dataframe globale trattato con interpolazione --> ha forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe, ...}, 'traiettoria1': {}, ...}

        self.pytorch_tensor_dict = {}                   # Si inizializza un dizionario che conterrà per ogni missione i tensori di ogni blocco sensori --> ha forma {'traiettoria0': {'missione1': {'IMU'; tensore, 'DVL':tensore...}, 'missione2': {'IMU'; tensore, 'DVL':tensore...}, ...}, 'traiettoria1': {}, ...}

        self.normalization_parameters_dict = {}         # Dizionario per salvare medie e deviazioni standard --> nella forma {'IMU': {'mean':valore, 'std':valore}, 'DVL': {}...}

        self.warn_count = 0                             # Contatore warning --> da riinizializzare a 0 quando si cambia traiettoria o blocco di sensori

        self.empty_missions_list = []                   # Lista di tuple (traiettoria, combinazione) per cui si sono verificati errori nella fase di pre-processing --> dal momento dell'errore i DataFrame associati nel dizionario sono vuoti

        # Dizionario con i nomi dei sensori e i singoli valori misurati da ognuno --> necessario in csv_analysis
        self.data_labels = {'DVL': ['DVL_Lock', 'DVL_Vx [m/s]', 'DVL_Vy [m/s]', 'DVL_Vz [m/s]', 'DVL_Altitude [m]'],
                            'Attitude': ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'],
                            'Depth': ['Depth [m]'],
                            'DepthVel': ['Depth_Rate [m/s]'],
                            'AxisRef': ['Ref_Vx [%]', 'Ref_Vy [%]', 'Ref_Vz [%]', 'Ref_ωx [%]', 'Ref_ωy [%]', 'Ref_ωz [%]'],
                            'MotorRef': ['Mot_FV [%]', 'Mot_FL [%]', 'Mot_RV [%]', 'Mot_RL [%]', 'Mot_FW1 [%]', 'Mot_FW2 [%]', 'Mot_FW3 [%]', 'Mot_FW4 [%]'],
                            'UTMPos': ['UTM_North [m]', 'UTM_East [m]', 'Zone', 'Quality']}

        # Si definiscono i blocchi di sensori con frequenze differenti --> tuple che contengono sia le colonne che la frequenza associata oltre che la lista di destinazione associata all'attributo self --> sono quelli usati per il processing
        self.sensors_labels_dict = {'DVL': ['timestamp', 'DVL_Vx [m/s]', 'DVL_Vy [m/s]', 'DVL_Vz [m/s]', 'DVL_Altitude [m]'],
                                    'IMU': ['timestamp', 'Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'],
                                    'Depth': ['timestamp', 'Depth [m]'],
                                    'DepthVel': ['timestamp', 'Depth_Rate [m/s]'],
                                    'MOT': ['timestamp', 'Mot_FV [%]', 'Mot_FL [%]', 'Mot_RV [%]', 'Mot_RL [%]', 'Mot_FW1 [%]', 'Mot_FW2 [%]'],
                                    'V_ref': ['timestamp', 'Ref_Vx [%]', 'Ref_Vy [%]', 'Ref_Vz [%]', 'Ref_ωx [%]', 'Ref_ωy [%]', 'Ref_ωz [%]'],
                                    'GPS': ['timestamp', 'UTM_North [m]', 'UTM_East [m]']}

        self.min_std = 0.01                         # Soglia minima di deviazione standard per i vari parametri

    # ======================================================
    #            ESTRAZIONE DATI DA FILE CSV
    # ======================================================

    # Funzione interna in grado di generare un dizionario contenente le combinazioni di simulazione utilizzate partendo dal file di testo di riferimento
    def config_txt_analysis(self, file_config):

        try:

            print(f"\n  Analisi file di configurazione per estrazione dati associati alle missioni in corso:")

            with open(file_config, "r") as file:

                for line in file:

                    line = line.strip(": \n")   # Si eliminano tutti gli spazi, il carattere ":" (dove presente) e l'invio al termine della linea analizzata
                    line = line.lstrip(" - ")   # Si eliminano gli spazi e il segno "-" all'inizio della linea analizzata
                    line = line.split(" ")      # Si fa in modo di trasformare il testo di ogni linea in una lista di stringhe --> separatore "spazio" per le intestazioni del tipo "Traiettoria 1:"

                    # Se il primo elemento della linea è la parola traiettoria si sta cambiando missione, altrimenti si sta analizzando una specifica configurazione
                    if line[0] == "Traiettoria":

                        # Si inizializza la lista di configurazioni associata alla traiettoria
                        traj_number = line[1]
                        if traj_number not in self.config_matrix:
                            self.config_matrix[traj_number] = []
                    else:

                        if line[0] != '':
                            self.config_matrix[traj_number].append(line[0])  # Si aggiunge alla lista della traiettoria la stringa con la combinazione scenario ambientale/operativo

        except FileNotFoundError:
            raise FileNotFoundError(f"    [ERROR] Il {file_config} non è stato trovato. Ricontrollare il percorso del file.") from None

        except Exception as e:
            print(f"    [WARNING] Errore nella lettura del file di testo {file_config} --> {e}")

        return self

    # Funzione interna che permette di eliminare dal file di telemetria in ingresso tutti i caratteri non numerici dalle colonne dei dati
    def dataframe_cleaning(self, mission_dataframe, trajectory, combination):

        try:

            # Colonne su cui applicare la pulizia (v1, v2, ..., v8)
            cols_names = [f'v{i}' for i in range(1, 9) if f'v{i}' in mission_dataframe.columns]

            # Colonna del dataframe associata al nome oggetto di analisi
            cols = mission_dataframe[cols_names]

            # Si salvano in una maschera booleana le posizioni in cui ci sono valori NaN per lo slice di dataframe
            nan_mask = cols.isna()

            # Si puliscono le celle tramite il modulo Regular Expression --> si eliminano tutti i caratteri che non siano un "-", un ".", un "e" (per numeri in notazione scientifica) o con una cifra da 0 a 9
            clean_cols = cols.astype(str).replace(r'[^-0-9.eE]', '', regex=True)

            # Si applica la maschera booleana alla struttura pulita --> elimina gli elementi True, che corrispondono ai valori iniziali NaN o alle celle diventate vuote dopo la sostituzione
            clean_cols = clean_cols.mask(nan_mask | (clean_cols == ''))

            # Si effettua la conversione da stringa a valore numerico di ogni cella --> i valori tagliati dalla maschera booleana o quelli non convertibili diventano NaN (per effetto di coerce)
            mission_dataframe[cols_names] = clean_cols.apply(pd.to_numeric, errors='coerce')

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nel processo di pulizia del dataframe associato alla missione {trajectory} - {combination} --> {e}")
            mission_dataframe = pd.DataFrame()

        return mission_dataframe

    # Funzione per il trattamento dei file .csv prodotti dalla simulazione --> si unificano i dati DVL e INS se separati e si genera un unico dizionario globale con i dataframe di tutte le missioni
    def csv_analysis(self, file_config, ROOT_DIR):

        # -----------------------------------------------------
        #            UNIFICAZIONE E ANALISI FILE CSV
        # -----------------------------------------------------

        # Inizializzazione del dizionario con le configurazioni per ogni missione
        self.config_txt_analysis(file_config)

        # Inizializzazione del dizionario in cui si immagazzineranno i dataframe
        dataframes_dict = {}

        # Si itera su ogni traiettoria simulata --> l'elemento config_list è una lista di stringhe contenenti la coppia configurazione ambientale/operativa del drone in simulazione
        for trajectory, config_list in self.config_matrix.items():

            print(f'\n  Analisi file csv per traiettoria {trajectory} in corso:')

            # Si definisce la cartella di riferimento
            input_folder = os.path.join(ROOT_DIR, f'Telemetrie/Telemetria_traiettoria_{trajectory}')

            # Si inizializza il dizionario interno associato alla singola missione
            if trajectory not in dataframes_dict:
                dataframes_dict[trajectory] = {}

            self.warn_count = 0
            for combination in config_list:

                # Se la missione è la 0 si evita l'unificazione e si procede con la traiettoria successiva --> ha già un solo file di telemetria per configurazione
                if trajectory == "0":

                    telemetry_file_name = os.path.join(input_folder, f'M{combination}.csv')

                    try:

                        # noinspection argument-list
                        mission_dataframe = pd.read_csv(telemetry_file_name)

                        # Check di verifica sul DataFrame di missione vuoto già al momento dell'estrazione --> si segnala ma per il momento non si fa gestisce appositamente
                        if mission_dataframe.empty:
                            print(f"    [WARNING] Il DataFrame associato alla missione {trajectory} - {combination} risulta essere vuoto al momento dell'estrazione dati. Ricontrollare il file csv di origine.")
                            self.warn_count += 1
                            dataframes_dict[trajectory][combination] = pd.DataFrame()
                            continue

                        dataframes_dict[trajectory][combination] = mission_dataframe

                    except FileNotFoundError:
                        dataframes_dict[trajectory][combination] = pd.DataFrame()
                        self.warn_count += 1
                        print(f'    [WANRING] File di telemetria per la missione {trajectory} - {combination} non trovato.')
                        continue

                    except DtypeWarning:
                        mission_dataframe = pd.read_csv(telemetry_file_name, low_memory=False)
                        dataframes_dict[trajectory][combination] = mission_dataframe
                        self.warn_count += 1
                        print(f'    [WARNING] Il file di telemetria per la missione {trajectory} - {combination} contiene valori di tipo misto (problema opportunamente corretto dopo).')

                    except Exception as e:
                        dataframes_dict[trajectory][combination] = pd.DataFrame()
                        self.warn_count += 1
                        print(f'    [WARNING] Errore nel processo di analisi ed estrazione dati dal file di telemetria per la missione {trajectory} - {combination} --> {e}')
                        continue

                else:

                    # Si definiscono i nomi che identificano i file csv di iterazione
                    DVL_file_name = f'Telemetria_DVL_missione_{trajectory}_{combination}.csv'
                    INS_file_name = f'Telemetria_INS_missione_{trajectory}_{combination}.csv'

                    # Si definiscono i percorsi complessivi
                    DVL_csv = os.path.join(input_folder, DVL_file_name)
                    INS_csv = os.path.join(input_folder, INS_file_name)

                    try:

                        # Si leggono i dati contenuti nei due file tramite pandas e si generano due dataframe separati
                        DVL_dataframe = pd.read_csv(DVL_csv)
                        INS_dataframe = pd.read_csv(INS_csv)

                        if DVL_dataframe.empty or INS_dataframe.empty:
                            print(f"    [WARNING] Il DataFrame associato alla missione {trajectory} - {combination} risulta essere vuoto al momento dell'estrazione dati. Ricontrollare il file csv di origine.")
                            self.warn_count += 1
                            dataframes_dict[trajectory][combination] = pd.DataFrame()
                            continue

                        else:

                            # Si unificano i dataframe tramite concatenazione e li si ordina per timestamp crescente
                            mission_dataframe = pd.concat([DVL_dataframe, INS_dataframe], axis=0, ignore_index=True)        # Con axis=0 si impone di impilare i dataframe e non affiancarli mentre con ignore_index=True si unifica l'indice
                            mission_dataframe = mission_dataframe.sort_values('timestamp', ignore_index=True)                    # Si resetta l'indice dopo aver eseguito il sort

                            dataframes_dict[trajectory][combination] = mission_dataframe

                    except FileNotFoundError:
                        dataframes_dict[trajectory][combination] = pd.DataFrame()
                        self.warn_count += 1
                        print(f'    [WANRING] File di telemetria per la missione {trajectory} - {combination} non trovato.')
                        continue

                    except DtypeWarning:
                        DVL_dataframe = pd.read_csv(DVL_csv, low_memory=False)
                        INS_dataframe = pd.read_csv(INS_csv, low_memory=False)
                        mission_dataframe = pd.concat([DVL_dataframe, INS_dataframe], axis=0, ignore_index=True)
                        mission_dataframe = mission_dataframe.sort_values('timestamp', ignore_index=True)
                        dataframes_dict[trajectory][combination] = mission_dataframe
                        self.warn_count += 1
                        print(f'    [WARNING] Il file di telemetria della missione {trajectory} - {combination} contiene valori di tipo misto (problema opportunamente corretto dopo).')

                    except Exception as e:
                        dataframes_dict[trajectory][combination] = pd.DataFrame()
                        self.warn_count += 1
                        print(f'    [WARNING] Errore nel processo di unificazione e estrazione dati dai file di telemetria per la missione {trajectory} - {combination} --> {e}')
                        continue

            print(f"    Unificazione file csv per la traiettoria {trajectory} completata con {self.warn_count} warnings.")

        # -----------------------------------------------------
        #      GENERAZIONE DIZIONARIO DATAFRAME GLOBALE
        # -----------------------------------------------------

        # Si itera su ogni missione presente all'interno del dizionario
        for trajectory, mission_dict in dataframes_dict.items():

            print(f'\n  Estrazione dati di telemetria per missioni traiettoria {trajectory} in corso:')

            # Si salva il dataframe all'interno di un dizionario assegnandola alla singola missione
            if trajectory not in self.raw_database:
                self.raw_database[trajectory] = {}

            self.warn_count = 0
            for combination, mission_dataframe in mission_dict.items():

                # Check sul Dataframe --> se è vuoto c'è stato un problema nella fase di unificazione ed estrazione dei dati --> si segnala la missione nella lista di quelle non considerate
                if mission_dataframe.empty:
                    self.raw_database[trajectory][combination] = pd.DataFrame()
                    if (trajectory, combination) not in self.empty_missions_list:
                        self.empty_missions_list.append((trajectory, combination))
                    continue

                # Si ripulisce il dataframe eliminando eventuali caratteri non numerici (problema dati misti) e gli spazi associati a ogni stringa (per uniformità)
                mission_dataframe = self.dataframe_cleaning(mission_dataframe, trajectory, combination)

                # Si verifica che il dataframe sia rimasto non vuoto --> altrimenti c'è stato un problema nella pulizia del dataframe e si passa alla missione successiva --> si segnala la missione nell'apposita lista
                if mission_dataframe.empty:
                    self.raw_database[trajectory][combination] = pd.DataFrame()
                    if (trajectory, combination) not in self.empty_missions_list:
                        self.empty_missions_list.append((trajectory, combination))
                    continue
                mission_dataframe['nome'] = mission_dataframe['nome'].str.strip()

                try:

                    # Si converte la colonna timestamp del DataFrame in tempo effettivo, con formato orario classico 12:12:12:200, tramite pandas
                    mission_dataframe['timestamp'] = pd.to_datetime(mission_dataframe['timestamp'], unit='s')

                    # Si itera su ogni elemento presente all'interno del dizionario di nomenclatura
                    parts: list[pd.DataFrame] = []
                    for sensor_name, columns_name in self.data_labels.items():

                        # Si crea un sottoinsieme contenente tutte le righe del file .csv in cui compare il nome del sensore di iterazione (si usa .copy() per non modificare file originale)
                        sensor_data = mission_dataframe[mission_dataframe['nome'] == sensor_name].copy()

                        # Si sostituiscono i nomi generici originali (v1, v2...) con i nomi definiti nel dizionario per il sensore di iterazione
                        for i, new_name in enumerate(columns_name):
                            orig_col = f'v{i + 1}'
                            sensor_data = sensor_data.rename(columns={orig_col: new_name})

                        # Si aggiorna la sezione di dataframe associata a quelle colonne
                        sensor_data = sensor_data[['timestamp'] + columns_name]                                                             # Si eliminano le colonne non modificate nel ciclo --> sono colonne inutilizzate dal sensore (solo NaN come valori)

                        parts.append(sensor_data)

                    # Si concatenano le sezioni del dataframe definite per ogni sensore e si riordinano le misurazioni in ordine temporale crescente
                    new_mission_dataframe = pd.concat(parts, axis=0, ignore_index=True)
                    new_mission_dataframe = new_mission_dataframe.sort_values('timestamp', ignore_index=True)

                    self.raw_database[trajectory][combination] = new_mission_dataframe

                except Exception as e:
                    self.raw_database[trajectory][combination] = pd.DataFrame()
                    if (trajectory, combination) not in self.empty_missions_list:
                        self.empty_missions_list.append((trajectory, combination))
                    self.warn_count += 1
                    print(f'    [WARNING] Errore nel processo di formattazione del dataframe per la missione {trajectory} - {combination} (missione non considerata) --> {e}')
                    continue

            print(f'    I dataframe globali per la traiettoria {trajectory} sono stati generati correttamente, al netto di {self.warn_count} nuovi warnings rilevati (le eccezioni corrispondono a DataFrame vuoti nel dizionario).')

        return self

    # ======================================================
    #                PROCESSING DATAFRAMES
    # ======================================================

    # Funzione interna che permette di trasformare le coordinate del drone da valori NED locale ad assi body XY
    def NED_to_body(self, processing_method, trajectory, combination):

        try:

            # Si definisce il dataframe di iterazione partendo da quello contenuto all'interno del dizionario globale (per semplicità di notazione)
            dataframe_GPS = pd.DataFrame()
            dataframe_IMU = pd.DataFrame()
            if processing_method == 'resampling':
                # Se il sotto-dizionario associato alla missione risulta vuoto (problemi nelle fasi precedenti) si salva il DataFrame resampled come vuoto e si esce subito dalla funzione (si potrebbe eliminare avendo il salto di missione nella funzione principale)
                if not self.sensors_divided_database[trajectory][combination]:
                    return self
                dataframe_GPS = self.sensors_divided_database[trajectory][combination]['GPS'].copy()
                dataframe_IMU = self.sensors_divided_database[trajectory][combination]['IMU'].copy()
            elif processing_method == 'interpolazione':
                dataframe_GPS = self.interpolated_database[trajectory][combination].copy()
                dataframe_IMU = self.interpolated_database[trajectory][combination][['Pitch [deg]', 'Yaw [deg]']].copy()

            if not dataframe_GPS.empty and not dataframe_IMU.empty:

                # Si verifica che le colonne interessate siano presenti --> nomi dati in formattazione_csv
                if 'Delta_East [m]' in dataframe_GPS.columns and 'Delta_North [m]' in dataframe_GPS.columns:

                    if 'Yaw [deg]' in dataframe_IMU.columns and 'Pitch [deg]' in dataframe_IMU.columns:

                        # Si converte il valore dello yaw e del pitch preso per ogni istante in radianti
                        yaw_rad = np.radians(dataframe_IMU['Yaw [deg]'])
                        pitch_rad = np.radians(dataframe_IMU['Pitch [deg]'])

                        # Si richiamano i valori di spostamento relativo
                        Delta_East = dataframe_GPS['Delta_East [m]']
                        Delta_North = dataframe_GPS['Delta_North [m]']

                        # Si effettua la trasformazione in assi body
                        dataframe_GPS['X_body_ist [m]'] = Delta_North * np.cos(yaw_rad) + Delta_East * np.sin(yaw_rad)
                        dataframe_GPS['Y_body_ist [m]'] = -Delta_North * np.sin(yaw_rad) + Delta_East * np.cos(yaw_rad)

                        # Si salvano le modifiche all'interno del dataframe originale
                        if processing_method == 'resampling':
                            self.sensors_divided_database[trajectory][combination]['GPS'] = dataframe_GPS
                        elif processing_method == 'interpolazione':
                            self.interpolated_database[trajectory][combination] = dataframe_GPS

                    else:
                        self.warn_count += 1
                        print(f"    [WARNING] Non trovata nessuna colonna con nome 'Yaw [deg]' o 'Pitch [deg]' nel dataframe della missione {trajectory} - {combination}.")

                else:
                    self.warn_count += 1
                    print(f"    [WARNING] Non trovata nessuna colonna con nome 'Delta_East [m]' o 'Delta_North [m]' nel dataframe della missione {trajectory} - {combination}.")

            else:
                self.warn_count += 1
                print(f"    [WARNING]  Il dataframe della missione {trajectory} - {combination} risulta vuoto.")

        except KeyError:
            raise KeyError(f"    [ERROR] Errore nella conversione delle coordinate NED -> Body per la missione {trajectory} - {combination}. Ricontrollare la corrispondenza delle etichette usate.") from None

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nella trasformazione delle coordinate NED -> Body per la missione {trajectory} - {combination} --> {e}")

        return self

    # Funzione interna che permette di trasformare le coordinate del drone da valori UTM ad assi NED
    def UTM_to_NED(self, processing_method, trajectory, combination):

        try:

            # Si definisce il dataframe da modificare in base al metodo di processing del dataframe utilizzato
            dataframe = pd.DataFrame()
            if processing_method == 'resampling':
                # Se il sotto-dizionario associato alla missione risulta vuoto (problemi nelle fasi precedenti) si salva il DataFrame resampled come vuoto e si esce subito dalla funzione (si potrebbe eliminare avendo il salto di missione nella funzione principale)
                if not self.sensors_divided_database[trajectory][combination]:
                    return self
                dataframe = self.sensors_divided_database[trajectory][combination]['GPS'].copy()
            elif processing_method == 'interpolazione':
                dataframe = self.interpolated_database[trajectory][combination].copy()

            if not dataframe.empty:

                # Si verifica che le colonne interessate siano presenti --> nomi dati in formattazione_csv
                if 'UTM_East [m]' in dataframe.columns and 'UTM_North [m]' in dataframe.columns:

                    # Si definiscono i valori iniziali delle coordinate UTM
                    east0 = dataframe['UTM_East [m]'].dropna().iloc[0]
                    north0 = dataframe['UTM_North [m]'].dropna().iloc[0]

                    # Si sottrae a ogni valore quello iniziale --> si ottengono così spostamenti in metri rispetto al punto iniziale, a prescindere di dove questo sia
                    dataframe['NED_East [m]'] = dataframe['UTM_East [m]'] - east0
                    dataframe['NED_North [m]'] = dataframe['UTM_North [m]'] - north0

                    # Si calcolano le differente per ogni step temporale da fornire alla rete come input
                    delta_east = dataframe['NED_East [m]'].diff()
                    delta_north = dataframe['NED_North [m]'].diff()

                    # Si imposta il primo spostamento a 0 --> altrimenti sarebbe NaN --> non si usa il .fillna(0) per evitare di trasformare anche i NaN interni al dataframe in 0
                    delta_east.iloc[0] = 0
                    delta_north.iloc[0] = 0

                    # Si aggiunge la colonna degli spostamenti al dataframe originale
                    dataframe['Delta_East [m]'] = delta_east
                    dataframe['Delta_North [m]'] = delta_north

                    # Si salvano le modifiche all'interno del dataframe originale
                    if processing_method == 'resampling':
                        self.sensors_divided_database[trajectory][combination]['GPS'] = dataframe
                    elif processing_method == 'interpolazione':
                        self.interpolated_database[trajectory][combination] = dataframe

                else:
                    self.warn_count += 1
                    print(f"    [WARNING] Non trovata nessuna colonna con nome 'UTM_East' o 'UTM_North' nel dataframe della missione {trajectory} - {combination}.")

            else:
                self.warn_count += 1
                print(f"    [WARNING] Il dataframe per la missione {trajectory} - {combination} risulta vuoto.")

        except KeyError:
            raise KeyError(f"    [ERROR] Errore nella conversione delle coordinate NED -> Body per la missione {trajectory} - {combination}. Ricontrollare la corrispondenza delle etichette usate.")

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nella trasformazione delle coordinate UTM -> NED per la missione {trajectory} - {combination} --> {e}")

        return self

    # Funzione interna in grado di ricevere in input la missione di riferimento traiettoria-combinazione e il dizionario con i dataframe per ogni sensore e produrre in output un dataframe a passo temporale uniforme
    def dataframe_resampling(self, trajectory, combination):

        try:

            # Se il sotto-dizionario associato alla missione risulta vuoto (problemi nelle fasi precedenti) si salva il DataFrame resampled come vuoto e si esce subito dalla funzione (si potrebbe eliminare avendo il salto di missione nella funzione principale)
            if not self.sensors_divided_database[trajectory][combination]:
                self.resampled_database[trajectory][combination] = pd.DataFrame()
                return self

            # Si inizializza il dataframe di output (comune a tutti i sensori) come il dataframe legato al solo DVL --> questo fornisce il riferimento temporale, avendo la frequenza di aggiornamento maggiore
            resampled_dataframe = self.sensors_divided_database[trajectory][combination]['DVL']

            # Si itera su ogni sensore presente all'interno del dizionario prodotto in precedenza --> se anche il dizionario fosse vuoto lo salterebbe non avendo items()
            for sensor_name, sensor_dataframe in self.sensors_divided_database[trajectory][combination].items():

                # Si effettua l'operazione di merge basata sui timestamp dei vari blocchi creati --> si impostano a NaN i valori delle righe che non hanno corrispondenza nel DataFrame di base (primo argomento funzione)
                if sensor_name != 'DVL':
                    resampled_dataframe = pd.merge(resampled_dataframe, sensor_dataframe, on='timestamp', how='outer')

            # Si riordina il dataframe prodotto secondo il timestamp, per maggiore sicurezza, creando anche un nuovo indice
            resampled_dataframe = resampled_dataframe.sort_values('timestamp', ignore_index=True)

            # Si aggiunge il dataframe generato per la missione al dizionario complessivo
            self.resampled_database[trajectory][combination] = resampled_dataframe

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nel processo di resampling per la missione {trajectory} - {combination} --> {e}")
            self.resampled_database[trajectory][combination] = pd.DataFrame()

        return self

    # Funzione incaricata di analizzare i dataframe di tutte le missioni per generare dei dataframe contenenti solo le colonne di ogni blocco di sensori da usare per addestramento
    def raw_dataframes_processing(self, generate_dataframe, save_file_path, sensors_frequencies):

        # -----------------------------------------------------
        #             CARICAMENTO DATABASE GLOBALE
        # -----------------------------------------------------

        # Se la scelta dell'utente è no, il dataframe globale si prende dal file pickle
        if generate_dataframe == 'N':

            try:
                with open(save_file_path, 'rb') as f:
                    dataframe_globale = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Se la scelta è si il dataframe globale si prende direttamente da quello generato con la lettura dei file csv
        elif generate_dataframe == 'S':

            # Si crea una copia del dizionario contenente il dataframe globale --> in modo da non modificare quello originale per errore
            dataframe_globale = self.raw_database.copy()

        # -----------------------------------------------------
        #             INIZIO PROCESSING DATABASE
        # -----------------------------------------------------

        # Si itera per ogni elemento contenuto nel dizionario separando il nome del foglio (= nome_missione) e il dataframe associato
        for trajectory, combinations_dict in dataframe_globale.items():

            print(f'\n  Analisi dei dataframe associati alla traiettoria {trajectory} in corso:')

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.sensors_divided_database:
                self.sensors_divided_database[trajectory] = {}

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.resampled_database:
                self.resampled_database[trajectory] = {}

            self.warn_count = 0
            for combination, mission_dataframe in combinations_dict.items():

                # Se c'è stato un problema nella csv_analysis si salta l'analisi della missione e si inizializzano a dizionario nullo e Dataframe vuoto gli elementi associati nei rispettivi dizionari
                if mission_dataframe.empty:
                    self.sensors_divided_database[trajectory][combination] = {}
                    self.resampled_database[trajectory][combination] = pd.DataFrame()
                    continue

                # Si inizializzano le liste necessarie al trattamento del dataframe di missione
                df_blocks_list = []                 # Lista che conterrà il dataframe di missione ridotto ai soli parametri necessari e suddiviso in blocchi (uno per sensore)
                t_start_list = []                   # Lista che conterrà i timestamp di inizio misurazioni di ogni sensore
                t_end_list = []                     # Lista che conterrà i timestamp di fine misurazioni di ogni sensore

                # Si crea il dizionario interno associato alla singola missione all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
                if combination not in self.sensors_divided_database[trajectory]:
                    self.sensors_divided_database[trajectory][combination] = {}

                # -----------------------------------------------------
                #        DEFINIZIONE LIMITI TEMPORALI DATAFRAMES
                # -----------------------------------------------------

                # Si itera su ogni blocco di sensori individuato per definire istante di inizio e fine di ognuno
                try:
                    for sensor_block_name, sensor_block_cols in self.sensors_labels_dict.items():

                        # Si crea un sottoinsieme del dataframe completo originale che contenga solo le colonne del blocco di iterazione --> si eliminano le righe in cui sono presenti valori NaN (utile per definizione istante iniziale)
                        dataframe_copy = mission_dataframe[sensor_block_cols].dropna(how='any')

                        # Si riordinano i valori in base al timestamp e lo si salva nella lista complessiva
                        dataframe_copy = dataframe_copy.sort_values('timestamp')
                        df_blocks_list.append(dataframe_copy)

                        # Calcolo istante iniziale e conclusivo del dataframe
                        t_start_list.append(dataframe_copy['timestamp'].min())
                        t_end_list.append(dataframe_copy['timestamp'].max())

                    # Si selezionano gli istanti di inizio e fine del dataframe aggiornato --> usati ceil e floor per arrotondare in modo tale da ridurre i NaN all'inizio (se l'approssimazione prendesse ad esempio un .800 per un .834 di misurazione)
                    t_start_dataframe = max(t_start_list).ceil(self.freq)        # Il dataframe inizia dal valore più alto tra gli istanti iniziali dei singoli blocchi --> così non occorre ipotizzare grandezze con metodo backward
                    t_end_dataframe = min(t_end_list).floor(self.freq)           # Il dataframe finisce con il valore più basso tra gli istanti finali dei singoli blocchi --> così riduco il numero di stati ipotizzati (ma solitamente è GPS a finire prima)

                    # Si determina l'istante finale effettivo del GPS a seguito dell'arrotondamento con griglia --> sarà questo il riferimento per tutte le altre griglia
                    GPS_freq = sensors_frequencies['GPS']
                    t_end_GPS = pd.date_range(start=t_start_dataframe, end=t_end_dataframe, freq=f'{(1/GPS_freq)*1000}ms').max()

                except KeyError:
                    raise KeyError(f"    [ERROR] Ci sono chiavi associate ai sensori errate per la traiettoria {trajectory}. Ricontrollare la corrispondenza delle etichette usate.") from None

                # -----------------------------------------------------
                #           MERGE SU GRIGLIA TEMPORALE COMUNE
                # -----------------------------------------------------

                # Si itera contemporaneamente sulla lista dei dataframe per sensori e sulle tuple dei blocchi per fare effettivamente il resampling
                sensors_dataframes = {}
                mission_failed = False
                for sensor_block_dataframe, (sensor_block_name, sensor_block_cols) in  zip(df_blocks_list, self.sensors_labels_dict.items()):

                    try:

                        # Check sulla variabile booleana --> se c'è già stato un errore per un sensore è inutile analizzare anche gli altri che verranno scartati ugualmente
                        if mission_failed:
                            continue

                        frequenza = sensors_frequencies[sensor_block_name]

                        # Si genera la griglia temporale con passo uniforme associata al blocco di sensori --> segue la frequenza dei dati
                        griglia_uniforme = pd.DataFrame({'timestamp': pd.date_range(start=t_start_dataframe, end=t_end_GPS, freq=f'{(1/frequenza)*1000}ms')})

                        # Si effettua il merge tra le misurazioni del sensore considerato e la griglia temporale --> usato metodo backward per garantire una maggiore attinenza con misurazioni real-time sul drone
                        if sensor_block_name == 'GPS':
                            sensor_block_dataframe = pd.merge_asof(griglia_uniforme, sensor_block_dataframe, on='timestamp', direction='nearest')
                        else:
                            sensor_block_dataframe = pd.merge_asof(griglia_uniforme, sensor_block_dataframe, on='timestamp', direction='backward')

                        # Si aggiunge il dataframe relativo al singolo blocco di sensori considerati al dizionario locale
                        sensors_dataframes[sensor_block_name] = sensor_block_dataframe

                    except Exception as e:
                        self.warn_count += 1
                        mission_failed = True
                        print(f"    [WARNING] Errore nella fase di unificazione griglia temporale per il blocco sensori {sensor_block_name} nella missione {trajectory} - {combination} (l'intera missione verrà scartata)--> {e}")
                        continue

                # Si aggiorna il dizionario scomposto globale unicamente se tutti i sensori presenti sono stati analizzati correttamente --> altrimenti nascerebbero problemi di dimensionalità delle batch con la rete neurale --> si effettua, inoltre, un continue passando immediatamente alla missione successiva
                if mission_failed:
                    self.sensors_divided_database[trajectory][combination] = {}
                    self.resampled_database[trajectory][combination] = pd.DataFrame()
                    if (trajectory, combination) not in self.empty_missions_list:
                        self.empty_missions_list.append((trajectory, combination))
                    continue
                else:
                    self.sensors_divided_database[trajectory][combination] = sensors_dataframes

                # -----------------------------------------------------
                #              CONVERSIONE COORDINATE UTM
                # -----------------------------------------------------

                self.UTM_to_NED('resampling', trajectory, combination)
                self.NED_to_body('resampling', trajectory, combination)

                # -----------------------------------------------------
                #            RESAMPLING IN DATAFRAME UNICO
                # -----------------------------------------------------

                self.dataframe_resampling(trajectory, combination)

            print(f'    Analisi delle missioni associate alla {trajectory} completata con {self.warn_count} nuovi warnings rilevati.')

        return self

    # Funzione incaricata di generare il dizionario dei dataframe per ogni missione con dati trattati tramite interpolazione --> parte dal dataframe puro
    def dataframe_interpolation(self, generate_dataframe, save_file_path):

        # -----------------------------------------------------
        #             CARICAMENTO DATABASE GLOBALE
        # -----------------------------------------------------

        # Se la scelta dell'utente è no, il dataframe globale si prende dal file pickle
        if generate_dataframe == 'N':

            try:
                with open(save_file_path, 'rb') as f:
                    raw_database = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Se la scelta è si il dataframe globale si prende direttamente da quello generato con la lettura dei file csv
        elif generate_dataframe == 'S':

            # Si crea una copia del dizionario contenente il dataframe globale --> in modo da non modificare quello originale per errore
            raw_database = self.raw_database.copy()

        # -----------------------------------------------------
        #             INIZIO PROCESSING DATABASE
        # -----------------------------------------------------

        # Si itera per ogni elemento contenuto nel dizionario separando il nome del foglio (= nome_missione) e il dataframe associato
        for trajectory, combinations_dict in raw_database.items():

            print(f'\n  Interpolazione dei dataframe associati alla traiettoria {trajectory} in corso:')

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.interpolated_database:
                self.interpolated_database[trajectory] = {}

            self.warn_count = 0
            for combination, mission_dataframe in combinations_dict.items():

                try:

                    # Se il DataFrame globale risulta vuoto (problema nella csv_analysis) si salta l'analisi e si impone la presenza di un DataFrame interpolato vuoto --> non si aggiorna il counter del warning perché non è un vero problema di interpolazione
                    if mission_dataframe.empty:
                        self.interpolated_database[trajectory][combination] = pd.DataFrame()
                        continue

                    # Si inizializza una copia del dataframe originale e si imposta come indice la colonna dei timestamp --> utile poiché usando il metodo di interpolazione lineare si tiene conto anche di misurazioni non equispaziate
                    interpolated_dataframe = mission_dataframe.set_index('timestamp')

                    # Si effettua un'operazione di unwrap delle grandezze angolari --> in questo modo si evitano problemi per quanto riguarda il passaggio da 0 a 360 e viceversa, che l'interpolazione non gestisce in maniera continua
                    IMU_cols = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
                    IMU_mask = interpolated_dataframe[IMU_cols].notna().all(axis=1)
                    interpolated_dataframe.loc[IMU_mask, IMU_cols] = np.unwrap(interpolated_dataframe.loc[IMU_mask, IMU_cols].to_numpy(), period=360, axis=0)

                    # Si effettua l'interpolazione lineare
                    interpolated_dataframe = interpolated_dataframe.interpolate(method='time', limit_area='inside')

                    # Si scalano le misurazioni angolari per stare nell0intervallo 0-360 classico --> si applica solo allo yaw perché rollio e pitch hanno anche valori negativi, quindi quest'operazione potrebbe corrompere i normali valori
                    interpolated_dataframe['Yaw [deg]'] = interpolated_dataframe['Yaw [deg]'] % 360

                    # Si eliminano le righe della tabella in cui non sono presenti i dati di tutti i sensori contemporaneamente --> di solito gli ultimi ad avviarsi sono axisref e motorref
                    interpolated_dataframe.dropna(how='any', inplace=True)

                    # Si ripristina il timestamp come colonna e non come indice
                    interpolated_dataframe.reset_index(inplace=True)

                    # Si eliminano le colonne non utili (DVL_Lock, Quality e Zone per GPS)
                    interpolated_dataframe.drop(columns=['DVL_Lock', 'Mot_FW3 [%]', 'Mot_FW4 [%]', 'Zone', 'Quality'], inplace=True)

                    # Si aggiunge il dataframe creato al dizionario complessivo
                    self.interpolated_database[trajectory][combination] = interpolated_dataframe

                except KeyError:
                    raise KeyError(f"    [ERROR] Errore nel drop delle colonne non utili. Ricontrollare la corrispondenza delle etichette usate.") from None

                except Exception as e:
                    self.interpolated_database[trajectory][combination] = pd.DataFrame()
                    self.warn_count += 1
                    print(f"    [WARNING] Errore nell'interpolazione della missione {trajectory} - {combination} --> {e}")
                    continue

                # Si applicano le trasformazioni in assi NED e assi body per ottenere il dataframe completo
                self.UTM_to_NED('interpolazione', trajectory, combination)
                self.NED_to_body('interpolazione', trajectory, combination)

                # Si eliminano le colonne associate alla posizione UTM, ormai inutili
                try:
                    if self.interpolated_database[trajectory][combination].empty:
                        if (trajectory, combination) not in self.empty_missions_list:
                            self.empty_missions_list.append((trajectory, combination))
                        continue
                    #else:
                        #self.interpolated_database[trajectory][combination].drop(columns=['UTM_East [m]', 'UTM_North [m]'], inplace=True)
                except KeyError:
                    raise KeyError(f"    [ERROR] Errore nel drop delle colonne non utili. Ricontrollare la corrispondenza delle etichette usate.") from None

            print(f"    Interpolazione dataframes associati alla traiettoria {trajectory} effettuata con {self.warn_count} nuovi warnings rilevati.")

        return self

    # ======================================================
    #            GESTIONE SALVATAGGIO DATAFRAMES
    # ======================================================

    # Funzione per il salvataggio del dataframe generato dalla funzione csv_analysis (nel formato di salvataggio opportuno)
    def save_raw_dataframes(self, formato, ROOT_DIR):

        # Se il formato è pickle si salva il dizionario globale dei dataframe per un uso futuro nella rete neurale
        if formato == 'pickle':

            print(f'\n  Salvataggio del file in formato pickle contenente il database in corso:')

            try:

                save_file_path = os.path.join(ROOT_DIR, 'Dataset_globale.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.raw_database, f)

                    print(f'    Salvataggio del dataframe in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.raw_database.items():

                print(f'  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_raw.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            if dataframe.empty:
                                continue

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = self.raw_database[trajectory][combination].copy()

                            # Formattazione per file Excel --> necessario creare una nuova colonna con i tempi per sostituire quella originale, che approssima male gli istanti di tempo rendendo l'analisi incomprensibile
                            dataframe_copy['timestamp'] = dataframe_copy['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]

                            # Si definisce il nome del foglio della missione e si crea all'interno del file Excel di traiettoria
                            nome_foglio = f"Missione_{combination}"
                            dataframe_copy.to_excel(writer, index=False, sheet_name=nome_foglio)

                            # Si apre la zona di modifica
                            workbook = writer.book
                            worksheet = writer.sheets[nome_foglio]

                            # Definizione manuale dei colori per i blocchi
                            formato_dvl = workbook.add_format({'bg_color': '#D9EAD3', 'border': 1})     # Verde chiaro
                            formato_att = workbook.add_format({'bg_color': '#CFE2F3', 'border': 1})     # Blu chiaro
                            formato_dep = workbook.add_format({'bg_color': '#F4CCCC', 'border': 1})     # Rosso chiaro
                            formato_check = workbook.add_format({'bg_color': '#E8E8E8', 'border': 1})   # Scala grigi
                            formato_axis = workbook.add_format({'bg_color': '#DCF4EF', 'border': 1})    # Verde - turchese
                            formato_mot = workbook.add_format({'bg_color': '#FFF2CC', 'border': 1})     # Giallo chiaro
                            formato_GPS = workbook.add_format({'bg_color': '#FF8C00', 'border': 1})     # Dark orange

                            # Applicazione colori associati alle colonne divise per sensori
                            worksheet.set_column('A:A', 17)                     # Timestamp
                            worksheet.set_column('B:F', 17, formato_dvl)        # Blocco misure DVL
                            worksheet.set_column('G:I', 17, formato_att)        # Blocco misure attitude
                            worksheet.set_column('J:K', 17, formato_dep)        # Blocco misura profondità e rateo di discesa
                            #worksheet.set_column('L:Q', 17, formato_check)      # Blocco misure sensori di controllo come alt e pressione interna
                            worksheet.set_column('L:Q', 17, formato_axis)       # Blocco misure axis_ref
                            worksheet.set_column('R:Y', 17, formato_mot)        # Blocchi misure dei motor_ref
                            worksheet.set_column('Z:AC', 19, formato_GPS)       # Blocchi misure del GPS

                            # Verifica della corretta scrittura della telemetria della missione analizzata
                            if nome_foglio in writer.sheets and len(dataframe_copy) > 0:
                                print(f'    La tabella in Excel per la missione {trajectory} - {combination} è stata generata correttamente con {len(dataframe_copy)} campioni.')
                            else:
                                print(f'    [ERRORE]: mancata generazione del foglio {nome_foglio}.\n')

                except Exception as e:
                    print(f'    [WARNING] Errore nel salvataggio del database su file Excel per  {trajectory} --> {e}\n')

    # Funzione che permette il salvataggio in file Excel del dataframe a seguito di pre-processing
    def save_dataframe_resampled(self, formato, ROOT_DIR):

        # Se il formato è pickle si salva il dizionario globale dei dataframe per un uso futuro nella rete neurale
        if formato == 'pickle':

            print(f'\n  Salvataggio del file in formato pickle contenente il database resampled in corso:')

            try:

                save_file_path = os.path.join(ROOT_DIR, 'Dataset_resampled.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.resampled_database, f)

                    print(f'    Salvataggio del dataframe resampled in file {save_file_path} eseguito correttamente.')

                save_file_path = os.path.join(ROOT_DIR, 'Dataset_resampled_blocchi.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.sensors_divided_database, f)

                    print(f'    Salvataggio del dataframe suddiviso per sensore in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.resampled_database.items():

                print(f'\n  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} trattate con resampling in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_resampled.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            if dataframe.empty:
                                continue

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = self.resampled_database[trajectory][combination].drop(columns=['UTM_East [m]', 'UTM_North [m]'])

                            # Formattazione per file Excel --> necessario creare una nuova colonna con i tempi per sostituire quella originale, che approssima male gli istanti di tempo rendendo l'analisi incomprensibile
                            dataframe_copy['timestamp'] = dataframe_copy['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]

                            # Si definisce il nome del foglio della missione e si crea all'interno del file Excel di traiettoria
                            nome_foglio = f"Missione_{combination}"
                            dataframe_copy.to_excel(writer, index=False, sheet_name=nome_foglio)

                            # Si apre la zona di modifica
                            workbook = writer.book
                            worksheet = writer.sheets[nome_foglio]

                            # Definizione manuale dei colori per i blocchi
                            formato_dvl = workbook.add_format({'bg_color': '#D9EAD3', 'border': 1})     # Verde chiaro
                            formato_att = workbook.add_format({'bg_color': '#CFE2F3', 'border': 1})     # Blu chiaro
                            formato_dep = workbook.add_format({'bg_color': '#F4CCCC', 'border': 1})     # Rosso chiaro
                            #formato_check = workbook.add_format({'bg_color': '#E8E8E8', 'border': 1})   # Scala grigi
                            formato_axis = workbook.add_format({'bg_color': '#DCF4EF', 'border': 1})    # Verde - turchese
                            formato_mot = workbook.add_format({'bg_color': '#FFF2CC', 'border': 1})     # Giallo chiaro
                            formato_GPS = workbook.add_format({'bg_color': '#FF8C00', 'border': 1})     # Dark orange

                            # Applicazione colori associati alle colonne divise per sensori
                            worksheet.set_column('A:A', 17)                 # Timestamp
                            worksheet.set_column('B:E', 17, formato_dvl)    # Blocco misure DVL
                            worksheet.set_column('F:H', 17, formato_att)    # Blocco misure attitude
                            worksheet.set_column('I:J', 17, formato_dep)    # Blocco misura profondità e rateo di discesa
                            #worksheet.set_column('K:L', 17, formato_check)  # Blocco misure sensori di controllo come alt e pressione interna
                            worksheet.set_column('K:P', 17, formato_mot)    # Blocchi misure dei motor_ref
                            worksheet.set_column('Q:V', 17, formato_axis)   # Blocco misure axis_ref
                            worksheet.set_column('W:AB', 19, formato_GPS)   # Blocchi misure del GPS

                            # Verifica della corretta scrittura della telemetria della missione analizzata
                            if nome_foglio in writer.sheets and len(dataframe_copy) > 0:
                                print(f'    La tabella in Excel per la missione {trajectory} - {combination} è stata generata correttamente con {len(dataframe_copy)} campioni.')
                            else:
                                print(f'    [ERRORE]: mancata generazione del foglio {nome_foglio}.\n')

                except Exception as e:
                    print(f'    [WARNING] Errore nel salvataggio del database su file Excel per  {trajectory} --> {e}\n')

    # Funzione che permette il salvataggio in file Excel del dataframe a seguito di pre-processing
    def save_dataframe_interpolated(self, formato, ROOT_DIR):

        # Se il formato è pickle si salva il dizionario globale dei dataframe per un uso futuro nella rete neurale
        if formato == 'pickle':

            print(f'\n  Salvataggio del file in formato pickle contenente il database interpolated in corso:')

            try:

                save_file_path = os.path.join(ROOT_DIR, 'Dataset_interpolated.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.interpolated_database, f)

                    print(f'    Salvataggio del dataframe in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.interpolated_database.items():

                print(f'\n  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} trattate con interpolazione in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_interpolated.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            if dataframe.empty:
                                continue

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = dataframe.drop(columns=['UTM_East [m]', 'UTM_North [m]'])

                            # Formattazione per file Excel --> necessario creare una nuova colonna con i tempi per sostituire quella originale, che approssima male gli istanti di tempo rendendo l'analisi incomprensibile
                            dataframe_copy['timestamp'] = dataframe_copy['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]

                            # Si definisce il nome del foglio della missione e si crea all'interno del file Excel di traiettoria
                            nome_foglio = f"Missione_{combination}"
                            dataframe_copy.to_excel(writer, index=False, sheet_name=nome_foglio)

                            # Si apre la zona di modifica
                            workbook = writer.book
                            worksheet = writer.sheets[nome_foglio]

                            # Definizione manuale dei colori per i blocchi
                            formato_dvl = workbook.add_format({'bg_color': '#D9EAD3', 'border': 1})     # Verde chiaro
                            formato_att = workbook.add_format({'bg_color': '#CFE2F3', 'border': 1})     # Blu chiaro
                            formato_dep = workbook.add_format({'bg_color': '#F4CCCC', 'border': 1})     # Rosso chiaro
                            formato_check = workbook.add_format({'bg_color': '#E8E8E8', 'border': 1})   # Scala grigi
                            formato_axis = workbook.add_format({'bg_color': '#DCF4EF', 'border': 1})    # Verde - turchese
                            formato_mot = workbook.add_format({'bg_color': '#FFF2CC', 'border': 1})     # Giallo chiaro
                            formato_GPS = workbook.add_format({'bg_color': '#FF8C00', 'border': 1})     # Dark orange

                            # Applicazione colori associati alle colonne divise per sensori
                            worksheet.set_column('A:A', 17)                     # Timestamp
                            worksheet.set_column('B:E', 17, formato_dvl)        # Blocco misure DVL
                            worksheet.set_column('F:H', 17, formato_att)        # Blocco misure attitude
                            worksheet.set_column('I:J', 17, formato_dep)        # Blocco misura profondità e rateo di discesa
                            #worksheet.set_column('K:L', 17, formato_check)      # Blocco misure sensori di controllo come alt e pressione interna
                            worksheet.set_column('K:P', 17, formato_axis)       # Blocco misure axis_ref
                            worksheet.set_column('O:V', 17, formato_mot)        # Blocchi misure dei motor_ref
                            worksheet.set_column('W:AB', 19, formato_GPS)       # Blocchi misure del GPS

                            # Verifica della corretta scrittura della telemetria della missione analizzata
                            if nome_foglio in writer.sheets and len(dataframe_copy) > 0:
                                print(f'    La tabella in Excel per la missione {trajectory} - {combination} è stata generata correttamente con {len(dataframe_copy)} campioni.')
                            else:
                                print(f'    [ERRORE]: mancata generazione del foglio {nome_foglio}.\n')

                except Exception as e:
                    print(f'    [WARNING] Errore nel salvataggio del database su file Excel per  {trajectory} --> {e}\n')

    # ======================================================
    #         TRASFORMAZIONE DATAFRAMES IN TENSORI
    # ======================================================

    # Funzione interna che permette di gestire l'eliminazione delle colonne dal database in vista dell'uso sulla rete neurale
    @staticmethod
    def drop_columns(sensor_name, sensor_dataframe):

        sensor_dataframe = sensor_dataframe.drop(columns=['timestamp'], errors='ignore')

        # Si eliminano le colonne non necessarie per il training
        if sensor_name == 'GPS':
            sensor_dataframe = sensor_dataframe.drop(columns=['UTM_North [m]', 'UTM_East [m]', 'NED_North [m]', 'NED_East [m]', 'X_body_ist [m]', 'Y_body_ist [m]'], errors='ignore')

        elif sensor_name == 'DVL':
            sensor_dataframe = sensor_dataframe.drop(columns=['DVL_Altitude [m]'], errors='ignore')

        elif sensor_name == 'V_ref':
            sensor_dataframe = sensor_dataframe.drop(columns=['Ref_Vy [%]', 'Ref_Vz [%]', 'Ref_ωx [%]', 'Ref_ωz [%]'], errors='ignore')

        elif sensor_name == 'MOT':
            sensor_dataframe = sensor_dataframe.drop(columns=['Mot_RV [%]', 'Mot_RL [%]'], errors='ignore')

        #elif sensor_name == 'IMU':
            #yaw_rad = np.radians(sensor_dataframe['Yaw [deg]'])
            #sensor_dataframe = sensor_dataframe.drop(columns=['Yaw [deg]'], errors='ignore')
            #sensor_dataframe['Yaw_sin'] = np.sin(yaw_rad)
            #sensor_dataframe['Yaw_cos'] = np.cos(yaw_rad)

        return sensor_dataframe

    # Funzione interna incaricata di calcolare il valore medio e la deviazione standard per ogni colonna appartenente a un blocco di sensori
    def compute_normalization_parameters(self, sensor_blocks_dataframes, missioni_test):

        # -----------------------------------------------------
        #         UNIFICAZIONE DATAFRAMES PER SENSORE
        # -----------------------------------------------------

        # Si itera su ogni elemento del dizionario di dataframe generato per i blocchi --> obiettivo è creare un dizionario che contiene una lista delle sezioni di dataframe di tutte le missioni per ogni sensore
        sensor_dataframes_list = {}
        for trajectory, mission_dict in sensor_blocks_dataframes.items():

            self.warn_count = 0
            for combination, sensors_dict in mission_dict.items():

                # Check sulla presenza di un sotto-dizionario non vuoto associato alla missione --> altrimenti c'è stato un errore all'interno del processing --> si genera un tensore vuoto e si passa oltre senza gestirlo al momento
                if not sensors_dict:
                    continue

                # Si utilizza la missione per la definizione dei valori di normalizzazione solo per le missioni di training --> poi si applicano direttamente a quelle di test per garantire che la rete non abbia in alcun modo informazioni sulle misurazioni future nella fase di test
                if (trajectory, combination) not in missioni_test:

                    for sensor_name, sensor_dataframe in sensors_dict.items():

                        # Si verifica di non avere già riscontrato un problema con il sensore in una missione precedente --> altrimenti si continua senza rieseguire l'analisi
                        if sensor_name in sensor_dataframes_list and sensor_dataframes_list[sensor_name] is None:
                            continue

                        try:

                            sensor_dataframe_copy = self.drop_columns(sensor_name, sensor_dataframe)

                            # Si aggiunge il dataframe per la missione e il nome del sensore associato a un apposito dizionario --> nella forma {'IMU': [], 'DVL': [], 'GPS': []}
                            if sensor_name not in sensor_dataframes_list:
                                sensor_dataframes_list[sensor_name] = []
                            sensor_dataframes_list[sensor_name].append(sensor_dataframe_copy)

                        except Exception as e:
                            self.warn_count += 1
                            print(f"    [WARNING] Errore nell'unificazione dei dataframe per sensore {sensor_name} --> {e}")
                            continue


        # -----------------------------------------------------
        #           CALCOLO PARAMETRI DI OGNI SENSORE
        # -----------------------------------------------------

        # Si itera ora sul nuovo dizionario --> un blocco di sensori alla volta in sostanza
        for sensor_name, dataframes_list in sensor_dataframes_list.items():

            if dataframes_list is not None:

                try:

                    # Si genera un dataframe unico unendo quelli di tutte le missioni contenuti nella lista senza la colonna 'timestamp' (non necessaria)
                    dataframe = pd.concat(dataframes_list)

                    # Si calcola il valore della media e della deviazione standard sulla totalità delle missioni considerate
                    mean_value = dataframe.mean().to_numpy(dtype=np.float32)           # Si usa .means() per la media, .values per convertire il tutto in valori numerici (array numpy)
                    std = dataframe.std().to_numpy(dtype=np.float32)                   # Si usa .std() per la deviazione, .values per convertire il tutto in valori numerici (array numpy)

                    # Si gestiscono separatamente le colonne con seno e coseno dello yaw --> non vanno normalizzate
                    for col_idx, col_name in enumerate(dataframe.columns):
                        if col_name in ['Yaw_sin', 'Yaw_cos']:
                            mean_value[col_idx] = 0.0
                            std[col_idx] = 1.0

                    # Si aggiorna il dizionario aggiungendo al blocco del sensore i valori trovati
                    self.normalization_parameters_dict[sensor_name] = {'media': mean_value, 'deviazione standard': std}

                    # Si estrae il valore di deviazione standard per verificare sia una misura valida
                    sigma = self.normalization_parameters_dict[sensor_name]['deviazione standard']

                    # Se la colonna del sensore dovesse avere problemi (std nulla o inferiore a una soglia minima) si genererebbero valori normalizzati enormi --> si impone quindi il valore di std = 1 (colonna rimane costante e non disturba la rete) --> utile soprattutto per il Roll che varia pochissimo in tutta la missione
                    degenerate_cols = ~np.isfinite(sigma) | (sigma < self.min_std)
                    if degenerate_cols.any():
                        print(f"    [WARNING] Blocco {sensor_name}: std inferiore a {self.min_std} per le colonne {list(dataframe.columns[degenerate_cols])} (valori: {sigma[degenerate_cols]}) --> usato sigma = 1 per queste colonne.")
                    sigma[degenerate_cols] = 1.0

                except Exception as e:
                    print(f"    [ERROR] Errore nel calcolo dei parametri e dei dataframe per sensore {sensor_name} --> {e} --> Impossibile continuare con l'analisi")
                    sys.exit()

        return self

    # Funzione incaricata di trasformare i dataframe definiti in tensori pytorch da fornire in input alla rete neurale
    def dataframe_to_tensor(self, process_dataframe, missioni_test, save_file_path):

        sensors_blocks_dataframes = {}
        if process_dataframe == 'S':
            sensors_blocks_dataframes = self.sensors_divided_database.copy()

        elif process_dataframe == 'N':
            try:
                with open(save_file_path, 'rb') as f:
                    sensors_blocks_dataframes = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Si richiama la funzione per generare i valori di media e deviazione standard per ogni blocco di missione --> salvati in un dizionario
        self.compute_normalization_parameters(sensors_blocks_dataframes, missioni_test)

        for trajectory, combinations_dict in sensors_blocks_dataframes.items():

            # Si inizializza il dizionario interno associato a ogni traiettoria
            if trajectory not in self.pytorch_tensor_dict:
                self.pytorch_tensor_dict[trajectory] = {}

            print(f'\n  Creazione tensori pytorch per le missioni associate alla traiettoria {trajectory} in corso:')

            for combination, sensors_dict in combinations_dict.items():

                # Si inizializza il dizionario interno associato a ogni singola missione
                if combination not in self.pytorch_tensor_dict[trajectory]:
                    self.pytorch_tensor_dict[trajectory][combination] = {}

                # Check sulla presenza di un sotto-dizionario non vuoto associato alla missione --> altrimenti c'è stato un errore all'interno del processing --> si genera un tensore vuoto e si passa oltre senza gestirlo al momento
                if not sensors_dict:
                    self.pytorch_tensor_dict[trajectory][combination] = {}
                    continue

                self.warn_count = 0
                mission_failed = False
                for sensor_name, sensor_dataframe in sensors_dict.items():

                    try:

                        # Se c'è stato un problema per un sensore si ignorano tutti gli altri in modo tale da avere il sotto-dizionario vuoto per quella missione
                        if mission_failed:
                            continue

                        # Si elimina dal dataframe del singolo sensore la colonna dei timestamp (questo crea automaticamente una copia del dataframe originale)
                        sensor_dataframe_copy = self.drop_columns(sensor_name, sensor_dataframe)

                        # Si estraggono i valori del dataframe in un array numpy
                        original_values = sensor_dataframe_copy.to_numpy(dtype=np.float32)

                        # Si modificano tali valori tenendo conto dei parametri definiti prima per la normalizzazione
                        sensor_mean = self.normalization_parameters_dict[sensor_name]['media']
                        sensor_std = self.normalization_parameters_dict[sensor_name]['deviazione standard']
                        normalized_values = (original_values - sensor_mean) / sensor_std

                        # Si genera ora il tensore pytorch
                        tensor = torch.tensor(normalized_values, dtype=torch.float32)

                        # Si concatenano al tensore di IMU anche quelli associati alle azioni propulsive e ai riferimenti di velocità --> hanno la stessa frequenza
                        if sensor_name == 'MOT' or sensor_name == 'V_ref':

                            # Si richiama il tensore generato in precedenza per il blocco IMU
                            IMU_tensor = self.pytorch_tensor_dict[trajectory][combination]['IMU']

                            # Si calcola il tensore combinato, concatenando ogni riga in direzione orizzontale (si affiancano i valori)
                            combined_tensor = torch.concat([IMU_tensor, tensor], dim=1)

                            # Si sovrascrive il tensore di pytorch per IMU con quello combinato
                            self.pytorch_tensor_dict[trajectory][combination]['IMU'] = combined_tensor

                        elif sensor_name == 'Depth' or sensor_name == 'DepthVel':
                            continue

                        # Si salva il tensore nel dizionario in tutti gli altri casi
                        else:
                            self.pytorch_tensor_dict[trajectory][combination][sensor_name] = tensor

                    except Exception as e:
                        self.pytorch_tensor_dict[trajectory][combination] = {}
                        mission_failed = True
                        self.warn_count += 1
                        print(f"    [WARNING] Errore nella creazione del tensore associato al sensore {sensor_name} --> {e}")
                        continue

                print(f'    Tensore associato alla missione {trajectory} - {combination} è stato generato con {self.warn_count} nuovi warnings rilevati, con dimensioni: DVL = {self.pytorch_tensor_dict[trajectory][combination]['DVL'].shape}, IMU = {self.pytorch_tensor_dict[trajectory][combination]['IMU'].shape}, GPS = {self.pytorch_tensor_dict[trajectory][combination]['GPS'].shape}.')

        return self


class PostProcessing:

    def __init__(self):

        # Dizionario per i Titoli (Coerente con i titoli dei capitoli/sezioni in blupolito)
        self.title_font = {
            'family': 'serif',      # Simula il font serif di Latin Modern usato all'interno del template Latex per la tesi
            'color': '#002E5F',     # Colore blupolito definito all'interno del template
            'weight': 'bold',       # Grassetto per far risaltare il titolo del grafico
            'size': 13,             # Dimensione equilibrata per il titolo del grafico
            'style': 'normal'
        }

        # Dizionario per le Etichette degli Assi (X e Y)
        self.axis_font = {
            'family': 'serif',      # Simula il font serif di Latin Modern usato all'interno del template Latex per la tesi
            'color': 'black',       # Testo nero ad alto contrasto per la massima leggibilità
            'weight': 'normal',     # Peso normale per le etichette descrittive
            'size': 11,             # Dimensione leggibile ma subordinata al titolo
            'style': 'normal'
        }

        self.waypoints_dict = {}            # Dizionario che conterrà le liste di waypoints in assi NED relativi di ogni traiettoria --> nella forma {'0': [], '1': []...}

        self.test_missions_dict = {}        # Dizionario che conterrà le liste dei range temporali associati alle singole traiettorie di ogni missione prescelta per il test --> nella forma {('0', '4'): [(), (), (), ()], ('0', '8'): [...], ...}

        self.warn_count = 0                 # Si inizializza il contatore dei warning per le operazioni iterative

        self.NN_recap_dict = {}             # Dizionario contenente i valori di recap delle prestazioni delle reti neurali

        self.first_mission = True           # Variabile booleana per l'intestazione del file di testo sull'errore RMSE

    # ======================================================
    #       DEFINIZIONE WAYPOINTS E TRAIETTORIE IDEALI
    # ======================================================

    # Funzione che permette la costruzione di un dizionario contenente le liste di waypoints (in coordinate WGS84 e UTM) di ogni traiettoria utilizzata
    def waypoints_dict_building(self, ROOT_DIR):

        # Si definisce la cartella di riferimento
        INPUT_FOLDER_PATH = os.path.join(ROOT_DIR, f'Telemetrie/Lista_punti')
        os.makedirs(INPUT_FOLDER_PATH, exist_ok=True)

        # Si definisce il percorso del file di output generato con il salvataggio
        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, f"Dizionari_datasets")
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, "Waypoints_dict.pkl")

        print(f"\nInizio lettura waypoints dai file presenti in {INPUT_FOLDER_PATH}")

        for i, file in enumerate(sorted(os.listdir(INPUT_FOLDER_PATH))):

            # Si estrae il nome/numero della traiettoria a partire dal nome del file --> oltre alla sesta traiettoria il nome cambia e occorre usare due lettere
            if i <= 5:
                trajectory = file[-5:-4]
            else:
                trajectory = file[-6:-4]

            # Si inizializza la lista di punti associata alla traiettoria (se non presente)
            if trajectory not in self.waypoints_dict:
                self.waypoints_dict[trajectory] = {}

            if 'WGS84' not in self.waypoints_dict[trajectory]:
                self.waypoints_dict[trajectory]['WGS84'] = []

            if 'UTM' not in self.waypoints_dict[trajectory]:
                self.waypoints_dict[trajectory]['UTM'] = []

            if 'UTM_abs' not in self.waypoints_dict[trajectory]:
                self.waypoints_dict[trajectory]['UTM_abs'] = []

            input_file_path = os.path.join(INPUT_FOLDER_PATH, file)
            with open(input_file_path, "r+") as txt_points:

                # Si verifica che il file di testo non sia vuoto
                if os.stat(input_file_path).st_size == 0:
                    print(f"  [WARNING] Il file {file} è vuoto --> si procede senza considerarlo.")
                    continue

                # Si itera per ogni riga (trattata come stringa) presente all'interno del file di iterazione
                not_valid_lines = 0
                for idx, line in enumerate(txt_points):

                    try:

                        clean_line = line.rstrip()          # Si eliminano eventuali spazi e caratteri di spaziatura (\n o \r) a fine riga
                        values = clean_line.split(",")      # Si analizza la linea isolando i termini presenti (nome_punto, latitudine, longitudine)
                        latitude = float(values[1])         # Si isola il valore della latitudine trasformandolo in numero decimale
                        longitude = float(values[2])        # Si isola il valore della longitudine trasformandolo in numero decimale

                        # Si aggiorna il dizionario con le coordinate latitude/longitudine
                        tuple_WGS84_coord = (latitude, longitude)
                        self.waypoints_dict[trajectory]['WGS84'].append(tuple_WGS84_coord)

                        # Si gestiscono i vari casi --> se l'indice è 0 si ha il waiting_point, che viene salvato e utilizzato una volta definite le coordinate (0,0) al punto 1
                        if idx == 0:
                            East_coord, North_coord, _, _ = utm.from_latlon(float(latitude), float(longitude))
                            wait_point_coord = (East_coord, North_coord)
                        elif idx == 1:
                            East_coord_0, North_coord_0, _, _ = utm.from_latlon(latitude, longitude)
                            tuple_UTM_coord_0 = (wait_point_coord[0] - East_coord_0, wait_point_coord[1] - North_coord_0)
                            tuple_UTM_coord = (0, 0)
                            self.waypoints_dict[trajectory]['UTM'].append(tuple_UTM_coord_0)
                            self.waypoints_dict[trajectory]['UTM'].append(tuple_UTM_coord)
                        else:
                            East_coord, North_coord, _, _ = utm.from_latlon(float(latitude), float(longitude))
                            tuple_UTM_coord = (East_coord - East_coord_0, North_coord - North_coord_0)
                            self.waypoints_dict[trajectory]['UTM'].append(tuple_UTM_coord)

                        # Si aggiorna il dizionario con le coordinate UTM assolute
                        tuple_UTM_abs_coord = (East_coord, North_coord)
                        self.waypoints_dict[trajectory]['UTM_abs'].append(tuple_UTM_abs_coord)

                    except Exception as e:
                        not_valid_lines += 1
                        print(f"  [WARNING] Rilevato un errore generico nella lettura delle coordinate {idx} della traiettoria {trajectory}: {e}")
                        continue

            print(f"  Lettura waypoints per la traiettoria {trajectory} eseguita con {not_valid_lines} punti non validi.")

        # Salvataggio del dizionario dei waypoints in formato pickle
        try:

            with open(output_file_path, 'wb') as f:
                pickle.dump(self.waypoints_dict, f)

        except Exception as e:
            print(f"  [WARNING] Errore nel salvataggio del database dei waypoints in formato pickle --> {e}")

        return self

    # Funzione per la creazione dei grafici delle traiettorie ideali, partendo dal dizionario di waypoints generato
    def ideal_trajectories_plot(self, ROOT_DIR):

        print("\nDefinizione delle traiettorie ideali basate sui waypoints definiti:")

        # Si itera per ogni percorso contenuto all'interno del dizionario di punti
        points_dict = self.waypoints_dict.copy()
        warn_counts = 0
        for trajectory_name, trajectory_points_list in points_dict.items():

            try:

                # Si itera per ogni metodo di espressione delle coordinate contenuto all'interno del dizionario --> per estrarre una lista con le sole coordinate East e una con quelle North
                East_list = []
                North_list = []
                for coord_type, coord_list in trajectory_points_list.items():

                    # Si salvano i soli dati associati al metodo UTM in un nuovo dizionario
                    if coord_type == "UTM":

                        # Si itera su ogni punto contenuto all'interno della lista di tuple contenenti le coordinate
                        for (East, North) in coord_list:
                            East_list.append(East)
                            North_list.append(North)

                # noinspection PyTypeChecker
                plt.figure(figsize=(10, 6))

                # Si richiama il plot dei punti utilizzando come input il dizionario tramite formulazione data
                plt.plot(East_list, North_list, label='Traiettoria ideale', linewidth=2, linestyle='--', marker='s', markersize=7)
                plt.xlabel("East [m]", fontdict=self.axis_font)
                plt.ylabel("North [m]", fontdict=self.axis_font)
                plt.title(f"Traiettoria del {trajectory_name}", fontdict=self.title_font, loc='center', pad=10)
                plt.minorticks_on()
                plt.grid(visible=True, which='both', alpha=0.5)
                plt.legend()
                plt.axis('equal')

                OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Risultati_NN/Traiettorie_ideali')
                os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
                output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"Traiettoria_ideale_{trajectory_name}.pdf")
                plt.savefig(output_file_path, bbox_inches='tight')
                #plt.show()
                plt.close()

            except Exception as e:
                warn_counts += 1
                print(f"  [WARNING] Errore nella generazione dei grafici ideali per la traiettoria {trajectory_name}--> {e}")
                continue

        print(f"  Traiettorie generate e salvate con {warn_counts} warnings (ogni warning è una figura non salvata).")

    # Funzione per la scrittura di un file .txt contenente tutti i valori (originali e convertiti) dei punti delle varie traiettorie
    def write_on_file(self, ROOT_DIR):

        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Risultati_NN/Traiettorie_ideali')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"Punti_missione.txt")

        try:

            print(f"\nTrascrizione su file di testo della lista dei waypoints in corso:")

            with open(output_file_path, "w+") as output_file:

                output_file.write("=" * 100 + "\n")
                output_file.write(" " * 18 + "ELENCO COMPLETO DEI PUNTI ASSOCIATI ALLE TRAIETTORIE DI MISSIONE\n")
                output_file.write("=" * 100 + "\n\n\n")

                # Si itera su ogni elemento del dizionario
                points_dict = self.waypoints_dict.copy()
                for trajectory_name, trajectory_points_list in points_dict.items():

                    output_file.write("-" * 100 + "\n")
                    output_file.write(" " * 40 + f"PERCORSO: {trajectory_name}" + "\n\n")

                    coord_type_list = list(trajectory_points_list.keys())
                    output_file.write(" " * 10 + f"Coordinate in formato: {coord_type_list[0]}" + " " * 10 + "|" + " " * 10 + f"Coordinate in formato: {coord_type_list[1]}" + "\n")

                    # Si itera sul numero dei punti e si scrive una riga di testo per ognuno
                    num_points = len(trajectory_points_list[coord_type_list[0]])
                    for i in range(num_points):
                        output_file.write(" " * 10 + f"({trajectory_points_list[coord_type_list[0]][i][0]:.7f}, {trajectory_points_list[coord_type_list[0]][i][1]:.7f})" + " " * 13 + " " * 13 + f"({trajectory_points_list[coord_type_list[1]][i][0]:.7f}, {trajectory_points_list[coord_type_list[1]][i][1]:.7f})""\n")

                    output_file.write("-" * 100 + "\n\n")

                output_file.write("=" * 100 + "\n\n")

            print(f"  Processo di salvataggio concluso correttamente.")

            return self

        except FileNotFoundError:
            raise FileNotFoundError(f"  [ERROR] Errore, cartella {OUTPUT_FOLDER_PATH} o file {output_file_path} non trovato") from None

        except Exception as e:
            print(f"  [WARNING] Errore nel processo di scrittura della lista di waypoints sul file {output_file_path} --> {e}")
            return 1

    # ======================================================
    #            SCOMPOSIZIONE MISSIONI DI TEST
    # ======================================================

    # Funzione interna che permette di costruire un dizionario
    def single_path_definition(self, resampled_database):

        trajectory_0_intervals = {('0', '4'): [(177, 414), (635, 980), (1262, 2366), (2630, 3402)],
                                 ('0', '8'): [(268, 611), (891, 1201), (1367, 1848), (2010, 2525)],
                                 ('0', '12'): [(199, 693), (832, 1176), (1465, 2440), (2659, 3469)],
                                 ('0', '16'): [(0, 697), (875, 1151), (1334, 1808), (1985, 2223)]}

        trajectory_T_intervals = {('T1', 'A'): [(99, 410)], ('T1', 'B'): [(72, 582)], ('T1', 'C'): [(108, 493)], ('T1', 'D'): [(97, 610)],
                                  ('T2', 'A'): [(131, 1449)], ('T2', 'B'): [(88, 1096)], ('T2', 'C'): [(96, 1364)], ('T2', 'D'): [(84, 1256)],
                                  ('T3', 'A'): [(98, 852)], ('T3', 'B'): [(52, 922)], ('T3', 'C'): [(132, 1389)], ('T3', 'D'): [(104, 963)],
                                  ('T4', 'A'): [(107, 598)], ('T4', 'B'): [(463, 1345)], ('T4', 'C'): [(409, 888)], ('T4', 'D'): [(298, 1010)]}

        for (trajectory, combination) in self.test_missions_dict:

            # Si verifica che la lista di waypoints per la traiettoria di iterazione non sia vuota --> altrimenti c'è stato un problema
            if not self.waypoints_dict[trajectory]['UTM']:
                print(f"  [ERROR] Per la traiettoria {trajectory} la lista di waypoints risulta vuota. Impossibile proseguire l'analisi per questa traiettoria.")
                continue

            # Se la missione è una di quelle di test (T...) l'analisi è inutile e se è una di quelle della traiettoria 0 si sono definiti gli intervalli autonomamente
            if trajectory == '0':
                self.test_missions_dict[(trajectory, combination)] = trajectory_0_intervals[(trajectory, combination)]
                continue
            elif trajectory in ['T1', 'T2', 'T3', 'T4']:
                self.test_missions_dict[(trajectory, combination)] = trajectory_T_intervals[(trajectory, combination)]
                continue

            # Si verifica che il DataFrame non sia vuoto --> altrimenti si salta la missione
            resampled_dataframe = resampled_database[trajectory][combination]
            if resampled_dataframe.empty:
                continue

            # Si definisce il timestamp iniziale del DataFrame come riferimento
            t0 = resampled_dataframe["timestamp"].iloc[0]

            # Si itera su ogni riga di ogni DataFrame presente nella lista delle missioni prescelte èer il test --> occorre gestire separatamente le T1, T2, T3 e T4
            start_found = False
            end_found = False
            rep_counter = 0
            start_time = None
            end_time = None
            for idx, line in resampled_dataframe.dropna().iterrows():

                # Si estraggono i valori di profondità del drone
                depth = line['Depth [m]']

                # Si converte il timestamp in secondi trascorsi dal timestamp di riferimento e si definisce il minimo delta temporale tra due percorsi
                t_s = (line['timestamp'] - t0).total_seconds()
                minimum_delta_time = 200

                # Si impone una condizione per verificare che sia stata completata una riemersione dopo essere arrivati alla fine --> altrimenti scatterebbe subito la condizione di start traiettoria
                if rep_counter >= 1:
                    if t_s - end_time <= 150:
                        continue

                # Si verifica che il drone sia nell'intorno del punto di attesa (il punto 0 del waypoints_dict) --> deve essere in un raggio di 1.5 metri
                if depth > 0.5 and not start_found:
                    start_time = t_s
                    start_found = True

                # Si definisce l'istante per cui il drone ha superato la quota di 6 metri (se sono nelle prime due ripetizioni) o 16 metri (se sono nelle ultime due) --> il check sul minimo delta_time è per essere sicuri che sia già una risalita e non ancora la prima discesa
                if start_found and (t_s - start_time) >= minimum_delta_time:

                    if rep_counter < 2:
                        min_valid_depth = 6
                    else:
                        min_valid_depth = 16

                    if depth <= min_valid_depth and not end_found:
                        end_time = t_s
                        end_found = True

                # Si aggiorna il contatore di traiettorie complete trovate e si salva la tupla con gli estremi di tempo della singola traiettoria
                if start_found and end_found:
                    rep_counter += 1
                    self.test_missions_dict[(trajectory, combination)].append((start_time, end_time))
                    start_found = False
                    end_found = False
                else:
                    continue

        return self

    # Funzione utile a inizializzare le chiavi del dizionario delle missioni di test tramite la lettura di un file di testo di input
    def test_missions_definition(self, resampled_database, ROOT_DIR):

        # Si inizializza il percorso del file di testo che contiene l'elenco delle missioni prescelte per il test
        INPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Telemetrie')
        os.makedirs(INPUT_FOLDER_PATH, exist_ok=True)
        input_file_path = os.path.join(INPUT_FOLDER_PATH, f"Missioni_test.txt")

        # Si inizializza il percorso del file pickle contenente il dizionario delle missioni di test
        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Dizionari_datasets')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"Test_missions_dict.pkl")

        print(f"\nDefinizione missioni di test e scomposizione in singole traiettorie in corso:")

        try:

            self.warn_count = 0
            with open(input_file_path, "r") as t:

                # Si analizza il file riga per riga estraendo le tuple traiettoria-combinazione
                for line in t:

                    try:

                        clean_line = line.lstrip().rstrip()
                        values = clean_line.split(", ")
                        trajectory = values[0]
                        combination = values[1]
                        mission = (trajectory, combination)

                        # Si inizializza la chiave all'interno del dizionario
                        if mission not in self.test_missions_dict:
                            self.test_missions_dict[(trajectory, combination)] = []

                    except Exception as e:
                        self.warn_count += 1
                        print(f"  [WARNING] Errore nella fase di lettura e salvataggio del contenuto del file --> {e}")
                        continue

        except FileNotFoundError:
            raise FileNotFoundError(f"  [ERROR] Errore nel processo di estrazione dati dei waypoints. Ricontrollare esistenza del file {input_file_path}") from None

        # Si richiama la funzione per scomporre la missione nelle quattro ripetizioni della traiettoria
        self.single_path_definition(resampled_database)

        # Si effettua il salvataggio del dizionario sul file pickle opportuno
        try:

            with open(output_file_path, 'wb') as f:
                pickle.dump(self.test_missions_dict, f)

        except Exception as e:
            self.warn_count += 1
            print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        print(f"  Dizionario delle missioni di test generato con {self.warn_count} warnings. Il dizionario è stato salvato all'interno del file {output_file_path} ")

        return self

    # ======================================================
    #       CALCOLO PARAMETRI E GRAFICI RETE NEURALE
    # ======================================================

    # Funzione interna che riceve in input i valori di RMSE della traiettoria e li trascrive su un file di testo riassuntivo
    def write_NN_recap_on_txt(self, ROOT_DIR, mission, path_name):

        # Si definisce il percorso di output del file
        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Risultati_NN')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"NN_evaluations_info.txt")

        try:

            print(f"\n  Salvataggio parametri di valutazione calcolati sul file di testo {output_file_path} in corso:")

            with open(output_file_path, "a") as output_file:

                if self.first_mission:
                    output_file.write("\n\n" + "=" * 100 + "\n")
                    output_file.write(" " * 26 + "RIASSUNTO VALORI DI RMSE (TEST vx)\n")
                    output_file.write("=" * 100 + "\n\n")

                output_file.write("\n" + "-" * 100 + "\n")
                output_file.write(f"\nPARAMETRI DI PERFORMANCE PER {mission} - {path_name}:\n")

                output_file.write(f"\nValori di Root Mean Square Error:\n")
                output_file.write(f"  RMSE in direzione East: {self.NN_recap_dict['RMSE_E']} m\n")
                output_file.write(f"  RMSE in direzione North: {self.NN_recap_dict['RMSE_N']} m\n")
                output_file.write(f"  RMSE complessivo: {self.NN_recap_dict['RMSE_tot']} m\n")

                output_file.write(f"\nValori di spostamento rispetto all'ultimo punto della missione:\n")
                output_file.write(f"  Delta_East finale: {self.NN_recap_dict['Delta_E_end']} m\n")
                output_file.write(f"  Delta_North finale: {self.NN_recap_dict['Delta_N_end']} m\n")
                output_file.write(f"  Delta_tot finale: {self.NN_recap_dict['Delta_tot_end']} m\n")

            print(f"    Salvataggio avvenuto correttamente.")

        except FileNotFoundError:
            raise FileNotFoundError(f"    [ERROR] Errore, cartella {OUTPUT_FOLDER_PATH} o file {output_file_path} non trovato") from None

        except Exception as e:
            print(f"    [WARNING] Errore nel processo di salvataggio su file dei dati --> {e}")
            return 1

        return self

    # Funzione che riceve in input una lista di tensori (rappresentanti i valori di posizione GPS e predette) per ogni missione di test e produce in output i 3 valori di errore RMSE
    def NN_recap_estimation(self, GPS_coordinates, NN_coordinates, ROOT_DIR, mission, path_name):

        print(f"\nCalcolo parametri di valutazione NN per la missione {mission[0]} - {mission[1]} in corso:")

        print(f"\n  Calcolo RMSE su assi E, N e complessiva:")

        # Si trasformano le liste di tensori in array numpy --> in modo da poter effettuare operazioni matematiche vettoriali
        GPS_coordinates_array = torch.cat(GPS_coordinates, dim=0).cpu().numpy()
        NN_coordinates_array = torch.cat(NN_coordinates, dim=0).cpu().numpy()
        # GPS_coordinates_array = np.array([t.detach().cpu().numpy().flatten() for t in GPS_coordinates])
        # NN_coordinates_array = np.array([t.detach().cpu().numpy().flatten() for t in NN_coordinates])

        # Si calcola l'errore RMSE per gli assi East e North separatamente
        self.NN_recap_dict['RMSE_E'] = np.sqrt(mean_squared_error(GPS_coordinates_array[:, 0], NN_coordinates_array[:, 0]))
        self.NN_recap_dict['RMSE_N'] = np.sqrt(mean_squared_error(GPS_coordinates_array[:, 1], NN_coordinates_array[:, 1]))

        # Si calcola l'errore RMSE complessivo sulla posizione --> si usa approccio differente per avere anche il vettore degli errori nel tempo, in modo da poterlo valutare graficamente
        total_errors = np.linalg.norm(GPS_coordinates_array - NN_coordinates_array, axis=1)  # Si calcola la norma di un vettore --> necessario per ottenere l'errore sulla posizione assoluta
        self.NN_recap_dict['RMSE_tot'] = np.sqrt(np.mean(total_errors ** 2))

        print(f'    Calcolo e salvataggio RMSE eseguito correttamente.')

        print(f"\n  Calcolo spostamenti lungo gli assi E, N e complessivo in corrispondenza del punto finale:")

        # Si scompongono i vettori nelle rispettive componenti
        GPS_East = [p[0] for p in GPS_coordinates_array]
        GPS_North = [p[1] for p in GPS_coordinates_array]
        NN_East = [p[0] for p in NN_coordinates_array]
        NN_North = [p[1] for p in NN_coordinates_array]

        # Si stampa il testo legato all'errore assoluto sulla posizione
        self.NN_recap_dict['Delta_E_end'] = np.abs(GPS_East[-1] - NN_East[-1])
        self.NN_recap_dict['Delta_N_end'] = np.abs(GPS_North[-1] - NN_North[-1])
        self.NN_recap_dict['Delta_tot_end'] = np.hypot(self.NN_recap_dict['Delta_E_end'], self.NN_recap_dict['Delta_N_end'])

        print(f"    Calcolo e salvataggio Delta eseguito correttamente.")

        # Si richiama la funzione per trascrivere le informazioni calcolate su file di testo
        self.write_NN_recap_on_txt(ROOT_DIR, mission, path_name)

        return self

    # Funzione che genera un grafico dati in input i dati target e quelli prodotti dalla rete con cui confrontarli e i nomi delle rispettive variabili
    def real_trajectories_plot(self, GPS_coordinates_list, NN_coordinates_list, GPS_origin, mission, path_name, title, ROOT_DIR):

        # Si inizializza il percorso del file di output che verrà generato
        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, f'Risultati_NN/Traiettorie_reali')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f'Confronto_traiettorie_M{mission}_{path_name}.pdf')

        # Si trasformano le liste di tensori in array numpy --> in modo da poter effettuare operazioni matematiche vettoriali
        GPS_coordinates_array = torch.cat(GPS_coordinates_list, dim=0).cpu().numpy()
        NN_coordinates_array = torch.cat(NN_coordinates_list, dim=0).cpu().numpy()

        # Si scompone il vettore nelle rispettive componenti
        GPS_East = [p[0] for p in GPS_coordinates_array]
        GPS_North = [p[1] for p in GPS_coordinates_array]

        NN_East = [p[0] for p in NN_coordinates_array]
        NN_North = [p[1] for p in NN_coordinates_array]

        waypoints = self.waypoints_dict[mission[0]]['UTM_abs']
        (east0, north0) = GPS_origin[0], GPS_origin[1]
        waypoints_East = [p[0]-east0 for p in waypoints]
        waypoints_North = [p[1]-north0 for p in waypoints]

        # noinspection PyTypeChecker
        plt.figure(figsize=(10, 6))

        #plt.plot(waypoints_East, waypoints_North, label = 'Waypoints', marker = 's', markersize = 6)
        plt.plot(GPS_East, GPS_North, color='blue', label='Target (GPS)', linewidth=2, linestyle='--')
        plt.plot(NN_East, NN_North, color='red', label='NavNet Prediction', linewidth=1.5, linestyle='-')

        # 2. Punti di Inizio (Start)
        plt.scatter(GPS_East[0], GPS_North[0], color='blue', s=50, edgecolors='black', zorder=5)
        plt.text(GPS_East[0], GPS_North[0], f' Start ({GPS_East[0]:.2f}, {GPS_North[0]:.2f})', color='blue', fontsize=9, fontweight='bold', va='bottom')
        plt.scatter(NN_East[0], NN_North[0], color='red', s=50, edgecolors='black', zorder=5)
        plt.text(NN_East[0], NN_North[0], f' Start NavNet ({NN_East[0]:.2f}, {NN_North[0]:.2f})', color='red', fontsize=9, fontweight='bold', va='bottom')

        # 3. Punti di Fine (End)
        plt.scatter(GPS_East[-1], GPS_North[-1], color='blue', marker='X', s=80, edgecolors='black', zorder=5)
        plt.text(GPS_East[-1], GPS_North[-1], f' End Target ({GPS_East[-1]:.2f}, {GPS_North[-1]:.2f})', color='blue', fontsize=9, fontweight='bold', va='top')
        plt.scatter(NN_East[-1], NN_North[-1], color='red', marker='X', s=80, edgecolors='black', zorder=5)
        plt.text(NN_East[-1], NN_North[-1], f' End NavNet ({NN_East[-1]:.2f}, {NN_North[-1]:.2f})', color='red', fontsize=9, fontweight='bold', va='bottom')

        # Definizione parametri grafico
        plt.title(title, fontdict = self.title_font, loc ="center", pad = 10)
        plt.xlabel('East [m]', fontdict = self.axis_font)
        plt.ylabel('North [m]', fontdict = self.axis_font)
        plt.legend()
        plt.minorticks_on()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')

        # Si stampa il testo legato all'errore assoluto sulla posizione
        Delta_E = np.abs(GPS_East[-1] - NN_East[-1])
        Delta_N = np.abs(GPS_North[-1] - NN_North[-1])
        print(f'\nPer la missione {mission} e la traiettoria {path_name}, si hanno i seguenti valori di Errore Assoluto sulla posizione:')
        print(f'    Delta sulla posizione in direzione North è pari a: {Delta_N} m.')
        print(f'    Delta sulla posizione in direzione East è pari a: {Delta_E} m.')

        plt.savefig(output_file_path, bbox_inches='tight')
        #plt.show()
        plt.close()

    # Funzione che permette di stampare il grafico relativo alla loss function durante il training
    def plot_loss_function(self, loss_train, loss_val, epoche, ROOT_DIR, titolo='Evoluzione della Loss function'):

        # Si crea un vettore con un numero di elementi equispaziati per le epoche di iterazione
        epoch_vector = list(range(epoche))

        # noinspection PyTypeChecker
        plt.figure(figsize=(10, 6))

        # Disegniamo la traiettoria reale
        plt.plot(epoch_vector, loss_train, color='blue', label='Training loss', linewidth=2)
        plt.plot(epoch_vector, loss_val, color = 'red', label='Validation loss', linewidth=2)

        # Abbellimento grafico
        plt.title(titolo, fontdict=self.title_font)
        plt.xlabel('Epoche di addestramento', fontdict=self.axis_font)
        plt.ylabel('Loss function', fontdict=self.axis_font)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Risultati_NN')
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.makedirs(OUTPUT_FOLDER_PATH)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'Loss_function.pdf')

        plt.savefig(output_file_path, bbox_inches='tight')
        #plt.show()
        plt.close()