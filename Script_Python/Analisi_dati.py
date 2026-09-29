'''       PRE-PROCESSING E ANALISI DATI
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             '''

''' Il pre-processing produce delle liste di liste di DataFrame (una lista contenente più dataframe, per ogni missione e per ogni blocco di sensori) --> fatto su più 
    griglie di riferimento con passo uniforme (0.100s per DVL, 0.200s per IMU e 1s per GPS) --> misurazioni asincrone spostate al decimo di secondo più vicino all'istante 
    reale --> inizio e fine del dataframe per ogni sensore (su singola missione) sono coincidenti per evitare sfasamenti nella valutazione della rete --> i dataframe
    sono poi convertiti in tensori pytorch secondo la medesima struttura di divisioni dei DataFrame

    La classe Post-Processing presenta varie funzioni utili all'analisi dei risultati dell'esecuzione della rete neurale, come la rappresentazione dei grafici e il calcolo 
    dell'errore medio (RMSE) --> è, inoltre, presente una funzione opzionale che permette di separare automaticamente le singole ripetizioni di traiettoria all'interno di 
    una missione (con criterio )
    '''

import numpy as np                              # Libreria per i calcoli matematici
import torch
import pandas as pd                             # Libreria per la gestione dei dataframe
import re                                       #
import warnings                                 # Libreria per la gestione dei warnings come exception
from pandas.errors import DtypeWarning          # Libreria per la gestione dei warnings di tipo misto nella lettura di un dataframe pandas
import pickle                                   # Libreria per la gestione del formato di salvataggio pickle
import os                                       # Libreria per la gestione dell'interfaccia tra sistema operativo e script
from sklearn.metrics import mean_squared_error  # Libreria per l'utilizzo della funzione per il calcolo dell'errore quadratico medio
import matplotlib.pyplot as plt                 # Libreria per la gestione dell'ambiente grafico
import utm                                      # Libreria per la gestione della conversione di coordinate espresse in gradi (WGS84)

pd.set_option('display.max_columns', None)              # Visualizza tutte le colonne (nessun limite al numero)
pd.set_option('display.max_colwidth', None)             # Visualizza tutto il contenuto della cella (evita di troncare stringhe lunghe)
pd.set_option('display.expand_frame_repr', False)       # Evita che le colonne vadano a capo su più righe nel terminale
pd.options.display.max_rows = 100                       # Imposta il numero massimo di righe visualizzabili a schermo

warnings.filterwarnings('error', category=pd.errors.DtypeWarning)           # Forza i DtypeWarning a comportarsi come eccezioni


class PreProcessing:

    # Si inizializza la struttura comune degli elementi appartenenti alla classe
    def __init__(self, freq_riferimento='100ms'):

        self.freq = freq_riferimento                    # Si assegna alla variabile interna la frequenza base della griglia

        self.config_matrix = {}                         # Si inizializza un dizionario che conterrà, per ogni traiettoria, una lista di stringhe che individuano la combinazione scenario ambientale/scenario operativo associata ad ogni missione --> avrà forma {'traiettoria0': ['comb0', 'comb1', ...], 'traiettoria1': [], ...}

        self.raw_trajectories_dataframes = {}           # Si inizializza un dizionario che una volta riempito avrà forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe...}, 'traiettoria1'; {}, ...}

        self.sensor_blocks_dataframes = {}              # Si inizializza un dizionario che conterrà per ogni missione i dataframe dei singoli sensori su griglia temporale comune --> ha forma {'traiettoria0': {'missione1': {'IMU'; dataframe, 'DVL':dataframe...} , 'missione2': {'IMU'; dataframe, 'DVL':dataframe...}, ...}, 'traiettoria1': {}, ...}

        self.resampled_trajectories_dataframes = {}     # Si inizializza un dizionario che conterrà per ogni missione il dataframe globale trattato con resampling --> ha forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe, ...}, 'traiettoria1': {}, ...}

        self.interpolated_trajectories_dataframes = {}  # Si inizializza un dizionario che conterrà per ogni missione il dataframe globale trattato con interpolazione --> ha forma {'traiettoria0': {'missione1': dataframe, 'missione2': dataframe, ...}, 'traiettoria1': {}, ...}

        self.pytorch_tensor = {}                        # Si inizializza un dizionario che conterrà per ogni missione i tensori di ogni blocco sensori --> ha forma {'traiettoria0': {'missione1': {'IMU'; tensore, 'DVL':tensore...}, 'missione2': {'IMU'; tensore, 'DVL':tensore...}, ...}, 'traiettoria1': {}, ...}

        self.normalization_parameters = {}              # Dizionario per salvare medie e deviazioni standard --> nella forma {'IMU': {'mean':valore, 'std':valore}, 'DVL': {}...}

        self.warn_count = 0                             # Contatore warning --> da riinizializzare a 0 quando si cambia traiettoria o blocco di sensori

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
                    line = line.split(" ")      # Si fa in modo di trasformare il testo di ogni linea in una lista di stringhe --> separatore " " per le intestazioni del tipo "Traiettoria 1:"

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
        dataframe_dict = {}

        # Si itera su ogni traiettoria simulata --> l'elemento config_list è una lista di stringhe contenenti la coppia configurazione ambientale/operativa del drone in simulazione
        for trajectory, config_list in self.config_matrix.items():

            print(f'\n  Analisi file csv per traiettoria {trajectory} in corso:')

            input_folder = os.path.join(ROOT_DIR, f'Telemetrie/Telemetria_traiettoria_{trajectory}')   # Si definisce la cartella di riferimento

            # Si inizializza il dizionario interno associato alla singola missione
            if trajectory not in dataframe_dict:
                dataframe_dict[trajectory] = {}

            self.warn_count = 0
            for combination in config_list:

                # Se la missione è la 0 si evita l'analisi e si procede con la traiettoria successiva --> ha già un solo file di telemetria per configurazione
                if trajectory == "0":

                    telemetry_file_name = os.path.join(input_folder, f'M{combination}.csv')

                    try:

                        mission_dataframe = pd.read_csv(telemetry_file_name)
                        dataframe_dict[trajectory][combination] = mission_dataframe

                    except FileNotFoundError:
                        self.warn_count += 1
                        print(f'    [WANRING] File di telemetria per la missione {trajectory} - {combination} non trovato.')
                        continue

                    except DtypeWarning:
                        self.warn_count += 1
                        print(f'    [WARNING] Il file di telemetria per la missione {trajectory} - {combination} contiene valori di tipo misto (problema opportunamente corretto dopo).')
                        mission_dataframe = pd.read_csv(telemetry_file_name, low_memory=False)
                        dataframe_dict[trajectory][combination] = mission_dataframe

                    except Exception as e:
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

                        # Si unificano i dataframe tramite concatenazione e li si ordina per timestamp crescente
                        mission_dataframe = pd.concat([DVL_dataframe, INS_dataframe], axis=0, ignore_index=True)    # Con axis=0 si impone di impilare i dataframe e non affiancarli mentre con ignore_index=True si unifica l'indice
                        mission_dataframe = mission_dataframe.sort_values('timestamp', ignore_index=True)            # Si resetta l'indice dopo aver eseguito il sort

                        dataframe_dict[trajectory][combination] = mission_dataframe

                    except FileNotFoundError:
                        self.warn_count += 1
                        print(f'    [WANRING] File di telemetria per la missione {trajectory} - {combination} non trovato.')
                        continue

                    except DtypeWarning:
                        self.warn_count += 1
                        print(f'    [WARNING]: In file di telemetria della missione {trajectory} - {combination} contiene valori di tipo misto (problema opportunamente corretto dopo).')
                        DVL_dataframe = pd.read_csv(DVL_csv, low_memory=False)
                        INS_dataframe = pd.read_csv(INS_csv, low_memory=False)
                        mission_dataframe = pd.concat([DVL_dataframe, INS_dataframe], axis=0, ignore_index=True)
                        mission_dataframe = mission_dataframe.sort_values('timestamp', ignore_index=True)
                        dataframe_dict[trajectory][combination] = mission_dataframe

                    except Exception as e:
                        self.warn_count += 1
                        print(f'    [WARNING] Errore nel processo di unificazione e estrazione dati dai file di telemetria per la missione {trajectory} - {combination} --> {e}')
                        continue

            print(f"    Unificazione file csv per la traiettoria {trajectory} completata con {self.warn_count} warnings.")

        # -----------------------------------------------------
        #      GENERAZIONE DIZIONARIO DATAFRAME GLOBALE
        # -----------------------------------------------------

        # Si itera su ogni missione presente all'interno del dizionario
        for trajectory, mission_dict in dataframe_dict.items():

            print(f'\n  Estrazione dati di telemetria per missioni traiettoria {trajectory} in corso:')
            if not mission_dict:
                print(f'    [WARNING] Per la traiettoria {trajectory} non risultano dataframe presenti.')
                continue

            self.warn_count = 0
            for combination, mission_dataframe in mission_dict.items():

                if mission_dataframe.empty:
                    self.warn_count += 1
                    print(f'    [WARNING] Il dataframe della missione {trajectory} - {combination} risulta vuoto.')
                    continue

                # Si ripulisce il dataframe eliminando eventuali caratteri non numerici (problema dati misti) e gli spazi associati a ogni stringa (per uniformità)
                mission_dataframe = self.dataframe_cleaning(mission_dataframe, trajectory, combination)

                # Si verifica che il dataframe sia rimasto non vuoto --> altrimenti c'è stato un problema nella pulizia del dataframe e si passa alla missione successiva
                if mission_dataframe.empty:
                    continue
                mission_dataframe['nome'] = mission_dataframe['nome'].str.strip()

                try:

                    # Si converte la colonna timestamp del dataframe in tempo effettivo, con formato orario classico 12:12:12:200, tramite pandas
                    mission_dataframe['timestamp'] = pd.to_datetime(mission_dataframe['timestamp'], unit='s')

                    # Generazione nomi per intestazione tabella
                    nomenclatura = {'DVL': ['DVL_Lock', 'DVL_Vx [m/s]', 'DVL_Vy [m/s]', 'DVL_Vz [m/s]', 'DVL_Altitude [m]'],
                                    'Attitude': ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'],
                                    'Depth': ['Depth [m]'],
                                    'DepthVel': ['Depth_Rate [m/s]'],
                                    'AxisRef': ['Ref_Vx [%]', 'Ref_Vy [%]', 'Ref_Vz [%]', 'Ref_ωx [%]', 'Ref_ωy [%]', 'Ref_ωz [%]'],
                                    'MotorRef': ['Mot_FV [%]', 'Mot_FL [%]', 'Mot_RV [%]', 'Mot_RL [%]', 'Mot_FW1 [%]', 'Mot_FW2 [%]', 'Mot_FW3 [%]', 'Mot_FW4 [%]'],
                                    'UTMPos': ['UTM_North [m]', 'UTM_East [m]', 'Zone', 'Quality']}

                    # Si itera su ogni elemento presente all'interno del dizionario di nomenclatura
                    parts: list[pd.DataFrame] = []
                    for sensor, columns_name in nomenclatura.items():

                        # Si crea un sottoinsieme contenente tutte le righe del file .csv in cui compare il nome del sensore di iterazione (si usa .copy() per non modificare file originale)
                        sensor_data = mission_dataframe[mission_dataframe['nome'] == sensor].copy()

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

                    # Si salva il dataframe all'interno di un dizionario assegnandola alla singola missione
                    if trajectory not in self.raw_trajectories_dataframes:
                        self.raw_trajectories_dataframes[trajectory] = {}
                    self.raw_trajectories_dataframes[trajectory][combination] = new_mission_dataframe

                except Exception as e:
                    self.warn_count += 1
                    print(f'    [ERROR] Errore nel processo di formattazione del dataframe per la missione {trajectory} - {combination} (missione non considerata) --> {e}')
                    self.raw_trajectories_dataframes[trajectory][combination] = None
                    continue

            print(f'    I dataframe globali per la traiettoria {trajectory} sono stati generati correttamente, al netto di {self.warn_count} warnings rilevati (le eccezioni sono None nel dizionario).')

        # -----------------------------------------------------
        #           PULIZIA FASE INIZIALE MISSIONE
        # -----------------------------------------------------

        # Si itera su ogni missione presente all'interno del dizionario
        dataframe_dict = self.raw_trajectories_dataframes.copy()
        reference_depth = 9
        for trajectory, mission_dict in dataframe_dict.items():

            print(f'\n  Eliminazione dati trasferta verso waiting point per missioni traiettoria {trajectory} in corso:')
            if not mission_dict:
                print(f'    [WARNING] Per la traiettoria {trajectory} non risultano dataframe presenti.')
                continue

            self.warn_count = 0
            for combination, mission_dataframe in mission_dict.items():

                if mission_dataframe.empty:
                    self.warn_count += 1
                    print(f'    [WARNING] Il dataframe della missione {trajectory} - {combination} risulta vuoto.')
                    continue

                try:

                    # Si definisce il timestamp associato al primo superamento della profondità di soglia prescelta
                    start_condition = mission_dataframe[mission_dataframe['Depth [m]'] >= reference_depth]
                    start_time = start_condition['timestamp'].iloc[0]

                    # Si modifica il dataframe eliminando tutti i timestamp precedenti
                    self.raw_trajectories_dataframes[trajectory][combination] = mission_dataframe[mission_dataframe['timestamp'] >= start_time].reset_index(drop=True)

                except Exception as e:
                    self.warn_count += 1
                    print(f'    [WARNING] Errore nel processo di pulizia della fase iniziale per la missione {trajectory} - {combination} (missione non considerata) --> {e}')
                    self.raw_trajectories_dataframes[trajectory][combination] = None
                    continue

            print(f'    I dataframe globali per la traiettoria {trajectory} sono stati modificati correttamente, al netto di {self.warn_count} warnings rilevati (le eccezioni sono None nel dizionario).')

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
                dataframe_GPS = self.sensor_blocks_dataframes[trajectory][combination]['GPS'].copy()
                dataframe_IMU = self.sensor_blocks_dataframes[trajectory][combination]['IMU'].copy()
            elif processing_method == 'interpolazione':
                dataframe_GPS = self.interpolated_trajectories_dataframes[trajectory][combination].copy()
                dataframe_IMU = self.interpolated_trajectories_dataframes[trajectory][combination][['Pitch [deg]', 'Yaw [deg]']].copy()

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
                            self.sensor_blocks_dataframes[trajectory][combination]['GPS'] = dataframe_GPS
                        elif processing_method == 'interpolazione':
                            self.interpolated_trajectories_dataframes[trajectory][combination] = dataframe_GPS

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
            raise KeyError(f"    [ERROR] Errore nella conversione delle coordinate NED -> Body per la missione {trajectory} - {combination}.") from None
            return 1

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
                dataframe = self.sensor_blocks_dataframes[trajectory][combination]['GPS'].copy()
            elif processing_method == 'interpolazione':
                dataframe = self.interpolated_trajectories_dataframes[trajectory][combination].copy()

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
                        self.sensor_blocks_dataframes[trajectory][combination]['GPS'] = dataframe
                    elif processing_method == 'interpolazione':
                        self.interpolated_trajectories_dataframes[trajectory][combination] = dataframe

                else:
                    self.warn_count += 1
                    print(f"    [WARNING] Non trovata nessuna colonna con nome 'UTM_East' o 'UTM_North' nel dataframe della missione {trajectory} - {combination}.")

            else:
                self.warn_count += 1
                print(f"    [WARNING] Il dataframe per la missione {trajectory} - {combination} risulta vuoto.")

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nella trasformazione delle coordinate UTM -> NED per la missione {trajectory} - {combination} --> {e}")

        return self

    # Funzione interna in grado di ricevere in input la missione di riferimento traiettoria-combinazione e il dizionario con i dataframe per ogni sensore e produrre in output un dataframe a passo temporale uniforme
    def dataframe_resampling(self, trajectory, combination):

        try:

            # Si inizializza il dataframe di output (comune a tutti i sensori) come il dataframe legato al solo DVL --> questo fornisce il riferimento temporale, avendo la frequenza maggiore
            resampled_dataframe = self.sensor_blocks_dataframes[trajectory][combination]['DVL']

            # Si itera su ogni sensore presente all'interno del dizionario prodotto in precedenza
            for sensor_name, sensor_dataframe in self.sensor_blocks_dataframes[trajectory][combination].items():

                if sensor_name != 'DVL':
                    resampled_dataframe = pd.merge(resampled_dataframe, sensor_dataframe, on='timestamp', how='outer')

            # Si riordina il dataframe prodotto secondo il timestamp per maggiore sicurezza
            resampled_dataframe = resampled_dataframe.sort_values('timestamp', ignore_index=True)

            # Si aggiunge il dataframe generato per la missione al dizionario complessivo
            resampled_dataframe.drop(columns=['UTM_East [m]', 'UTM_North [m]'], inplace=True)
            self.resampled_trajectories_dataframes[trajectory][combination] = resampled_dataframe

        except Exception as e:
            self.warn_count += 1
            print(f"    [WARNING] Errore nel processo di resampling per la missione {trajectory} - {combination} --> {e}")
            self.resampled_trajectories_dataframes[trajectory][combination] = None

        return self

    # Funzione incaricata di analizzare i dataframe di tutte le missioni per generare dei dataframe contenenti solo le colonne di ogni blocco di sensori da usare per addestramento
    def raw_dataframes_processing(self, generate_dataframe, save_file_path, sensors_frequencies):

        # Se la scelta dell'utente è no, il dataframe globale si prende dal file pickle
        if generate_dataframe == 'N':

            try:
                with open(save_file_path, 'rb') as f:
                    dataframe_globale = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Se la scelta è si il dataframe globale si prende direttamente da quello generaato con la lettura dei file csv
        elif generate_dataframe == 'S':

            # Si crea una copia del dizionario contenente il dataframe globale --> in modo da non modificare quello originale per errore
            dataframe_globale = self.raw_trajectories_dataframes.copy()

        # Si itera per ogni elemento contenuto nel dizionario separando il nome del foglio (= nome_missione) e il dataframe associato
        for trajectory, mission_dict in dataframe_globale.items():

            print(f'\n  Analisi dei dataframe associati alla traiettoria {trajectory} in corso:')

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.sensor_blocks_dataframes:
                self.sensor_blocks_dataframes[trajectory] = {}

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.resampled_trajectories_dataframes:
                self.resampled_trajectories_dataframes[trajectory] = {}

            # Si definiscono i blocchi di sensori con frequenze differenti --> tuple che contengono sia le colonne che la frequenza associata oltre che la lista di destinazione associata all'attributo self
            try:
                cols_IMU = ('IMU', ['timestamp', 'Roll [deg]', 'Pitch [deg]', 'Yaw [deg]'])
                cols_DVL = ('DVL', ['timestamp', 'DVL_Vx [m/s]', 'DVL_Vy [m/s]', 'DVL_Vz [m/s]', 'DVL_Altitude [m]'])
                cols_depth = ('Depth', ['timestamp', 'Depth [m]'])
                cols_depth_rate = ('Depth_rate', ['timestamp', 'Depth_Rate [m/s]'])
                cols_Vref = ('V_ref', ['timestamp', 'Ref_Vx [%]', 'Ref_Vy [%]', 'Ref_Vz [%]', 'Ref_ωx [%]', 'Ref_ωy [%]', 'Ref_ωz [%]'])
                cols_MOT = ('MOT', ['timestamp', 'Mot_FV [%]', 'Mot_FL [%]', 'Mot_RV [%]', 'Mot_RL [%]', 'Mot_FW1 [%]', 'Mot_FW2 [%]'])
                cols_GPS = ('GPS', ['timestamp', 'UTM_North [m]', 'UTM_East [m]'])
                sensors_blocks = [cols_DVL, cols_IMU, cols_depth, cols_depth_rate, cols_MOT, cols_Vref, cols_GPS]
            except KeyError:
                raise KeyError(f"  [ERROR] Ci sono chiavi associate ai sensori errate per la traiettoria {trajectory}. Ricontrollare la corrispondenza dei nomi con il dataframe puro generato in precedenza.") from None

            self.warn_count = 0
            for combination, dataframe in mission_dict.items():

                # Si inizializzano le liste necessarie al trattamento del dataframe di missione
                df_blocks_list = []                 # Lista che conterrà il dataframe di missione ridotto ai soli parametri necessari e suddiviso in blocchi (uno per sensore)
                t_start_list = []                   # Lista che conterrà i timestamp di inizio misurazioni di ogni sensore
                t_end_list = []                     # Lista che conterrà i timestamp di fine misurazioni di ogni sensore

                # Si crea il dizionario interno associato alla singola missione all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
                if combination not in self.sensor_blocks_dataframes[trajectory]:
                    self.sensor_blocks_dataframes[trajectory][combination] = {}

                # -----------------------------------------------------
                #        DEFINIZIONE LIMITI TEMPORALI DATAFRAMES
                # -----------------------------------------------------

                # Si itera su ogni blocco di sensori individuato per definire istante di inizio e fine di ognuno
                for (sensor_block_name, sensor_block_cols) in sensors_blocks:

                    # Si crea un sottoinsieme del dataframe completo originale che contenga solo le colonne del blocco di iterazione --> si eliminano le righe in cui sono presenti valori NaN (utile per definizione istante iniziale)
                    dataframe_copy = dataframe[sensor_block_cols].dropna(how='any')

                    # Si riordinano i valori in base al timestamp e lo si salva nella lista complessiva
                    dataframe_copy = dataframe_copy.sort_values('timestamp')
                    df_blocks_list.append(dataframe_copy)

                    # Calcolo istante iniziale e conclusivo del dataframe
                    t_start_list.append(dataframe_copy['timestamp'].min())
                    t_end_list.append(dataframe_copy['timestamp'].max())

                # Si selezionano gli istanti di inizio e fine del dataframe aggiornato
                t_start_dataframe = max(t_start_list).round(self.freq)        # Il dataframe inizia dal valore più alto tra gli istanti iniziali dei singoli blocchi --> così non occorre ipotizzare grandezze con metodo backward
                t_end_dataframe = min(t_end_list).round(self.freq)            # Il dataframe finisce con il valore più basso tra gli istanti finali dei singoli blocchi --> così riduco il numero di stati ipotizzati (ma solitamente è GPS a finire prima)

                # Si determina l'istante finale effettivo del GPS a seguito dell'arrotondamento con griglia --> sarà questo il riferimento per tutte le altre griglia
                GPS_freq = sensors_frequencies['GPS']
                t_end_GPS = pd.date_range(start=t_start_dataframe, end=t_end_dataframe, freq=f'{(1/GPS_freq)*1000}ms').max()

                # -----------------------------------------------------
                #           MERGE SU GRIGLIA TEMPORALE COMUNE
                # -----------------------------------------------------

                # Si itera contemporaneamente sulla lista dei dataframe per sensori e sulle tuple dei blocchi per fare effettivamente il resampling
                for sensor_block_dataframe, (sensor_block_name, sensor_block_cols) in  zip(df_blocks_list, sensors_blocks):

                    try:

                        frequenza = sensors_frequencies[sensor_block_name]

                        # Si genera la griglia temporale con passo uniforme associata al blocco di sensori --> segue la frequenza dei dati
                        griglia_uniforme = pd.DataFrame({'timestamp': pd.date_range(start=t_start_dataframe, end=t_end_GPS, freq=f'{(1/frequenza)*1000}ms')})

                        # Si effettua il merge tra le misurazioni del sensore considerato e la griglia temporale --> usato metodo nearest per garantire che una misurazione leggermente successiva al timestamp di riferimento non venga ignorata
                        sensor_block_dataframe = pd.merge_asof(griglia_uniforme, sensor_block_dataframe, on='timestamp', direction='backward')

                        # Si aggiunge il dataframe relativo al singolo blocco di sensori considerati al dizionario generato appositamente --> suddividendo per
                        self.sensor_blocks_dataframes[trajectory][combination][sensor_block_name] = sensor_block_dataframe

                    except Exception as e:
                        self.warn_count += 1
                        print(f'    [WARNING] Errore nella fase di unificazione griglia temporale per il blocco sensori {sensor_block_name} nella missione {trajectory} - {combination} --> {e}')
                        continue

                # -----------------------------------------------------
                #              CONVERSIONE COORDINATE UTM
                # -----------------------------------------------------

                self.UTM_to_NED('resampling', trajectory, combination)
                self.NED_to_body('resampling', trajectory, combination)

                # -----------------------------------------------------
                #            RESAMPLING IN DATAFRAME UNICO
                # -----------------------------------------------------

                self.dataframe_resampling(trajectory, combination)

                if self.resampled_trajectories_dataframes[trajectory][combination] is None:
                    continue

            print(f'    Analisi delle missioni associate alla {trajectory} completata con {self.warn_count} warnings rilevati.')

        return self

    # Funzione incaricata di generare il dizionario dei dataframe per ogni missione con dati trattati tramite interpolazione --> parte dal dataframe puro
    def dataframe_interpolation(self, generate_dataframe, save_file_path):

        # Se la scelta dell'utente è no, il dataframe globale si prende dal file pickle
        if generate_dataframe == 'N':

            try:
                with open(save_file_path, 'rb') as f:
                    dataframe_globale = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Se la scelta è si il dataframe globale si prende direttamente da quello generaato con la lettura dei file csv
        elif generate_dataframe == 'S':

            # Si crea una copia del dizionario contenente il dataframe globale --> in modo da non modificare quello originale per errore
            dataframe_globale = self.raw_trajectories_dataframes.copy()

        # Si itera per ogni elemento contenuto nel dizionario separando il nome del foglio (= nome_missione) e il dataframe associato
        for trajectory, mission_dict in dataframe_globale.items():

            print(f'\n  Interpolazione dei dataframe associati alla traiettoria {trajectory} in corso:')

            # Si crea il dizionario associato alla singola traiettoria all'interno del dizionario globale, che contiene tutti i dataframe trattati (se non presente)
            if trajectory not in self.interpolated_trajectories_dataframes:
                self.interpolated_trajectories_dataframes[trajectory] = {}

            self.warn_count = 0
            for combination, mission_dataframe in mission_dict.items():

                try:

                    # Si inizializza una copia del dataframe originale e si imposta come indice la colonna dei timestamp --> utile poiché usando il metodo di interpolazione lineare si tiene conto anche di misurazioni non equispaziate
                    interpolated_dataframe = mission_dataframe.set_index('timestamp')

                    # Si effettua un'operazione di unwrap delle grandezze angolari --> in questo modo si evitano problemi per quanto riguarda il passaggio da 0 a 360 e viceversa, che l'interpolazione non gestisce in maniera continua
                    IMU_cols = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
                    IMU_mask = interpolated_dataframe[IMU_cols].notna().all(axis=1)
                    interpolated_dataframe.loc[IMU_mask, IMU_cols] = np.unwrap(interpolated_dataframe.loc[IMU_mask, IMU_cols].to_numpy(), period=360, axis=0)

                    # Si effettua l'interpolazione lineare
                    interpolated_dataframe = interpolated_dataframe.interpolate(method='time', limit_area='inside')

                    # Si scalano le misurazioni angolari per stare nell0intervallo 0-360 classico --> si applica solo allo yaw perché rollio e pitch hanno anche valori negativi, quindi quest'operazione potrebbe corrompere i normali vslori
                    interpolated_dataframe['Yaw [deg]'] = interpolated_dataframe['Yaw [deg]'] % 360

                    # Si eliminano le righe della tabella in cui non sono presenti i dati di tutti i sensori contemporaneamente --> di solito gli ultimi ad avviarsi sono axisref e motorref
                    interpolated_dataframe.dropna(how='any', inplace=True)

                    # Si ripristina il timestamp come colonna e non come indice
                    interpolated_dataframe.reset_index(inplace=True)

                    # Si eliminano le colonne non utili (DVL_Lock, Quality e Zone per GPS)
                    interpolated_dataframe.drop(columns=['DVL_Lock', 'Mot_FW3 [%]', 'Mot_FW4 [%]', 'Zone', 'Quality'], inplace=True)

                    # Si aggiunge il dataframe creato al dizionario complessivo
                    self.interpolated_trajectories_dataframes[trajectory][combination] = interpolated_dataframe

                except KeyError:
                    raise KeyError(f"    [ERROR] Errore nel drop delle colonne non utili. Ricontrollare che non siano presenti errori.") from None

                except Exception as e:
                    self.warn_count += 1
                    print(f"    [WARNING] Errore nell'interpolazione della missione {trajectory} - {combination} --> {e}")
                    self.interpolated_trajectories_dataframes[trajectory][combination] = None
                    continue

                # Si applicano le trasformazioni in assi NED e assi body per ottenere il dataframe completo
                self.UTM_to_NED('interpolazione', trajectory, combination)
                self.NED_to_body('interpolazione', trajectory, combination)

                # Si eliminano le colonne associate alla posizione UTM, ormai inutili
                try:
                    self.interpolated_trajectories_dataframes[trajectory][combination].drop(columns=['UTM_East [m]', 'UTM_North [m]'], inplace=True)
                except KeyError:
                    raise KeyError(f"    [ERROR] Errore nel drop delle colonne non utili. Ricontrollare che non siano presenti errori.") from None

            print(f"    Interpolazione dataframes associati alla traiettoria {trajectory} effettuata con {self.warn_count} warnings rilevati.")

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

                save_file_path = os.path.join(ROOT_DIR, 'Dataframe_globale.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.raw_trajectories_dataframes, f)

                    print(f'    Salvataggio del dataframe in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.raw_trajectories_dataframes.items():

                print(f'  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_raw.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = self.raw_trajectories_dataframes[trajectory][combination].copy()

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

                save_file_path = os.path.join(ROOT_DIR, 'Dataframe_resampled.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.resampled_trajectories_dataframes, f)

                    print(f'    Salvataggio del dataframe resampled in file {save_file_path} eseguito correttamente.')

                save_file_path = os.path.join(ROOT_DIR, 'Dataframe_resampled_blocchi.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.sensor_blocks_dataframes, f)

                    print(f'    Salvataggio del dataframe suddiviso per sensore in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.resampled_trajectories_dataframes.items():

                print(f'\n  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} trattate con resampling in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_resampled.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = self.resampled_trajectories_dataframes[trajectory][combination].copy()

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
                            worksheet.set_column('M:R', 17, formato_axis)   # Blocco misure axis_ref
                            worksheet.set_column('S:X', 17, formato_mot)    # Blocchi misure dei motor_ref
                            worksheet.set_column('Y:AD', 19, formato_GPS)   # Blocchi misure del GPS

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

                save_file_path = os.path.join(ROOT_DIR, 'Dataframe_interpolated.pkl')
                with open(save_file_path, 'wb') as f:
                    pickle.dump(self.interpolated_trajectories_dataframes, f)

                    print(f'    Salvataggio del dataframe in file {save_file_path} eseguito correttamente.')

            except Exception as e:
                print(f'    [WARNING] Errore nel salvataggio del dizionario contenente tutti i dataframe in formato pickle --> {e}')

        elif formato == 'excel':

            for trajectory, dataframes in self.interpolated_trajectories_dataframes.items():

                print(f'\n  Salvataggio del file in formato Excel per le missioni della traiettoria {trajectory} trattate con interpolazione in corso:')

                try:

                    # Si genera un file Excel differente per ogni traiettoria
                    save_folder_path = os.path.join(ROOT_DIR, f'File_Excel/Traiettoria_{trajectory}')
                    os.makedirs(save_folder_path, exist_ok=True)
                    save_file_path = os.path.join(save_folder_path, f'Telemetria_{trajectory}_interpolated.xlsx')
                    with pd.ExcelWriter(save_file_path, engine='xlsxwriter') as writer:

                        for combination, dataframe in dataframes.items():

                            # Si crea una copia del dataframe per evitare modifiche impattanti nel main
                            dataframe_copy = dataframe.copy()

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

    # Funzione interna incaricata di calcolare il valore medio e la deviazione standard per ogni colonna appartenente a un blocco di sensori
    def compute_normalization_parameters(self, sensor_blocks_dataframes, missioni_test):

        # -----------------------------------------------------
        #          UNIFICAZIONE DATAFRAME PER SENSORE
        # -----------------------------------------------------

        # Si itera su ogni elemento del dizionario di dataframe generato per i blocchi --> obiettivo è creare un dizionario che contiene una lista delle sezioni di dataframe di tutte le missioni per ogni sensore
        sensor_dataframes_list = {}
        for trajectory, mission_dict in sensor_blocks_dataframes.items():

            self.warn_count = 0
            for combination, sensors_dict in mission_dict.items():

                # Si utilizza la missione per la definizione dei valori di normalizzazione solo per le missioni di training --> poi si applicano direttamente a quelle di test per garantire che la rete non abbia in alcun modo informazioni sulle misurazioni future nella fase di test
                if (trajectory, combination) not in missioni_test:

                    for sensor_name, sensor_dataframe in sensors_dict.items():

                        # Si verifica di non avere già riscontrato un problema con il sensore in una missione precedente --> altrimenti si continua senza rieseguire l'analisi
                        if sensor_name in sensor_dataframes_list and sensor_dataframes_list[sensor_name] is None:
                            continue

                        try:

                            # Si eliminano le colonne relative agli assi UTM e NED assoluti
                            if sensor_name == 'GPS':
                                sensor_dataframe = sensor_dataframe.drop(columns=['UTM_North [m]', 'UTM_East [m]', 'NED_North [m]', 'NED_East [m]', 'X_body_ist [m]', 'Y_body_ist [m]'])

                            # Si aggiunge il dataframe per la missione e il nome del sensore associato a un apposito dizionario --> nella forma {'IMU': [], 'DVL': [], 'GPS': []}
                            if sensor_name not in sensor_dataframes_list:
                                sensor_dataframes_list[sensor_name] = []
                            sensor_dataframes_list[sensor_name].append(sensor_dataframe)

                        except Exception as e:
                            self.warn_count += 1
                            print(f"    [WARNING] Errore nell'unificazione dei dataframe per sensore {sensor_name} --> {e}")
                            sensor_dataframes_list[sensor_name] = None
                            continue


        # -----------------------------------------------------
        #           CALCOLO PARAMETRI DI OGNI SENSORE
        # -----------------------------------------------------

        # Si itera ora sul nuovo dizionario --> un blocco di sensori alla volta in sostanza
        for sensor_name, dataframes_list in sensor_dataframes_list.items():

            if dataframes_list is not None:

                try:

                    # Si genera un dataframe unico unendo quelli di tutte le missioni contenuti nella lista senza la colonna 'timestamp' (non necessaria)
                    dataframe = pd.concat(dataframes_list).drop(columns=['timestamp'])

                    # Si calcola il valore della media e della deviazione standard sulla totalità delle missioni considerate
                    mean_value = dataframe.mean().to_numpy(dtype=np.float32)           # Si usa .means() per la media, .values per convertire il tutto in valori numerici (array numpy)
                    std = dataframe.std().to_numpy(dtype=np.float32)                   # Si usa .std() per la deviazione, .values per convertire il tutto in valori numerici (array numpy)

                    # Si aggiorna il dizionario aggiungendo al blocco del sensore i valori trovati
                    self.normalization_parameters[sensor_name] = {'media': mean_value, 'deviazione standard': std}

                    # Si modifica un eventuale valore di deviazione standard pari a 0 con un 1 --> per evitare errori matematici durante l'esecuzione della normalizzazione
                    sigma = self.normalization_parameters[sensor_name]['deviazione standard']
                    sigma[sigma == 0] = 1.0

                except Exception as e:
                    print(f"    [WARNING] Errore nel calcolo dei parametri e dei dataframe per sensore {sensor_name} --> {e}")

        return self

    # Funzione incaricata di trasformare i dataframe definiti in tensori pytorch da fornire in input alla rete neurale
    def dataframe_to_tensor(self, process_dataframe, missioni_test, save_file_path):

        sensors_blocks_dataframes = {}
        if process_dataframe == 'S':
            sensors_blocks_dataframes = self.sensor_blocks_dataframes.copy()

        elif process_dataframe == 'N':
            try:
                with open(save_file_path, 'rb') as f:
                    sensors_blocks_dataframes = pickle.load(f)

            except Exception as e:
                print(f" [WARNING] Errore nell'apertura del file pickle per l'elaborazione dei dataframe --> {e}")
                raise

        # Si richiama la funzione per generare i valori di media e deviazione standard per ogni blocco di missione --> salvati in un dizionario
        self.compute_normalization_parameters(sensors_blocks_dataframes, missioni_test)

        for trajectory, mission_dict in sensors_blocks_dataframes.items():

            # Si inizializza il dizionario interno associato a ogni traiettoria
            if trajectory not in self.pytorch_tensor:
                self.pytorch_tensor[trajectory] = {}

            print(f'\n  Creazione tensori pytorch per le missioni associate alla traiettoria {trajectory} in corso:')

            for combination, sensors_dict in mission_dict.items():

                # Si inizializza il dizionario interno associato a ogni singola missione
                if combination not in self.pytorch_tensor[trajectory]:
                    self.pytorch_tensor[trajectory][combination] = {}

                self.warn_count = 0
                for sensor_name, sensor_dataframe in sensors_dict.items():

                    try:

                        # Si elimina dal dataframe del singolo sensore la colonna dei timestamp (questo crea automaticamente una copia del dataframe originale)
                        sensor_dataframe_copy = sensor_dataframe.drop(columns=['timestamp'])

                        # Si eliminano le colonne con le coordinate assolute, non necessarie per il training
                        if sensor_name == 'GPS':
                            sensor_dataframe_copy = sensor_dataframe_copy.drop(columns=['UTM_North [m]', 'UTM_East [m]', 'NED_North [m]', 'NED_East [m]', 'X_body_ist [m]', 'Y_body_ist [m]'])

                        # Si estraggono i valori del dataframe
                        original_values = sensor_dataframe_copy.to_numpy(dtype=np.float32)

                        # Si modificano tali valori tenendo conto dei parametri definiti prima per la normalizzazione
                        sensor_mean = self.normalization_parameters[sensor_name]['media']
                        sensor_std = self.normalization_parameters[sensor_name]['deviazione standard']
                        normalized_values = (original_values - sensor_mean) / sensor_std

                        # Si genera ora il tensore pytorch
                        tensor = torch.tensor(normalized_values, dtype=torch.float32)

                        # Si concatenano al tensore di IMU anche quelli associati alle azioni propulsive e ai riferimenti di velocità --> hanno la stessa frequenza
                        if sensor_name == 'MOT' or sensor_name == 'V_ref':

                            # Si richiama il tensore generato in precedenza per il blocco IMU
                            IMU_tensor = self.pytorch_tensor[trajectory][combination]['IMU']

                            # Si calcola il tensore combinato, concatenando ogni riga in direzione orizzontale (si affiancano i valori)
                            combined_tensor = torch.concat([IMU_tensor, tensor], dim=1)

                            # Si sovrascrive il tensore di pytorch per IMU con quello combinato
                            self.pytorch_tensor[trajectory][combination]['IMU'] = combined_tensor

                        if sensor_name == 'Depth' or sensor_name == 'DepthVel':
                            continue

                        # Si salva il tensore nel dizionario in tutti gli altri casi
                        else:
                            self.pytorch_tensor[trajectory][combination][sensor_name] = tensor

                    except Exception as e:
                        self.warn_count += 1
                        print(f"    [WARNING] Errore nella creazione del tensore associato al sensore {sensor_name} --> {e}")
                        continue

                print(f'    Tensore associato alla missione {trajectory} - {combination} è stato generato con {self.warn_count}  warnings rilevati, con dimensioni: DVL = {self.pytorch_tensor[trajectory][combination]['DVL'].shape}, IMU = {self.pytorch_tensor[trajectory][combination]['IMU'].shape}, GPS = {self.pytorch_tensor[trajectory][combination]['GPS'].shape}.')

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

    # Funzione che permette di
    #def single_path_definition(self):

    # Funzione che permette la costruzione di un dizionario contenente le liste di waypoints (in coordinate WGS84 e UTM) di ogni traiettoria utilizzata
    def waypoints_dict_building(self, ROOT_DIR):

        input_folder = os.path.join(ROOT_DIR,f'Telemetrie/Lista_punti')  # Si definisce la cartella di riferimento

        for trajectory, file in enumerate(sorted(os.listdir(input_folder))):

            trajectory = str(trajectory)

            # Si inizializza la lista di punti associata alla traiettoria (se non presente)
            if trajectory not in self.waypoints_dict:
                self.waypoints_dict[trajectory] = {}

            if 'WGS84' not in self.waypoints_dict[trajectory]:
                self.waypoints_dict[trajectory]['WGS84'] = []

            if 'UTM' not in self.waypoints_dict[trajectory]:
                self.waypoints_dict[trajectory]['UTM'] = []

            file_path = os.path.join(input_folder, file)
            with open(file_path, "r+") as txt_points:

                # Si verifica che il file di testo non sia vuoto
                if os.stat(file_path).st_size == 0:
                    print(f"  [WARNING] Il file {file} è vuoto --> si procede senza considerarlo.")
                    continue

                # Si itera per ogni riga (trattata come stringa) presente all'interno del file di iterazione
                for idx, line in enumerate(txt_points):

                    try:

                        clean_line = line.rstrip()          # Si eliminano eventuali spazi e caratteri di spaziatura (\n o \r) a fine riga
                        values = clean_line.split(",")      # Si analizza la linea isolando i termini presenti (nome_punto, latitudine, longitudine)
                        latitude = float(values[1])         # Si isola il valore della latitudine trasformandolo in numero decimale
                        longitude = float(values[2])        # Si isola il valore della longitudine trasformandolo in numero decimale
                        tuple_WGS84_coord = (latitude, longitude)

                        if idx == 0:
                            continue
                        if idx == 1:
                            East_coord_0, North_coord_0, _, _ = utm.from_latlon(latitude, longitude)
                            tuple_UTM_coord = (0, 0)
                        else:
                            East_coord, North_coord, _, _ = utm.from_latlon(float(latitude), float(longitude))

                            # Si effettua la sottrazione per arrivare alle coordinate NED relative al punto iniziale --> sono quelle necessarie per definire i waypoint
                            tuple_UTM_coord = (East_coord - East_coord_0, North_coord - North_coord_0)

                        self.waypoints_dict[trajectory]['WGS84'].append(tuple_WGS84_coord)
                        self.waypoints_dict[trajectory]['UTM'].append(tuple_UTM_coord)

                    except Exception as e:
                        print(f"  [WARNING] Rilevato un errore generico nella lettura delle coordinate {idx} della traiettoria {trajectory}: {e}")
                        continue

    # Funzione per la creazione dei grafici delle traiettorie ideali, partendo dal dizionario di waypoints generato
    def ideal_trajectories_plot(self, ROOT_DIR):

        # Si itera per ogni percorso contenuto all'interno del dizionario di punti
        points_dict = self.waypoints_dict.copy()
        for trajectory_name, trajectory_points_list in points_dict.items():

            # Si itera per ogni metodo di espressione delle coordinate contenuto all'interno del dizionario
            East_list = []
            North_list = []
            for coord_type, coord_list in trajectory_points_list.items():

                # Si salvano i soli dati associati al metodo UTM in un nuovo dizionario
                if coord_type == "UTM":

                    # Si itera su ogni punto contenuto all'interno della lista di tuple contenenti le coordinate
                    for (East, North) in coord_list:
                        East_list.append(East)
                        North_list.append(North)

            # Si apre il grafico
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
            plt.show()

    # Funzione per la scrittura di un file .txt contenente tutti i valori (originali e convertiti) dei punti delle varie traiettorie
    def write_on_file(self, ROOT_DIR):

        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, 'Risultati_NN/Traiettorie_ideali')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f"Punti_missione_UTM.txt")

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

    # Funzione che riceve in input una lista di tensori (rappresentanti i valori di posizione GPS e predette) per ogni missione di test e produce in output i 3 valori di errore RMSE
    def RMSE_estimation(self, GPS_coordinates, NN_coordinates):

        # Si trasformano le liste di tensori in array numpy --> in modo da poter effettuare operazioni matematiche vettoriali
        GPS_coordinates_array = torch.cat(GPS_coordinates, dim=0).cpu().numpy()
        NN_coordinates_array = torch.cat(NN_coordinates, dim=0).cpu().numpy()
        #GPS_coordinates_array = np.array([t.detach().cpu().numpy().flatten() for t in GPS_coordinates])
        #NN_coordinates_array = np.array([t.detach().cpu().numpy().flatten() for t in NN_coordinates])

        # Si calcola l'errore RMSE per gli assi East e North separatamente
        RMSE_E = np.sqrt(mean_squared_error(GPS_coordinates_array[:, 0], NN_coordinates_array[:, 0]))
        RMSE_N = np.sqrt(mean_squared_error(GPS_coordinates_array[:, 1], NN_coordinates_array[:, 1]))

        # Si calcola l'errore RMSE complessivo sulla posizione --> si usa approccio differente per avere anche il vettore degli errori nel tempo, in modo da poterlo valutare graficamente
        errori_tot = np.linalg.norm(GPS_coordinates_array - NN_coordinates_array, axis=1)  # Si calcola la norma di un vettore --> necessario per ottenere l'errore sulla posizione assoluta
        RMSE_tot = np.sqrt(np.mean(errori_tot ** 2))

        print(f'    RMSE sulla posizione in direzione North è pari a: {RMSE_N} m.')
        print(f'    RMSE sulla posizione in direzione East è pari a: {RMSE_E} m.')
        print(f'    RMSE sulla posizione complessiva è pari a: {RMSE_tot} m.')

        return RMSE_tot, RMSE_E, RMSE_N

    # Funzione che genera un grafico dati in input i dati target e quelli prodotti dalla rete con cui confrontarli e i nomi delle rispettive variabili
    def real_trajectories_plot(self, GPS_coordinates_list, NN_coordinates_list, mission, path_name, title, ROOT_DIR):

        # Si trasformano le liste di tensori in array numpy --> in modo da poter effettuare operazioni matematiche vettoriali
        GPS_coordinates_array = torch.cat(GPS_coordinates_list, dim=0).cpu().numpy()
        NN_coordinates_array = torch.cat(NN_coordinates_list, dim=0).cpu().numpy()

        # Si scompone il vettore nelle rispettive componenti
        GPS_East = [p[0] for p in GPS_coordinates_array]
        GPS_North = [p[1] for p in GPS_coordinates_array]

        NN_East = [p[0] for p in NN_coordinates_array]
        NN_North = [p[1] for p in NN_coordinates_array]

        waypoints = self.waypoints_dict[mission[0]]
        waypoints_East = [p[0] for p in waypoints]
        waypoints_North = [p[1] for p in waypoints]

        plt.figure(figsize=(10, 6))

        plt.plot(waypoints_East, waypoints_North, label = 'Waypoints', marker = 's', markersize = 6)
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
        Delta_N = np.abs(GPS_East[-1] - NN_East[-1])
        Delta_E = np.abs(GPS_North[-1] - NN_North[-1])
        print(f'\nPer la missione {mission} e la traiettoria {path_name}, si hanno i seguenti valori di Errore Assoluto sulla posizione:')
        print(f'    Delta sulla posizione in direzione North è pari a: {Delta_N} m.')
        print(f'    Delta sulla posizione in direzione East è pari a: {Delta_E} m.')

        OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, f'Risultati_NN/Traiettorie_reali')
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        output_file_path = os.path.join(OUTPUT_FOLDER_PATH, f'Confronto_traiettorie_M{mission}_{path_name}.pdf')
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.show()
