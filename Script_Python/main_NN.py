'''     IMPLEMENTAZIONE RETE NEURALE NAVNET
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             '''

'''All'interno del main si effettuano le seguenti operazioni in ordine:
    - pre-processing dei dati, ottenendo i tensori utili per la rete neurale --> tiene conto delle differenti frequenze dei sensori:
        -- 10 Hz per DVL
        -- 5 Hz per IMU e REF
        -- 1 Hz per GPS
    - definizione delle batch di iterazione per la rete neurale --> tramite class Dataset si definisce la singola batch per ogni blocco e poi si uniscono insieme
    - esecuzione rete neurale per la stima della posizione
    - ciclo di addestramento sui dati per l'ottimizzazione dei pesi utilizzati
    '''

import os                               # Libreria per l'utilizzo di Python integrato nel sistema
import torch                            # Libreria per il Deep Learning
from Analisi_dati import PreProcessing  # Importo il modulo per eseguire il pre-processing dei dati da utilizzare per la rete
from Rete_neurale import Dataset        # Importo il modulo per definire le batch dei blocchi di sensore per ogni secondo di missione
from Rete_neurale import NavNet         # Importo qui la rete neurale
from Rete_neurale import PostProcessing # Importo il modulo per eseguire il post-processing dei dati prodotti dalla rete
import matplotlib.pyplot as plt         # Libreria per la gestione dell'ambiente grafico
import numpy as np

# Funzione che permette di effettuare la trasformazione inversa per la traiettoria
#def

def stima_trapezoidale(velocita, timestamps, pos_iniziale=0):

    pos = [pos_iniziale]
    for i in range(len(velocita) - 1):
        dt = timestamps[i+1] - timestamps[i]
        # Formula trapezoidale
        spostamento = 0.5 * (velocita[i] + velocita[i+1]) * dt
        pos.append(pos[-1] + spostamento)
    return np.array(pos)


# Funzione che permette di stampare il grafico relativo alla loss function durante il training
def plot_loss_function(loss_train, loss_val, epoche, titolo='Evoluzione della Loss function'):

    # Si crea un vettore con un numero di elementi equispaziati per le epoche di iterazione
    vettore_epoche = list(range(epoche))

    plt.figure(figsize=(10, 6))

    # Disegniamo la traiettoria reale
    plt.plot(vettore_epoche, loss_train, color='blue', label='Training loss', linewidth=2)
    plt.plot(vettore_epoche, loss_val, color = 'red', label='Validation loss', linewidth=2)

    # Abbellimento grafico
    plt.title(titolo, fontsize=14)
    plt.xlabel('Epoche di addestramento', fontsize=12)
    plt.ylabel('Loss function', fontsize=12)
    plt.legend()  # Mostra la legenda opportuna
    plt.grid(True, linestyle=':', alpha=0.6)

    if not os.path.exists('Grafici'):
        os.makedirs('Grafici')

    plt.savefig('Grafici/Loss_function.png', bbox_inches='tight')
    plt.show()


# ======================================================
#              INIZIALIZZAZIONE VARIABILI
# ======================================================

# Variabili legate alla rete neurale
batch_size = 32                                                                         # Numero di secondi di missione forniti contemporaneamente alla rete
batch_size_test = 1                                                                     # Per il test si usa una batch size di 1s --> si ottiene la stima della posizione per ogni secondo di navigazione (coincidenza perfetta con batch GPS)
num_epoch = 2                                                                           # Numero di epoche di iterazione per l'addestramento
numero_missioni = 16                                                                    # Numero di missioni da analizzare
test_missions = [('0', '4'), ('0', '8'), ('0', '12'), ('0' , '16')]                     # Missioni utilizzate solo nella fase di test
sensors_frequencies = {'IMU': 5, 'DVL': 10, 'MOT': 5, 'V_ref': 5, 'GPS': 1}             # Frequenze associate ai sensori

# File di input/output necessarie al pre-processing
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DICT_DIR = os.path.join(ROOT_DIR, 'Dizionari dataframes')
os.makedirs(DF_DICT_DIR, exist_ok=True)                                                 # Si genera la cartella di destinazione dei file di salvataggio pickle con i vari dataframe
file_database_raw = os.path.join(DF_DICT_DIR, 'Dataframe_globale.pkl')                  # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_database_resampled = os.path.join(DF_DICT_DIR, 'Dataframe_resampled.pkl')          # File pickle di output/input contenente i dati sul dizionario globale dei dataframe resampled generato al termine del pre-processing
file_database_sensors = os.path.join(DF_DICT_DIR, 'Dataframe_resampled_blocchi.pkl')    # File pickle di output/input contenente i dati sul dizionario suddiviso per i vari sensori dei dataframe generato al termine del pre-processing
file_config = os.path.join(ROOT_DIR, "Telemetrie/Matrice_configurazioni.txt")           # File di testo di input contenente un elenco delle combinazioni condizione ambientale/operativa per ogni traiettoria simulata
weights_save_path = os.path.join(ROOT_DIR, "NavNet_weights.pkl")


# ======================================================
#             PRE-PROCESSING TELEMETRIA
# ======================================================

# Si richiama la classe associata al pre-processing
pre_processing = PreProcessing()

# -----------------------------------------------------
#         ESTRAZIONE DATAFRAMES GREZZI DA CSV
# -----------------------------------------------------
while True:

    # Si richiede all'utente se si vuole generare un dataframe a partire dai file di telemetria
    generate_dataframe = input(f'\nSi desidera generare un nuovo dataframe contenente i dati di telemetria (S/N)?').upper()

    if generate_dataframe == 'S':

        # Si analizzano i file .csv contenenti i dati di telemetria di ogni missione --> si genera un dizionario globale e si salva in un apposito file pickle
        pre_processing.csv_analysis(file_config, ROOT_DIR)
        pre_processing.save_raw_dataframes('pickle', DF_DICT_DIR)

        # Gestione del salvataggio del file in formato Excel del dataframe
        while True:
            save_excel = input(f'\nSi vuole salvare il dataframe anche in formato Excel (S/N)?').upper()
            if save_excel == 'S':
                pre_processing.save_dataframe_resampled('excel', ROOT_DIR)
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
        print(f'  Input inserito non riconosciuto. Ripetere la scelta.')

# -----------------------------------------------------
#          PROCESSING DEI DATAFRAMES GREZZI
# -----------------------------------------------------

while True:

    process_dataframe = input(f"\nSi vuole generare da zero il dataframe elaborato (S/N)?").upper()

    if process_dataframe == 'S':
        pre_processing.raw_dataframes_processing(generate_dataframe, file_database_raw, sensors_frequencies)        # Si analizzano i dati di telemetria --> si generano in output due dizionari, uno con un dataframe ricampionato per ogni missione e l'altro suddiviso anche per sensori
        pre_processing.save_dataframe_resampled('pickle', DF_DICT_DIR)

        # Gestione del salvataggio del file in formato Excel del dataframe ricampionato
        while True:
            save_excel = input(f'\n  Si vuole salvare il dataframe elaborato con resampling in formato Excel (S/N)?').upper()
            if save_excel == 'S':
                pre_processing.save_dataframe_resampled('excel', ROOT_DIR)
                break
            elif save_excel == 'N':
                print(f'    Continuazione analisi senza salvataggio Excel.')
                break
            else:
                print(f'    Inserito input non valido. Ripetere la scelta.')

        break

    elif process_dataframe == 'N':

        if os.path.isfile(file_database_resampled):
            print(f'  Caricamento del dataframe resampled e interpolated dai file {file_database_resampled}')
            break
        else:
            print(f'  File contenenti i database non trovati. Verificare o procedere con la generazione dei database.')

    else:

        print(f'  Inserito un input non valido. Ripetere la scelta.')

pre_processing.dataframe_to_tensor(process_dataframe, test_missions, file_database_sensors)       # Si generano i tensori pytorch -->  si genera in output un dizionario in cui per ogni missione sono presenti 3 tensori, uno per ogni frequenza di misurazioni presente
tensor_dict = pre_processing.pytorch_tensor                                                       # Si richiama il dizionario di tensori da usare per la rete neurale --> ogni tensore è una lista ordinata (per sensore) contenente le misurazioni di tutte missioni


# ======================================================
#           INIZIALIZZAZIONE RETE NEURALE
# ======================================================

# Inizializzazione del modello di rete neurale
rete_neurale = NavNet()                                             # Si inizializza la rete principale
rete_neurale.initialize_weights()                                   # Si inizializzano i pesi tramite funzione definita

# Scheduler --> dimezza il learning rate se la Val_Loss non scende per 5 epoche
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rete_neurale.optimizer, mode='min', factor=0.5, patience=3)


# ======================================================
#                TRAINING RETE NEURALE
# ======================================================

# Si richiede se si desidera fare il training della rete --> da fare al cambiamento del dataset
do_training = input(f'\nSi desidera effettuare il training della rete (S/N)?').upper()
if do_training == 'S':

    # Si richiama la classe Dataset per definire la lunghezza della batch unitaria per ogni blocco di sensori --> successivamente tramite libreria pytorch si generano batch di forma [32, frequenza, numero di colonne]
    training_dataset = Dataset(tensor_dict, sensors_frequencies, test_missions, mode='train')

    # Si scompone il dataset di addestramento per poter fare anche validazione
    dataset_length = len(training_dataset)
    split_point = int(dataset_length * 0.80)                                                                    # Definizione del punto di split dei dati --> 80% delle missioni usate per training e 25% per validazione
    training_subset = torch.utils.data.Subset(training_dataset, range(0, split_point))                          # Definizione del subset legato ai soli dati di train
    validation_subset = torch.utils.data.Subset(training_dataset, range(split_point, dataset_length))           # Definizione del subset legato ai soli dati di validazione --> sono le ultime due missioni

    # Si trasformano i subset definiti nei dataloader necessari alla rete neurale --> si aggregano più blocchi in base alla batch_size definita
    training_loader = torch.utils.data.DataLoader(training_subset, batch_size=batch_size, shuffle=True)                 # Definizione del dataloader associato al training
    validation_loader = torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=False)            # Definizione del dataloader associato alla validazione

    # Si calcola la loss e si effettua la backpropagation per la stima dei migliori pesi
    print(f'\n  Inizio training rete neurale:')
    loss_train = []
    loss_val = []
    for epoch in range(num_epoch):

        # Si imposta la rete neurale in modalità training
        rete_neurale.train()

        # Inizializzazione parametri necessari
        epoch_batches_number = 0                      # Inizializzazione numero di batch per l'epoca corrente
        training_epoch_loss = 0                       # Inizializzazione perdita complessiva per l'epoca corrente in merito al training
        validation_epoch_loss = 0                     # Inizializzazione perdita complessiva per l'epoca corrente in merito alla validazione

        # -----------------------------------------------------
        #           STIMA PERDITA E BACKPROPAGATION
        # -----------------------------------------------------

        for batch in training_loader:

            IMU_data = batch[0]                 # Batch di dimensioni [32, 5, 9] --> contiene i 3 valori IMU, i 4 sensori MOT (FV, FL, FW1, FW2) e i 2 di velocità reference (Vxref, omega_yref)
            DVL_data = batch[1]                 # Batch di dimensioni [32, 10, 3]
            dati_gps = batch[2].squeeze(1)      # Batch di dimensioni [32, 1, 2] --> si vuole avere dimensione [32, 2] per congruenza con output rete neurale --> si elimina una dimensione dalla batch

            # Si richiama la funzione di aggiornamento pesi
            training_batch_loss = rete_neurale.backpropagation(IMU_data, DVL_data, dati_gps)

            # Si aggiornano i valori per il calcolo della media dei pesi sulla singola epoca
            training_epoch_loss += training_batch_loss       # Aggiornamento del valore totale di perdita per l'epoca corrente
            epoch_batches_number += 1                        # Aggiornamento del numero di batches per l'epoca corrente

        # Calcolo media associata all'epoca corrente
        loss_epoca_train = training_epoch_loss / epoch_batches_number
        loss_train.append(loss_epoca_train)
        print(f"    Loss media di training calcolata per l'epoca {epoch}: {loss_epoca_train:.4f}")

        # -----------------------------------------------------
        #               VALIDAZIONE RETE NEURALE
        # -----------------------------------------------------

        rete_neurale.eval()
        with torch.no_grad():
            for batch in validation_loader:

                IMU_data = batch[0]
                DVL_data = batch[1]
                GPS_data = batch[2].squeeze(1)

                NN_displacement_estimation = rete_neurale.forward(IMU_data, DVL_data)

                # Calcolo della loss function
                validation_batch_loss = rete_neurale.loss_criterion(NN_displacement_estimation, GPS_data)
                validation_epoch_loss += validation_batch_loss

            # Calcolo della loss media sulla singola epoca
            loss_epoca_validazione = validation_epoch_loss / epoch_batches_number
            loss_val.append(loss_epoca_validazione)
            print(f"    Loss media di validazione calcolata per l'epoca {epoch}: {loss_epoca_validazione:.4f}")

        # -----------------------------------------------------
        #      AGGIORNAMENTO PARAMETRI TRAMITE SCHEDULER
        # -----------------------------------------------------
        scheduler.step(loss_epoca_validazione)

    # Si realizza il grafico complessivo
    plot_loss_function(loss_train, loss_val, num_epoch)

    # Si salvano i pesi aggiornati alla fine dell'addestramento in un dizionario pytorch apposito
    torch.save(rete_neurale.state_dict(), weights_save_path)


# ======================================================
#             ESECUZIONE RETE NEURALE
# ======================================================

# Indipendentemente dalla scelta effettuata si entra in modalità valutazione sulle missioni scelte come test
rete_neurale.eval()                                                     # Si imposta la rete neurale in modalità valutazione

try:
    rete_neurale.load_state_dict(torch.load(weights_save_path))             # Si richiamano i pesi salvati nella fase di training
except FileNotFoundError:
    raise FileNotFoundError(f"Il file {weights_save_path}, contenente i pesi della rete neurale, non esiste. Eseguire il training.") from None

# Si richiama la funzione per creare il dizionario contenente le missioni suddivise in batch unitarie
test_dataset = Dataset(tensor_dict, sensors_frequencies, test_missions, mode='test')

# -----------------------------------------------------
#              INIZIALIZZAZIONE VARIABILI
# -----------------------------------------------------

# Si recuperano le statistiche del GPS salvate nel pre-processing --> trasformandole in tensori per compatibilità con output rete neurale
GPS_mean = torch.tensor(pre_processing.normalization_parameters['GPS']['media'])
GPS_std = torch.tensor(pre_processing.normalization_parameters['GPS']['deviazione standard'])

post_processing = PostProcessing()

# Raccolta di tutte le traiettorie complete individuate per ogni missione
single_path_intervals = {('0', '4'): [(177, 414), (635, 980), (1262, 2366), (2630, 3402)],
                         ('0', '8'): [(268, 611), (891, 1201), (1367, 1848), (2010, 2525)],
                         ('0', '12'): [(199, 693), (832, 1176), (1465, 2440), (2659, 3469)],
                         ('0', '16'): [(0, 697), (875, 1151), (1334, 1808), (1985, 2223)]}

# Si itera sulle missioni presenti nella lista definita appositamente
NN_displacements_dict = {}              # Dizionario che contiene gli elementi di posizione stimati dalla rete per la singola missione [m] --> nella forma {('0', '4'): {'Traiettoria1': [], 'Traiettoria2': [], 'Traiettoria3': [], 'Traiettoria4': []}, ('0', '8'): {}.....}
GPS_displacements_dict = {}             # Dizionario che contiene gli elementi di posizione GPS target ottenuti dalle batch per la singola missione [m] --> nella forma {('0', '4'): {'Traiettoria1': [], 'Traiettoria2': [], 'Traiettoria3': [], 'Traiettoria4': []}, ('0', '8'): {}.....}
RMSE_NN = []                            # Lista che contiene i valori di RMSE medi per tutte le missioni

# Si genera il dizionario che contiene le liste di waypoints
post_processing.waypoints_dict_building(ROOT_DIR)

for (trajectory, combination) in test_missions:

    # Si richiama la lista delle batch unitarie per la missione di iterazione
    mission = (trajectory, combination)
    mission_batch = test_dataset[mission]

    # Si genera il dataloader per la singola missione
    test_loader = torch.utils.data.DataLoader(mission_batch, batch_size=batch_size_test, shuffle=False)

    # Si inizializzano i sotto-dizionari associati a ogni missione
    if mission not in NN_displacements_dict:
        NN_displacements_dict[mission] = {}
    if mission not in GPS_displacements_dict:
        GPS_displacements_dict[mission] = {}

    # Richiamare la lista di waypoints opportuna

    # Si disabilita da questo momento in poi il calcolo del gradiente --> per risparmiare memora
    with ((torch.no_grad())):

        # -----------------------------------------------------
        #              INIZIALIZZAZIONE VARIABILI
        # -----------------------------------------------------

        # Si itera per ogni batch del test_loader (sono batch_size secondi)
        GPS_starting_coordinates = pre_processing.resampled_trajectories_dataframes[trajectory][combination][['NED_North [m]', 'NED_East [m]']].iloc[0].to_numpy(dtype=np.float32)      # Definizione della posizione iniziale dal dataframe pandas ricampionato
        GPS_NED_coordinates = torch.tensor(GPS_starting_coordinates, dtype=torch.float32).unsqueeze(0)                                                                                  # Trasformazione del vettore posizione in un tensore pytorch --> con unsqueeze si rende compatibile con le dimensioni della batch [1,2]
        NN_NED_coordinates = {}                                                                                                                                                         # Inizializzazione dizionario per l'immagazzinamento dei valori di posizione per ogni traiettoria della missione --> ha forma {'Traiettoria1': [], 'Traiettoria2': [],....}

        # -----------------------------------------------------
        #      STIMA POSIZIONE PER OGNI SECONDO DI MISSIONE
        # -----------------------------------------------------

        t_iteration = 0
        for batch in test_loader:

            IMU_data = batch[0]
            DVL_data = batch[1]
            GPS_data = batch[2].squeeze(1)

            NN_displacement_estimation = rete_neurale.forward(IMU_data, DVL_data)

            # Si ritrasformano i dati in metri non-normalizzati, si sommano al valore di posizione precedente e si aggiungono alla lista complessiva per ottenere la traiettoria
            NN_displacement_meters = (NN_displacement_estimation * GPS_std) + GPS_mean
            #NN_NED_coordinates += NN_displacement_meters

            # Si ripete lo stesso procedimento con la batch di iterazione dei dati GPS di partenza
            GPS_displacement_meters = (GPS_data * GPS_std) + GPS_mean
            GPS_NED_coordinates += GPS_displacement_meters

            # -----------------------------------------------------
            #        AGGIORNAMENTO STIMA COORDINATE NED
            # -----------------------------------------------------
            for path_number, time_interval in enumerate(single_path_intervals[mission]):

                # Si inizializzano le liste associate a ogni traiettoria
                if f'Percorso_{path_number+1}' not in NN_displacements_dict[mission]:
                    NN_displacements_dict[mission][f'Percorso_{path_number+1}'] = []
                if f'Percorso_{path_number+1}' not in GPS_displacements_dict[mission]:
                    GPS_displacements_dict[mission][f'Percorso_{path_number+1}'] = []

                # Si definiscono istanti di inizio e fine percorso
                t_start_path = time_interval[0]
                t_end_path = time_interval[1]

                # Se il secondo di missione analizzato si trova nell'intervallo occorre salvarla in un apposito dizionario
                if t_start_path <= t_iteration <= t_end_path:

                    # Si imposta il valore iniziale di posizione assoluta a inizio traiettoria --> per forzare inizio coincidente
                    if t_iteration == t_start_path:
                        NN_NED_coordinates[f'Percorso_{path_number + 1}'] = GPS_NED_coordinates.clone()
                    else:
                        NN_NED_coordinates[f'Percorso_{path_number + 1}'] += NN_displacement_meters

                    NN_displacements_dict[mission][f'Percorso_{path_number + 1}'].append(NN_NED_coordinates[f'Percorso_{path_number + 1}'].clone())
                    GPS_displacements_dict[mission][f'Percorso_{path_number + 1}'].append(GPS_NED_coordinates.clone())

            t_iteration += batch_size_test

        # -----------------------------------------------------
        #             GESTIONE RISULTATI PRODOTTI
        # -----------------------------------------------------

        # Si itera su ogni elemento presente nei dizionari di traiettoria per la missione di iterazione (e per estensione RMSE)
        for path_name, NN_coordinates_list in NN_displacements_dict[mission].items():

            # Si richiama la traiettoria target associata
            GPS_coordinates_list = GPS_displacements_dict[mission][path_name]

            # Si calcola l'errore RMSE sui vettori prodotti
            RMSE_NN_tot, RMSE_NN_N, RMSE_NN_E = post_processing.RMSE_estimation(GPS_coordinates_list, NN_coordinates_list)
            #RMSE_NN.append(RMSE_NN_tot)
            print(f'\nPer la missione {trajectory} - {combination} e il percorso {path_name}, si hanno i seguenti valori di Root Mean Squared Error:')

            # Si realizza il grafico associato alla singola traiettoria
            figure_title = f'Confronto tra posizione target e predette per la missione {mission}'
            post_processing.plot_confronto_traiettoria(GPS_coordinates_list, NN_coordinates_list, mission, path_name, figure_title)




