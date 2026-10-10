"""     IMPLEMENTAZIONE RETE NEURALE NAVNET
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             """

'''

All'interno del main si effettuano le seguenti operazioni in ordine:
    - pre-processing dei dati, ottenendo i tensori utili per la rete neurale --> tiene conto delle differenti frequenze dei sensori:
        -- 10 Hz per DVL
        -- 5 Hz per IMU e REF
        -- 1 Hz per GPS
    - definizione delle batch di iterazione per la rete neurale --> tramite class Dataset si definisce la singola batch per ogni blocco e poi si uniscono insieme
    - esecuzione rete neurale per la stima della posizione
    - ciclo di addestramento sui dati per l'ottimizzazione dei pesi utilizzati
                                                                                                            '''

import os                                   # Libreria per l'utilizzo di Python integrato nel sistema
import torch                                # Libreria per il Deep Learning
from Analisi_dati import PreProcessing      # Importo il modulo per eseguire il pre-processing dei dati da utilizzare per la rete
from Neural_network import Dataset          # Importo il modulo per definire le batch dei blocchi di sensore per ogni secondo di missione
from Neural_network import NavNet           # Importo qui la rete neurale
from Analisi_dati import PostProcessing     # Importo il modulo per eseguire il post-processing dei dati prodotti dalla rete
import numpy as np                          # Libreria per i calcoli matematici
import sys                                  # Libreria che permette di interagire con il Runtime di Python
import pickle                               # Libreria per la gestione del formato di salvataggio pickle
import json                                 # libreria per la gestione del formato di salvataggio json


def stima_trapezoidale(velocita, timestamps, ROOT_DIR, pos_iniziale=0):

    pos = [pos_iniziale]
    for i in range(len(velocita) - 1):
        dt = timestamps[i+1] - timestamps[i]
        # Formula trapezoidale
        spostamento = 0.5 * (velocita[i] + velocita[i+1]) * dt
        pos.append(pos[-1] + spostamento)
    return np.array(pos)


# ======================================================
#              INIZIALIZZAZIONE VARIABILI
# ======================================================

# Variabili legate alla configurazione della rete neurale
batch_size = 32                                                                         # Numero di secondi di missione forniti contemporaneamente alla rete
batch_size_test = 1                                                                     # Per il test si usa una batch size di 1s --> si ottiene la stima della posizione per ogni secondo di navigazione (coincidenza perfetta con batch GPS)
num_epoch = 60                                                                          # Numero di epoche di iterazione per l'addestramento
network_config = {'INS': 9, 'DVL': 3, 'GPS': 2}                                         # Dizionario di configurazione della rete neurale --> utile per impostare ordine, numero e nome dei rami associati a LSTM e SAM
hidden_size = 100                                                                       # Numero di neuroni appartenenti al layer nascosto del LSTM

sensors_frequencies = {'IMU': 5, 'DVL': 10, 'Depth': 5, 'DepthVel': 5, 'MOT': 5, 'V_ref': 5, 'GPS': 1}             # Frequenze associate ai sensori

# Definizione delle cartelle di riferimento per il progetto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                                 # Cartella in cui sono contenuti gli script Python
ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)                                             # Cartella più esterna in cui è contenuta la cartella degli script e le restanti, sia di input che di output
DF_DICT_DIR = os.path.join(ROOT_DIR, 'Dizionari_datasets')                              # Cartella all'interno della quale verranno salvati i dizionari in formato pickle e i Dataset in formato Excel
os.makedirs(DF_DICT_DIR, exist_ok=True)                                                 # Si genera la cartella di destinazione dei file di salvataggio pickle con i vari dataframe
RAW_DF_DIR = os.path.join(ROOT_DIR, 'Telemetrie')                                       # Cartella all'interno della quale sono contenuti tutti i file .csv e .txt necessari al pre-processing

# File di input/output necessarie al pre-processing
file_dataset_raw = os.path.join(DF_DICT_DIR, 'Dataset_globale.pkl')                     # File pickle di output/input contenente i dati sul dizionario globale dei dataframe generato al termine del pre-processing
file_dataset_resampled = os.path.join(DF_DICT_DIR, 'Dataset_resampled.pkl')             # File pickle di output/input contenente i dati sul dizionario globale dei dataframe resampled generato al termine del pre-processing
file_database_sensors = os.path.join(DF_DICT_DIR, 'Dataset_resampled_scomposto.pkl')    # File pickle di output/input contenente i dati sul dizionario suddiviso per i vari sensori dei dataframe generato al termine del pre-processing
file_dataset_interpolated = os.path.join(DF_DICT_DIR, 'Dataset_interpolated.pkl')       # File pickle di output/input contenente i dati sul dizionario globale dei dataframe interpolati al termine del pre-processing
file_config = os.path.join(ROOT_DIR, "Telemetrie/Matrice_configurazioni.txt")           # File di testo di input contenente un elenco delle combinazioni condizione ambientale/operativa per ogni traiettoria simulata
weights_save_path = os.path.join(ROOT_DIR, "NavNet_weights.pkl")                        # File pickle contenente i pesi della rete neurale
file_database_waypoints = os.path.join(DF_DICT_DIR, 'Waypoints_dict.pkl')               # File pickle contenente i waypoints per ogni missione
file_test_missions_dict = os.path.join(DF_DICT_DIR, 'Test_missions_dict.pkl')           # File pickle contenente il dizionario con le missioni prescelte per il test sotto forma di (trajectory, combination) e gli intervalli temporali associati a ogni ripetizione della traiettoria
empy_list_file = os.path.join(DF_DICT_DIR, 'empy_list.json')                            # File json contenente la lista delle missioni ritenute non valide
file_tensor_dict = os.path.join(DF_DICT_DIR, 'Tensor_dict.pt')                          # File pickle di pytorch contenente il dizionario dei tensori
file_norm_parameters_dict = os.path.join(DF_DICT_DIR, 'Norm_parameters_dict.pkl')       # File pickle contenente il dizionario con tutti i valori di media e std per ogni blocco di sensori

# Si inizializzano i metodi prescelti e il dizionario per l'immagazzinamento dei dataset associati
possible_processing_methods = ['resampling', 'interpolazione']              # Lista di metodi per il trattamento dei dati di telemetria pura
processed_datasets_dict = {}                                                # Dizionario che conterrà i dizionari processati con i vari metodi prescelti --> nella forma {'metodo 1': {}, 'metodo 2': {}, ...}

files_processed_datasets = {'resampling': file_dataset_resampled, 'interpolazione': file_dataset_interpolated}           # Dizionario con i percorsi di salvataggio di tutti i metodi di processing disponibili


# ======================================================
#             PRE-PROCESSING TELEMETRIA
# ======================================================

print('\n\n===================================================')
print('============ AVVIO PRE-PROCESSING DATI ============')
print('===================================================')

# Si richiamano le classi associate al pre-processing e al post-processing
pre_processing = PreProcessing()
post_processing = PostProcessing()

print(f"\nSelezione parametri di pre-processing:")

# -----------------------------------------------------
#         ESTRAZIONE DATAFRAMES GREZZI DA CSV
# -----------------------------------------------------

print('\n\n----------------------------------------------------')
print('---------- CARICAMENTO DATI DI TELEMETRIA ----------')
print('----------------------------------------------------')

raw_dataset = {}
while True:

    generate_new_dataset = input(f"\n  Si vuole generare un nuovo Dataset contenente i dati di telemetria (S/N/EXIT)?").upper()

    if generate_new_dataset == 'S':

        pre_processing.csv_analysis(file_config, RAW_DF_DIR)
        pre_processing.save_raw_datasets_dict('pickle', DF_DICT_DIR, file_dataset_raw)
        raw_dataset = pre_processing.raw_dataset

        while True:

            save_raw_dataset = input(f"\n    Si vuole salvare in formato Excel il Dataset prodotto (S/N/EXIT)?").upper()

            if save_raw_dataset == 'S':

                pre_processing.save_raw_datasets_dict('excel', DF_DICT_DIR, file_dataset_raw)
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
        if os.path.isfile(file_dataset_raw) and not os.stat(file_dataset_raw).st_size == 0:
            print(f"    Utilizzo del Dataset precedentemente generato e salvato in {file_dataset_raw}.")
            with open(file_dataset_raw, 'rb') as f:
                raw_dataset = pickle.load(f)
            break
        else:
            print(f"    Il file {file_dataset_raw} risulta non presente o vuoto. Verificare la presenza o generare un nuovo dataset.")

    elif generate_new_dataset == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")


# -----------------------------------------------------
#         GESTIONE SCELTA SU NUOVA ELABORAZIONE
# -----------------------------------------------------

print('\n\n---------------------------------------------------')
print('------------- ELABORAZIONE DATAFRAMES -------------')
print('---------------------------------------------------')

processing_methods = ['resampling']
while True:

    process_dataset = input(f"\n  Si vuole procedere con una nuova elaborazione dei dati di telemetria (S/N/EXIT)?").upper()

    # Se si è generato un nuovo Dataset grezzo è inevitabile rieseguire anche l'elaborazione dello stesso
    if generate_new_dataset == 'S' and process_dataset != 'EXIT':
        process_dataset = 'S'

    if process_dataset == 'S':

        # Si itera su ogni metodo selezionato in precedenza --> per ognuno si effettua l'analisi tramite apposita funzione
        for processing_method in processing_methods:

            if processing_method == 'resampling':
                pre_processing.raw_dataframes_processing(raw_dataset, sensors_frequencies)
                pre_processing.save_processed_dataset_dict('pickle', DF_DICT_DIR, file_dataset_resampled, 'resampling')
                pre_processing.save_processed_dataset_dict('pickle', DF_DICT_DIR, file_database_sensors, 'sensors_blocks')
                processed_datasets_dict[processing_method] = pre_processing.resampled_dataset

            elif processing_method == 'interpolazione':
                pre_processing.dataframe_interpolation(raw_dataset)
                pre_processing.save_processed_dataset_dict('pickle', DF_DICT_DIR, file_dataset_interpolated, 'interpolazione')
                processed_datasets_dict[processing_method] = pre_processing.interpolated_dataset

            # Gestione eventuale salvataggio in formato Excel
            while True:

                save_processed_dataset = input(f"\n    Si desidera salvare il Dataset trattato con {processing_method.upper()} in formato Excel (S/N/EXIT)?").upper()

                if save_processed_dataset == 'S':

                    pre_processing.save_processed_dataset_dict('excel', DF_DICT_DIR, files_processed_datasets, processing_method)
                    break

                elif save_processed_dataset == 'N':
                    print(f"      Prosecuzione analisi senza salvare i Dataset elaborati prodotti.")
                    break

                elif save_processed_dataset == 'EXIT':
                    print(f"      Chiusura del programma forzata da utente in corso...")
                    sys.exit()

                else:
                    print(f"      Input inserito non valido. Ripetere la scelta con s, n o exit.")

        # Salvataggio in formato json la lista delle missioni non valide
        with open(empy_list_file, "w", encoding="utf-8") as f:
            json.dump(pre_processing.empty_missions_list, f)
        break

    elif process_dataset == 'N':

        file_exists = True

        # Caricamento dataset processati
        for processing_method in processing_methods:
            dict_file_path = files_processed_datasets[processing_method]
            if os.path.isfile(dict_file_path) and os.stat(dict_file_path).st_size > 0:
                print(f"    Utilizzo del Dataset elaborato con {processing_method.upper()} precedentemente generato e salvato in {dict_file_path}.")
                with open(dict_file_path, 'rb') as f:
                    processed_datasets_dict[processing_method] = pickle.load(f)
            else:
                print(f"    Il file {dict_file_path} risulta non presente o vuoto. Ricontrollare o rieseguire l'analisi dei dati grezzi. ")
                file_exists = False

        # Caricamento lista delle missioni non valide
        if os.path.isfile(empy_list_file) and os.stat(empy_list_file).st_size > 0:
            print(f"    Caricamento della lista di missioni non valide precedentemente generato e salvato in {empy_list_file}.")
            with open(empy_list_file, 'r', encoding='utf-8') as f:
                pre_processing.empty_missions_list = json.load(f)
        else:
            print(f"    Impossibile caricare la lista di missioni non valide dal file {empy_list_file}. Verificare.")
            file_exists = False

        # Caricamento dataset scomposto a blocchi
        if os.path.isfile(file_database_sensors) and os.stat(file_database_sensors).st_size > 0:
            print(f"    Utilizzo del Dataset scomposto per sensori elaborato con resampling precedentemente generato e salvato in {file_database_sensors}.")
            with open(file_database_sensors, 'rb') as f:
                pre_processing.sensors_divided_dataset = pickle.load(f)
        else:
            print(f"    Impossibile caricare la lista di missioni non valide dal file {empy_list_file}. Verificare.")
            file_exists = False

        # Solo se tutti i file pickle di interesse sono stati letti correttamente si esce dal ciclo
        if file_exists:
            break

    elif process_dataset == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")

resampled_dataset = processed_datasets_dict['resampling']


# -----------------------------------------------------
#        ANALISI WAYPOINTS E TRAIETTORIA IDEALE
# -----------------------------------------------------

print('\n\n--------------------------------------------------------')
print('--------- CALCOLO ELEMENTI PER POST-PROCESSING ---------')
print('--------------------------------------------------------')

# Si gestiste il caricamento o la generazione dei dizionari con le informazioni utili al post-processing delle traiettorie in base a quanto scelto da utente --> se si è eseguita una nuova analisi in una delle due fasi precedenti si rigenera
while True:

    new_waypoints_dict = input(f"\n  Si vuole generare un nuovo dizionario dei Waypoints associati alle varie traiettorie (S/N/EXIT)?").upper()

    if new_waypoints_dict == 'S':

        post_processing.waypoints_dict_building(ROOT_DIR)       # Si genera il dizionario che contiene le liste di waypoints
        post_processing.write_on_file(ROOT_DIR)                 # Si salva il dizionario sotto forma testuale
        post_processing.ideal_trajectories_plot(ROOT_DIR)       # Si generano i grafici delle traiettorie ideali
        break

    elif new_waypoints_dict == 'N':

        # Controllo esistenza del file pickle --> se esiste si carica dal pickle e si salva nella variabile associata alla classe di post-processing (per averla anche nelle successive funzioni)
        if os.path.isfile(file_database_waypoints) and not os.stat(file_database_waypoints).st_size == 0:
            print(f"    Utilizzo del dizionario di waypoints precedentemente generato e salvato in {file_database_waypoints}.")
            with open(file_database_waypoints, 'rb') as f:
                post_processing.waypoints_dict = pickle.load(f)
            break
        else:
            print(f"    Il file {file_database_waypoints} risulta non presente o vuoto. Verificare la presenza o generare un nuovo dizionario.")

    elif new_waypoints_dict == 'EXIT':
        print(f"    Chiusura del programma forzata da utente in corso...")
        sys.exit()

    else:
        print(f"    Input inserito non valido. Ripetere la selezione inserendo s, n o exit.")


# Per la definizione delle missioni di test e la scomposizione in singoli percorsi di ognuna si asseconda la scelta fatta per la generazione di nuovi Dataset --> in tal caso è obbligatorio per evitare sfasamenti temporali
test_missions_dict = {}
while True:

    if generate_new_dataset == 'S' or process_dataset == 'S':

        for processing_method in processing_methods:

            post_processing.test_missions_definition(resampled_dataset, ROOT_DIR)
            test_missions_dict = post_processing.test_missions_dict

        break

    else:

        if os.path.isfile(file_test_missions_dict) and os.stat(file_test_missions_dict).st_size > 0:

            try:
                with open(file_test_missions_dict, 'rb') as f:
                    test_missions_dict = pickle.load(f)

                break

            except FileNotFoundError:
                raise FileNotFoundError(f"   [ERROR] Errore nell'apertura del file pickle per le missioni di test e i relativi intervalli temporali.")


# -----------------------------------------------------
#         TRASFORMAZIONE DATAFRAMES IN TENSORI
# -----------------------------------------------------

print('\n\n----------------------------------------------------------------------')
print('--------- DEFINIZIONE TENSORI PYTORCH E NORMALIZZAZIONE DATI ---------')
print('----------------------------------------------------------------------')

print(f"\n  Trasformazione Dataset processato in tensore pytorch in corso:")

tensor_dict = {}
if generate_new_dataset == 'S' or process_dataset == 'S':

    pre_processing.dataframe_to_tensor(pre_processing.sensors_divided_dataset, test_missions_dict)      # Si generano i tensori pytorch --> si genera in output un dizionario in cui per ogni missione sono presenti 3 tensori, uno per ogni frequenza di misurazione presente
    tensor_dict = pre_processing.pytorch_tensor_dict                                                    # Si richiama il dizionario di tensori da usare per la rete neurale --> ogni tensore è una lista ordinata (per sensore) contenente le misurazioni di tutte missioni
    torch.save(tensor_dict, file_tensor_dict)                                                           # Si salva il dizionario per iterazioni successive
    with open(file_norm_parameters_dict, 'wb') as f:
        pickle.dump(pre_processing.normalization_parameters_dict, f)

else:

    if (os.path.isfile(file_tensor_dict) and os.stat(file_tensor_dict).st_size > 0) and (os.path.isfile(file_norm_parameters_dict) and os.stat(file_norm_parameters_dict).st_size > 0):

        print(f"    Caricamento del dizionario contenente il tensore pytorch di ogni missione precedentemente generato e salvato in {file_tensor_dict}.")
        tensor_dict = torch.load(file_tensor_dict)

        print(f"    Caricamento del dizionario contenente i parametri di normalizzazione di ogni sensore precedentemente generato e salvato in {file_norm_parameters_dict}.")
        with open(file_norm_parameters_dict, 'rb') as f:
            pre_processing.normalization_parameters_dict = pickle.load(f)

    else:

        print(f"  Impossibile trovare il file di salvataggio del dizionario dei tensori.")


print('\n\n---------------------------------------------------------')
print('----------------- RECAP PRE-PROCESSING ------------------')
print('---------------------------------------------------------')

# Si stampa un resoconto del pre-processing con le missioni ritenute non valide e non utilizzate per l'analisi
print(f"\nFase di pre-processing delle telemetrie conclusa. Le seguenti missioni sono state escluse dai database prodotti:")
print(f"  {pre_processing.empty_missions_list}")


# ======================================================
#           INIZIALIZZAZIONE RETE NEURALE
# ======================================================

# Inizializzazione del modello di rete neurale
rete_neurale = NavNet(network_config, hidden_size)                  # Si inizializza la rete principale
rete_neurale.initialize_weights()                                   # Si inizializzano i pesi tramite funzione definita

# Scheduler --> dimezza il learning rate se la Val_Loss non scende per 5 epoche
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rete_neurale.optimizer, mode='min', factor=0.5, patience=3)


# ======================================================
#                TRAINING RETE NEURALE
# ======================================================

print('\n\n=========================================================')
print('=============== ADDESTRAMENTO RETE NEURALE ==============')
print('=========================================================')

while True:

    do_training = input(f'\nSi desidera effettuare il training della rete (S/N/EXIT)?').upper()

    if do_training == 'S':

        # Si richiama la classe Dataset per definire la lunghezza della batch unitaria per ogni blocco di sensori --> successivamente tramite libreria pytorch si generano batch di forma [32, frequenza, numero di colonne]
        training_dataset = Dataset(tensor_dict, sensors_frequencies, test_missions_dict, mode='train')
        validation_dataset = Dataset(tensor_dict, sensors_frequencies, test_missions_dict, mode='validazione')

        # Si trasformano i subset definiti nei dataloader necessari alla rete neurale --> si aggregano più blocchi in base alla batch_size definita
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)                 # Definizione del dataloader associato al training
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)            # Definizione del dataloader associato alla validazione

        # Si calcola la loss e si effettua la backpropagation per la stima dei migliori pesi
        print(f'\n  Inizio training rete neurale:')
        loss_train = []
        loss_val = []
        best_epoch = -1
        best_val_loss = float('inf')
        data = {}
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

                # Si suddividono le batch per ramo della rete neurale
                for key in network_config:

                    if key == 'GPS':
                        data[key] = batch[key].squeeze(1)
                    else:
                        data[key] = batch[key]

                #DVL_data = batch[0]                 # Batch di dimensioni [32, 10, 3]
                #IMU_data = batch[1]                 # Batch di dimensioni [32, 5, 10] --> contiene i 3 valori IMU, i 4 sensori MOT (FV, FL, FW1, FW2) e i 2 di velocità reference (Vxref, omega_yref)
                #dati_gps = batch[2].squeeze(1)      # Batch di dimensioni [32, 1, 2] --> si vuole avere dimensione [32, 2] per congruenza con output rete neurale --> si elimina una dimensione dalla batch

                # Si richiama la funzione di aggiornamento pesi
                training_batch_loss = rete_neurale.backpropagation(data)

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
                epoch_batches_number = 0
                for batch in validation_loader:

                    for key in network_config:

                        if key == 'GPS':
                            data[key] = batch[key].squeeze(1)
                        else:
                            data[key] = batch[key]

                    NN_displacement_estimation = rete_neurale.forward(data)

                    # Calcolo della loss function
                    validation_batch_loss = rete_neurale.loss_criterion(NN_displacement_estimation)
                    validation_epoch_loss += validation_batch_loss
                    epoch_batches_number += 1

                # Calcolo della loss media sulla singola epoca
                loss_validation_epoch = validation_epoch_loss / epoch_batches_number
                loss_val.append(loss_validation_epoch)
                print(f"    Loss media di validazione calcolata per l'epoca {epoch}: {loss_validation_epoch:.4f}")

                # Si salvano i pesi aggiornati in un dizionario pytorch apposito per l'epoca a cui è associata la minore perdita di validazione
                if loss_validation_epoch < best_val_loss:
                    best_val_loss = loss_validation_epoch
                    best_epoch = epoch
                    torch.save(rete_neurale.state_dict(), weights_save_path)

            # -----------------------------------------------------
            #      AGGIORNAMENTO PARAMETRI TRAMITE SCHEDULER
            # -----------------------------------------------------
            scheduler.step(loss_validation_epoch)

        # Si realizza il grafico complessivo
        post_processing.plot_loss_function(loss_train, loss_val, num_epoch, ROOT_DIR)

        break

    elif do_training == 'N':

        print(f"  Prosecuzione nell'esecuzione delle rete neurale con i pesi salvati in un precedente addestramento all'interno di {weights_save_path}.")
        break

    elif do_training == 'EXIT':
        print(f"  Chiusura del programma forzata da utente in corso...")
        sys.exit()


    else:
        print(f"  Input inserito non valido. Ripetere la selezione con s, n o exit.")


# ======================================================
#             ESECUZIONE RETE NEURALE
# ======================================================

print('\n\n=========================================================')
print('================ ESECUZIONE RETE NEURALE ================')
print('=========================================================')

# Indipendentemente dalla scelta effettuata si entra in modalità valutazione sulle missioni scelte come test
rete_neurale.eval()                                                     # Si imposta la rete neurale in modalità valutazione

# Si richiamano i pesi salvati nella fase di training
try:
    rete_neurale.load_state_dict(torch.load(weights_save_path))
except FileNotFoundError:
    raise FileNotFoundError(f"Il file {weights_save_path}, contenente i pesi della rete neurale, non esiste. Eseguire il training.") from None

# Si richiama la funzione per creare il dizionario contenente le missioni suddivise in batch unitarie
test_dataset = Dataset(tensor_dict, sensors_frequencies, test_missions_dict, mode='test')

# -----------------------------------------------------
#              INIZIALIZZAZIONE VARIABILI
# -----------------------------------------------------

# Si recuperano le statistiche del GPS salvate nel pre-processing --> trasformandole in tensori per compatibilità con output rete neurale
GPS_mean = torch.tensor(pre_processing.normalization_parameters_dict['GPS']['media'])
GPS_std = torch.tensor(pre_processing.normalization_parameters_dict['GPS']['deviazione standard'])

# Si itera sulle missioni presenti nella lista definita appositamente
NN_displacements_dict = {}              # Dizionario che contiene gli elementi di posizione stimati dalla rete per la singola missione [m] --> nella forma {('0', '4'): {'Traiettoria1': [], 'Traiettoria2': [], 'Traiettoria3': [], 'Traiettoria4': []}, ('0', '8'): {}.....}
GPS_displacements_dict = {}             # Dizionario che contiene gli elementi di posizione GPS target ottenuti dalle batch per la singola missione [m] --> nella forma {('0', '4'): {'Traiettoria1': [], 'Traiettoria2': [], 'Traiettoria3': [], 'Traiettoria4': []}, ('0', '8'): {}.....}
RMSE_NN = []                            # Lista che contiene i valori di RMSE medi per tutte le missioni

# -----------------------------------------------------
#       DEFINIZIONE DATALOADER PER RETE NEURALE
# -----------------------------------------------------

for (trajectory, combination) in test_missions_dict:

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

    # Si disabilita da questo momento in poi il calcolo del gradiente --> per risparmiare memora
    with torch.no_grad():

        # -----------------------------------------------------
        #   INIZIALIZZAZIONE VETTORI SALVATAGGIO TRAIETTORIE
        # -----------------------------------------------------

        # Si itera per ogni batch del test_loader (sono batch_size secondi)
        GPS_starting_coordinates = resampled_dataset[trajectory][combination][['NED_East [m]', 'NED_North [m]']].iloc[0].to_numpy(dtype=np.float32)      # Definizione della posizione iniziale dal dataframe pandas ricampionato
        GPS_NED_coordinates = torch.tensor(GPS_starting_coordinates, dtype=torch.float32).unsqueeze(0)                                                    # Trasformazione del vettore posizione in un tensore pytorch --> con unsqueeze si rende compatibile con le dimensioni della batch [1,2]
        NN_NED_coordinates = {}                                                                                                                           # Inizializzazione dizionario per l'immagazzinamento dei valori di posizione per ogni traiettoria della missione --> ha forma {'Traiettoria1': [], 'Traiettoria2': [],....}
        GPS_abs_origin = resampled_dataset[trajectory][combination][['UTM_East [m]', 'UTM_North [m]']].iloc[0].to_numpy(dtype=np.float64)

        # -----------------------------------------------------
        #      STIMA POSIZIONE PER OGNI SECONDO DI MISSIONE
        # -----------------------------------------------------

        t_iteration = 0
        data = {}
        for batch in test_loader:

            for key in network_config:

                if key == 'GPS':
                    data[key] = batch[key].squeeze(1)
                else:
                    data[key] = batch[key]

            NN_displacement_estimation = rete_neurale.forward(data)

            # Si ritrasformano i dati in metri non-normalizzati, si sommano al valore di posizione precedente e si aggiungono alla lista complessiva per ottenere la traiettoria
            NN_displacement_meters = (NN_displacement_estimation * GPS_std) + GPS_mean
            #NN_NED_coordinates += NN_displacement_meters

            # Si ripete lo stesso procedimento con la batch di iterazione dei dati GPS di partenza
            GPS_displacement_meters = (data['GPS'] * GPS_std) + GPS_mean
            GPS_NED_coordinates += GPS_displacement_meters

            # -----------------------------------------------------
            #        AGGIORNAMENTO STIMA COORDINATE NED
            # -----------------------------------------------------
            for path_number, time_interval in enumerate(test_missions_dict[mission]):

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
            #RMSE_NN.append(RMSE_NN_tot)
            post_processing.NN_recap_estimation(GPS_coordinates_list, NN_coordinates_list, ROOT_DIR, mission, path_name)
            post_processing.first_mission = False

            # Si realizza il grafico associato alla singola traiettoria
            figure_title = f'Confronto tra posizione target e predette per la missione {mission}'
            post_processing.real_trajectories_plot(GPS_coordinates_list, NN_coordinates_list, GPS_abs_origin, mission, path_name, figure_title, ROOT_DIR)




