'''     IMPLEMENTAZIONE RETE NEURALE NAVNET
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             '''

''' La classe Dataset suddivide i tensori generati nel pre-processing in batch che tengono conto del singolo riferimento temporale considerato (1 secondo), senza avere più 
    distinzione tra le varie missioni per il training --> ciò non vale per le missioni di test prescelte che sono estratte e trattate con funzione separata, ottenendo così 
    un secondo dataset analogo per struttura ma suddiviso per missione

    La rete neurale si compone di:
    - due blocchi LSTM in serie --> producono gli hidden state a_t (frequenza(blocco)*n(hidden_layers)
    - un simplified attention mechanism --> produce il vettore c per ogni blocco variabili (1*n(hidden_layers_LSTM)) --> creato in modo da verificare quale 
        decimo di secondo dell'osservazione è più importante
    - concatenazione context vectors --> si devono unire i vettori arrivando a dimensione n(blocchi)*n(hidden_layers_LSTM)
    - 2 FC layers --> --> producono il vettore degli outputs (2 elementi)
    Il tempo unitario di riferimento per la rete è 1 secondo e si aggiorna tramite backward propagation con metodo Adams e perdita calcolata con metodo Euclideo
    
    La classe Post-Processing presenta varie funzioni utili all'analisi dei risultati dell'esecuzione della rete neurale, come la rappresentazione dei grafici e il calcolo 
    dell'errore medio (RMSE) --> è, inoltre, presente una funzione opzionale che permette di separare automaticamente le singole ripetizioni di traiettoria all'interno di 
    una missione (con criterio )
    '''

import numpy as np                              # Libreria per i calcoli matematici
import torch                                    # Libreria per il Deep Learning
import torch.nn as nn                           # Libreria utilizzata per le proprietà nelle reti neurali
import torch.nn.functional as F                 # Libreria specifica per le funzioni come softmax
import torch.optim as optim                     # Libreria per il metodo di ottimizzazione dei pesi
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

torch.manual_seed(0)            # Si imposta il valore dei seed di partenza per l'addestramento della rete


# Si definisce una nuova classe che prende in input i tensori suddivisi per blocco e missione (generati da PreProcessing) e li trasforma in singole batch di dati (1 secondo) da fornire alla rete
class Dataset(torch.utils.data.Dataset):

    # Si inizializza la struttura comune degli elementi appartenenti alla classe --> in input riceve 3 liste di tensori
    def __init__(self, tensor_dict, sensor_frequencies, test_missions, mode):

        self.mode = mode

        # Si richiama la funzione opportuna in base alla modalità prescelta --> si ottiene la lista delle batch unitarie
        if mode == 'train':

            self.training_batches_list = []                                                             # Nella forma [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], [...], ....] --> struttura ripetuta per tutti i secondi di ogni missione
            self.training_unitary_batches_building(tensor_dict, sensor_frequencies, test_missions)

        elif mode == 'test':

            self.test_batches_dict = {}                                                                 # Nella forma {'Missione1': [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)],...], 'Missione2': []...} --> struttura ripetuta all'interno di ogni missione per tutti i secondi della missione e per tutte le missioni
            self.test_unitary_batches_building(tensor_dict, sensor_frequencies, test_missions)

    # Funzione che permette di ricevere in input il dizionario dei tensori prodotto e restituire una lista di liste, con le misurazioni normalizzate ricevute per singolo secondo di missione
    def training_unitary_batches_building(self, tensor_dict, sensors_frequencies, test_missions):

        for trajectory, mission_dict in tensor_dict.items():

            print(f"\n  Scomposizione del tensore per le missioni associate alla traiettoria {trajectory} in corso:")

            for combination, sensors_dict in mission_dict.items():

                # Si definisce la durata di missione [s] --> si prende come riferimento la colonna di GPS che ha 1 misura al secondo
                mission_time = len(sensors_dict['GPS'])-1

                # Verifica della missione --> se il numero di missione è nella lista fornita in input si salta il resto del ciclo passando alla missione dopo
                if (trajectory, combination) in test_missions:
                    continue

                for second in range(mission_time):

                    # Si inizializza una lista vuota che conterrà n tensori --> un tensore per ogni blocco sensori contenente le relative misurazioni di 1 secondo
                    single_second_batches_list = []

                    # Si inizializza la variabile booleana di controllo
                    batch_is_valid = True

                    # Si itera su ogni elemento presente all'interno del dizionario dei sensori, che rappresenta il tensore di ogni gruppo di misurazioni previsto
                    for sensor_name, sensor_tensor in sensors_dict.items():

                        try:

                            # Si calcola il numero di misurazioni presenti in un secondo per il blocco di misurazioni di iterazione
                            sensor_frequency = sensors_frequencies[sensor_name]     # Frequenza del sensore
                            batch_start_line = second * sensor_frequency            # ad es. 0 per t0=0s, 5/10/1 per t1=1s in base al sensore...
                            batch_end_line = (second + 1) * sensor_frequency        # ad es. 5/10/1 per t0=0s, 10/20/2 per t1=1s in base al sensore...

                            # Si isola la porzione di tensore associata a quella durata di misurazioni
                            porzione_tensore = sensor_tensor[batch_start_line:batch_end_line]

                            # Controllo di validità della batch --> serve che il numero sia sempre lo stesso o altrimenti si scarta l'intero secondo
                            if porzione_tensore.shape[0] != sensor_frequency:
                                batch_is_valid = False
                                print(f'    [WARNING]: Al secondo {second} della missione {trajectory} - {combination}, il blocco di sensori {sensor_name} presenta un numero di valori diverso dalla frequenza prevista.')
                                break

                            # Si aggiunge la porzione del sensore alla lista di batch per il secondo di iterazione
                            single_second_batches_list.append(porzione_tensore)

                        except Exception as e:
                            print(f"    [WARNING] Errore nella fase di creazione del gruppo di batch unitarie per il secondo {second} --> {e}")
                            batch_is_valid = False
                            continue

                    # Si aggiunge alla lista completa delle batch la sottolista contenente i dati dei sensori per il secondo di iterazione se le dimensioni singole sono corrette
                    if batch_is_valid:
                        self.training_batches_list.append(single_second_batches_list)

                print(f'  Batch unitarie per la missione {trajectory} - {combination} definite correttamente.')

        return self

    # Funzione che permette di ricevere in input il dizionario di tensori prodotto e restituire un dizionario contenente le liste di liste contenenti le misurazioni normalizzate ricevute per singolo secondo delle sole missioni di test
    def test_unitary_batches_building(self, tensor_dict, sensors_frequencies, test_missions):

        for trajectory, mission_dict in tensor_dict.items():

            print(f"\nScomposizione del tensore per le missioni di test associate alla traiettoria {trajectory} in corso:")

            for combination, sensors_dict in mission_dict.items():

                # Si definisce la durata di missione [s] --> si prende come riferimento la colonna di GPS che ha 1 misura al secondo
                mission_time = len(sensors_dict['GPS'])-1

                # Verifica della missione --> se il numero di missione non è nella lista fornita in input si salta il resto del ciclo passando alla missione dopo
                if (trajectory, combination) not in test_missions:
                    continue

                # Si inizializza la lista associata alla missione nel dizionario delle batch
                mission = (trajectory, combination)
                if mission not in self.test_batches_dict:
                    self.test_batches_dict[mission] = []

                for second in range(mission_time):

                    # Si inizializza una lista vuota che conterrà n tensori --> un tensore per ogni blocco sensori contenente le relative misurazioni di 1 secondo
                    single_second_batches_list = []

                    # Si inizializza la variabile booleana di controllo
                    batch_is_valid = True

                    # Si itera su ogni elemento presente all'interno del dizionario dei sensori, che rappresenta il tensore di ogni gruppo di misurazioni previsto
                    for sensor_name, sensor_tensor in sensors_dict.items():

                        try:

                            # Si calcola il numero di misurazioni presenti in un secondo per il blocco di misurazioni di iterazione
                            sensor_frequency = sensors_frequencies[sensor_name]     # Frequenza del sensore
                            batch_start_line = second * sensor_frequency            # ad es. 0 per t0=0s, 5/10/1 per t1=1s in base al sensore...
                            batch_end_line = (second + 1) * sensor_frequency        # ad es. 5/10/1 per t0=0s, 10/20/2 per t1=1s in base al sensore...

                            # Si isola la porzione di tensore associata a quella durata di misurazioni
                            porzione_tensore = sensor_tensor[batch_start_line:batch_end_line]

                            # Controllo di validità della batch --> serve che il numero sia sempre lo stesso o altrimenti si scarta l'intero secondo
                            if porzione_tensore.shape[0] != sensor_frequency:
                                batch_is_valid = False
                                print(f'    [WARNING]: Al secondo {second} della missione {trajectory} - {combination}, il blocco di sensori {sensor_name} presenta un numero di valori diverso dalla frequenza prevista.')
                                break

                            # Si aggiunge la porzione del sensore alla lista di batch per il secondo di iterazione
                            single_second_batches_list.append(porzione_tensore)

                        except Exception as e:
                            print(f"    [WARNING] Errore nella fase di creazione del gruppo di batch unitarie per il secondo {second} --> {e}")
                            batch_is_valid = False
                            continue

                    # Si aggiunge alla lista completa delle batch la sottolista contenente i dati dei sensori per il secondo di iterazione se le dimensioni singole sono corrette
                    if batch_is_valid:
                        self.test_batches_dict[mission].append(single_second_batches_list)

                print(f'  Batch unitarie per la missione {trajectory} - {combination} definite correttamente.')

        return self.test_batches_dict

    # Funzione che porta a definire il numero di campioni da 1 secondo accumulati --> necessaria per usare len(dataset) e per il dataloader
    def __len__(self):

        if self.mode == "train":
            return len(self.training_batches_list)
        else:
            return len(self.test_batches_dict)

    # Funzione che gestisce la chiamata di una specifica batch unitaria del dataset --> necessaria per usare dataset[] e per il dataloader
    def __getitem__(self, idx):

        if self.mode == "train":
            return self.training_batches_list[idx]
        else:
            return self.test_batches_dict[idx]


# Si definisce la classe utilizzata per il Simplified Attention Mechanism --> trattata come classe per non appesantire troppo il codice della NavNet --> prodotto in output un vettore dim(hidden_layer)*1
class SimplifiedAttentionMechanism(nn.Module):

    # Si inizializza il layer del simplified attention mechanism
    def __init__(self, hidden_size):

        # Si inizializzano i meccanismi interni legati a pytorch
        super(SimplifiedAttentionMechanism, self).__init__()

        # Si definisce la funzione che permette di trasformare i vettori freq(sensore)*100 in un unico valore dipendente dai pesi associati alle singole misurazioni fornite in input
        self.attention = nn.Linear(hidden_size, 1)

    # Si definisce la funzione che permette di gestire la forward propagation della rete neurale
    def forward(self, LSTM_output):

        # LSTM_output nella forma [batch_size, sequential_length, feature] per come è stata definita nella classe della rete neurale

        # Si applica la funzione di attivazione tanh alla learnable function introdotta in precedenza --> si ottiene un valore compreso tra -1 e +1
        e_t = torch.tanh(self.attention(LSTM_output))

        # Si calcola il peso percentuale associato alla learnable function --> applico funzione Softmax
        alpha_t = F.softmax(e_t, dim=1)

        # Si calcola il context vector --> somma degli output LSTM (hidden layers) pesati (con il peso percentuale)
        c = torch.sum(alpha_t*LSTM_output, dim=1)

        return c

# Si definisce ora la classe all'interno della quale sono gestite tutte le operazioni della rete neurale
class NavNet(nn.Module):

    # Si inizializza la rete neurale --> permette di salvare i layer prodotti all'interno della funzione in modo permanente nella variabile del main x = NavNet()
    def __init__(self):

        # Si inizializzano i meccanismi interni legati a pytorch
        super(NavNet, self).__init__()

        ## Si definiscono le caratteristiche dei layer utilizzati dalla rete neurale, andando a salvare all'interno di variabile self tutti i parametri a essi legati ##

        # Rete ricorsiva --> due LSTM da 100 hidden states l'uno posti in serie --> prodotti per ogni unità temporale vettori da freq(input)*100
        self.LSTM_IMU = nn.LSTM(input_size=9, hidden_size=100, num_layers=2, batch_first=True)
        self.LSTM_DVL = nn.LSTM(input_size=3, hidden_size=100, num_layers=2, batch_first=True)
        #self.LSTM_REF = nn.LSTM(input_size=7, hidden_size=100, num_layers=2, batch_first=True)

        # Simplified attention mechanism --> trattato come classe separata --> riceve in input i vettori dati dall'LSTM e produce in output un vettore 100*1
        self.SAM_IMU = SimplifiedAttentionMechanism(hidden_size=self.LSTM_IMU.hidden_size)
        self.SAM_DVL = SimplifiedAttentionMechanism(hidden_size=self.LSTM_DVL.hidden_size)
        #self.SAM_REF = SimplifiedAttentionMechanism(hidden_size=self.LSTM_REF.hidden_size)

        # Fully connected layers --> in input ricevono il vettore c concatenato --> passano da dim(c) a 100 e poi 2 valori (output di spostamento su East e North)
        input_size_FC = 200
        self.FC = nn.Sequential(
            nn.Linear(input_size_FC, 100),
            nn.ReLU(),
            nn.Linear(100, 2))

        # Definizione struttura funzione di costo --> si sceglie una Euclidean Loss per restare coerenti con il paper
        self.loss_criterion = nn.MSELoss()

        # Definizione ottimizzatore per la backward propagation --> qui si usa il metodo Adams
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Si crea una funzione utile per modificare, in caso di necessità, l'ottimizzatore utilizzato --> utilizzato kwargs per poter usare metodi con numero di argomenti di input differenti
    def set_optimizer(self, new_optimizer, **kwargs):
        self.optimizer = new_optimizer(self.parameters(), **kwargs)

    # Si crea ora la funzione che gestisce la forward propagation della rete neurale
    def forward(self, dati_IMU, dati_DVL):

        # 1. Passaggio negli LSTM
        LSTM_outputs_IMU, _ = self.LSTM_IMU(dati_IMU)
        LSTM_outputs_DVL, _ = self.LSTM_DVL(dati_DVL)
        #LSTM_outputs_REF, _ = self.LSTM_REF(dati_ref)

        # 2. Passaggio al meccanismo di attenzione semplificato
        Context_vector_IMU = self.SAM_IMU(LSTM_outputs_IMU)
        Context_vector_DVL = self.SAM_DVL(LSTM_outputs_DVL)
        #Context_vector_REF = self.SAM_REF(LSTM_outputs_REF)

        # 3. Concatenazione dei vettori di contesto
        c = torch.cat((Context_vector_IMU, Context_vector_DVL), dim=1)

        # 4. Passaggio dai fully connected layers
        posizione_predetta = self.FC(c)

        return posizione_predetta

    # Si crea una funzione per l'aggiornamento dei pesi tramite backpropagation per ogni epoca di addestramento
    def backpropagation(self, dati_IMU, dati_DVL, pos_target):

        # Fase di forward propagation della rete neurale --> viene eseguita e genera i valori obiettivo
        self.optimizer.zero_grad()                              # Funzione che resetta tutti i gradienti dei tensori gestiti dall'optimizer all'inizio di ogni iterazione
        pos_predetta = self.forward(dati_IMU, dati_DVL)         # Richiamo la rete neurale sui dati (all'interno di appositi dataframe) da analizzare

        # Fase di backward optimization --> si ottimizzano i pesi introdotti
        loss = self.loss_criterion(pos_predetta, pos_target)    # Calcolo della perdita associata alla predizione rispetto al target
        loss.backward()                                         # Calcolo del gradiente della funzione di costo
        self.optimizer.step()                                   # Update dei parametri di ottimizzazione

        return loss.item()

    # Si crea una funzione per l'inizializzazione corretta dei pesi
    def initialize_weights(self):
        for m in self.modules():
            # Per i livelli lineari (FC e Attention)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Per i livelli LSTM (Opzionale ma consigliato)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)  # Ottimo per le matrici ricorrenti
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)


class PostProcessing:

    def __init__(self):

        # Dizionario per i Titoli (Coerente con i titoli dei capitoli/sezioni in blupolito)
        self.font_titolo = {
            'family': 'serif',      # Simula il font serif di Latin Modern usato all'interno del template Latex per la tesi
            'color': '#002E5F',     # Colore blupolito definito all'interno del template
            'weight': 'bold',       # Grassetto per far risaltare il titolo del grafico
            'size': 13,             # Dimensione equilibrata per il titolo del grafico
            'style': 'normal'
        }

        # Dizionario per le Etichette degli Assi (X e Y)
        self.font_assi = {
            'family': 'serif',      # Simula il font serif di Latin Modern usato all'interno del template Latex per la tesi
            'color': 'black',       # Testo nero ad alto contrasto per la massima leggibilità
            'weight': 'normal',     # Peso normale per le etichette descrittive
            'size': 11,             # Dimensione leggibile ma subordinata al titolo
            'style': 'normal'
        }

        self.waypoints_dict = {}            # Dizionario che conterrà le liste di waypoints in assi NED relativi di ogni traiettoria --> nella forma {'0': [], '1': []...}

    # Funzione che permette di
    #def single_path_definition(self):

    # Funzione che permette la costruzione di un dizionario contenente le liste di punti (in coordinate UTM) di ogni traiettoria utilizzata
    def waypoints_dict_building(self, ROOT_DIR):

        input_folder = os.path.join(ROOT_DIR,f'Telemetrie/Lista_punti')  # Si definisce la cartella di riferimento

        for trajectory, file in enumerate(sorted(os.listdir(input_folder))):

            trajectory = str(trajectory)

            # Si inizializza la lista di punti associata alla traiettoria (se non presente)
            if trajectory not in self.waypoints_dict:
                self.waypoints_dict[trajectory] = []

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

                        if idx == 0:
                            continue
                        if idx == 1:
                            East_coord_0, North_coord_0, _, _ = utm.from_latlon(latitude, longitude)
                            tuple_coord = (0, 0)
                        else:
                            East_coord, North_coord, _, _ = utm.from_latlon(float(latitude), float(longitude))

                            # Si effettua la sottrazione per arrivare alle coordinate NED relative al punto iniziale --> sono quelle necessarie per definire i waypoint
                            tuple_coord = (East_coord - East_coord_0, North_coord - North_coord_0)

                        self.waypoints_dict[trajectory].append(tuple_coord)

                    except Exception as e:
                        print(f"  [WARNING] Rilevato un errore generico nella lettura delle coordinate {idx} della traiettoria {trajectory}: {e}")
                        continue

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
    def plot_confronto_traiettoria(self, GPS_coordinates_list, NN_coordinates_list, mission, path_name, title):

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
        plt.title(title, fontdict = self.font_titolo, loc ="center", pad = 10)
        plt.xlabel('East [m]', fontdict = self.font_assi)
        plt.ylabel('North [m]', fontdict = self.font_assi)
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

        if not os.path.exists('Grafici'):
            os.makedirs('Grafici')

        plt.savefig(f'Grafici/Confronto_traiettorie_M{mission}_{path_name}.png', bbox_inches='tight')
        plt.show()
