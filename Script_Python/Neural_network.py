"""     IMPLEMENTAZIONE RETE NEURALE NAVNET
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             """

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
import warnings                                 # Libreria per la gestione dei warnings come exception
from pandas.errors import DtypeWarning          # Libreria per la gestione dei warnings di tipo misto nella lettura di un dataframe pandas

pd.set_option('display.max_columns', None)              # Visualizza tutte le colonne (nessun limite al numero)
pd.set_option('display.max_colwidth', None)             # Visualizza tutto il contenuto della cella (evita di troncare stringhe lunghe)
pd.set_option('display.expand_frame_repr', False)       # Evita che le colonne vadano a capo su più righe nel terminale
pd.options.display.max_rows = 100                             # Imposta il numero massimo di righe visualizzabili a schermo

warnings.filterwarnings('error', category=pd.errors.DtypeWarning)           # Forza i DtypeWarning a comportarsi come eccezioni

torch.manual_seed(0)            # Si imposta il valore dei seed di partenza per l'addestramento della rete


# Si definisce una nuova classe che prende in input i tensori suddivisi per blocco e missione (generati da PreProcessing) e li trasforma in singole batch di dati (1 secondo) da fornire alla rete
class Dataset(torch.utils.data.Dataset):

    # Si inizializza la struttura comune degli elementi appartenenti alla classe --> in input riceve 3 liste di tensori
    def __init__(self, tensor_dict, sensor_frequencies, test_missions, mode):

        self.mode = mode

        self.validation_missions_dict = {('0', '2'), ('1', '8H'), ('2', '11G'), ('3', '16C'), ('4', '12H'), ('5', '5A')}

        # Si richiama la funzione opportuna in base alla modalità prescelta --> si ottiene la lista delle batch unitarie
        if mode == 'train':

            self.training_batches_list = []                                                             # Nella forma [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], [...], ...] --> struttura ripetuta per tutti i secondi di ogni missione
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions)

        elif mode == 'validazione':

            self.validation_batches_list = []                                                           # Nella forma [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], [...], ...] --> struttura ripetuta per tutti i secondi di ogni missione
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions)

        elif mode == 'test':

            self.test_batches_dict = {}                                                                 # Nella forma {'Missione1': [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], ...], 'Missione2': []...} --> struttura ripetuta all'interno di ogni missione per tutti i secondi della missione e per tutte le missioni
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions)

    # Funzione che permette di ricevere in input il dizionario dei tensori prodotto e restituire una lista di liste, con le misurazioni normalizzate ricevute per singolo secondo di missione
    def unitary_batches_building(self, tensor_dict, sensors_frequencies, test_missions):

        for trajectory, mission_dict in tensor_dict.items():

            print(f"\n  Scomposizione del tensore per {self.mode} per le missioni associate alla traiettoria {trajectory} in corso:")

            for combination, sensors_dict in mission_dict.items():

                # Se c'è stato un errore nella fase di creazione del tensore si salta la missione anche qui
                if not sensors_dict:
                    continue

                # Si definisce la durata di missione [s] --> si prende come riferimento la colonna di GPS che ha 1 misura al secondo
                mission_time = len(sensors_dict['GPS']) - 1

                # Verifica della missione --> se il numero di missione è nella lista fornita in input si salta il resto del ciclo passando alla missione dopo
                if (trajectory, combination) in test_missions and self.mode == 'test':
                    if (trajectory, combination) not in self.test_batches_dict:
                        self.test_batches_dict[(trajectory, combination)] = []
                elif (trajectory, combination) in test_missions and (
                        self.mode == 'train' or self.mode == 'validazione'):
                    continue

                for second in range(mission_time):

                    # Si inizializza un dizionario vuoto che conterrà n tensori --> un tensore per ogni blocco sensori contenente le relative misurazioni di 1 secondo
                    single_second_batches_list = {}

                    # Si inizializza la variabile booleana di controllo
                    batch_is_valid = True

                    # Si itera su ogni elemento presente all'interno del dizionario dei sensori, che rappresenta il tensore di ogni gruppo di misurazioni previsto
                    for sensor_name, sensor_tensor in sensors_dict.items():

                        try:

                            if sensor_name == 'Depth' or sensor_name == 'DepthVel':
                                continue

                            # Si calcola il numero di misurazioni presenti in un secondo per il blocco di misurazioni di iterazione
                            sensor_frequency = sensors_frequencies[sensor_name]             # Frequenza del sensore
                            offset = 1 if sensor_name == 'GPS' else 0                       # Si definisce un offset di 1 secondo per il GPS --> in questo modo il delta viene stimato in base alle misurazioni del blocco dati precedente (il delta è la prima posizione del blocco per come è generato il dataset quindi userebbe dati non ancora esistenti al momento del calcolo del delta)
                            batch_start_line = (second + offset) * sensor_frequency         # Ad es. 0 per t0=0s, 5/10/1 per t1=1s in base al sensore...second + 1 + offset) * sensor_frequency
                            batch_end_line = (second + 1 + offset) * sensor_frequency       # Ad es. 5/10/1 per t0=0s, 10/20/2 per t1=1s in base al sensore

                            # Si isola la porzione di tensore associata a quella durata di misurazioni
                            porzione_tensore = sensor_tensor[batch_start_line:batch_end_line]

                            # Controllo di validità della batch --> serve che il numero sia sempre lo stesso o altrimenti si scarta l'intero secondo
                            if porzione_tensore.shape[0] != sensor_frequency:
                                batch_is_valid = False
                                print(f'    [WARNING]: Al secondo {second} della missione {trajectory} - {combination}, il blocco di sensori {sensor_name} presenta un numero di valori diverso dalla frequenza prevista.')
                                break

                            # Si aggiunge la porzione del sensore alla lista di batch per il secondo di iterazione
                            single_second_batches_list[sensor_name] = porzione_tensore

                        except Exception as e:
                            print(f"    [WARNING] Errore nella fase di creazione del gruppo di batch unitarie per il secondo {second} --> {e}")
                            batch_is_valid = False
                            continue

                    # Si aggiunge alla lista completa delle batch la sottolista contenente i dati dei sensori per il secondo di iterazione se le dimensioni singole sono corrette
                    if batch_is_valid and (trajectory, combination) not in self.validation_missions_dict and self.mode == 'train':
                        self.training_batches_list.append(single_second_batches_list)
                    elif batch_is_valid and (trajectory, combination) in self.validation_missions_dict and self.mode == 'validazione':
                        self.validation_batches_list.append(single_second_batches_list)
                    elif batch_is_valid and (trajectory, combination) in test_missions and self.mode == 'test':
                        self.test_batches_dict[(trajectory, combination)].append(single_second_batches_list)
                    else:
                        continue

                print(f'  Batch unitarie per la missione {trajectory} - {combination} definite correttamente.')

        return self

    # Funzione che porta a definire il numero di campioni da 1 secondo accumulati --> necessaria per usare len(dataset) e per il dataloader
    def __len__(self):

        if self.mode == "train":
            return len(self.training_batches_list)
        elif self.mode == "validazione":
            return len(self.validation_batches_list)
        elif self.mode == 'test':
            return len(self.test_batches_dict)
        else:
            return 1

    # Funzione che gestisce la chiamata di una specifica batch unitaria del dataset --> necessaria per usare dataset[] e per il dataloader
    def __getitem__(self, index):

        if self.mode == "train":
            return self.training_batches_list[index]
        elif self.mode == "validazione":
            return self.validation_batches_list[index]
        elif self.mode == "test":
            return self.test_batches_dict[index]
        else:
            return 1


# Classe che permette di implementare la loss function Log-Cosh
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return torch.mean(diff + nn.functional.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0)))


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
    def __init__(self, network_config, hidden_size):

        # Si inizializzano i meccanismi interni legati a pytorch
        super(NavNet, self).__init__()

        # Si inizializzano i blocchi della rete neurale
        self.LSTMS = nn.ModuleDict()
        self.SAMS = nn.ModuleDict()

        # Si itera su ogni ramo da calcolare, applicando LSTM e attention layer
        for branch_name, branch_size in network_config.items():

            if branch_name == 'GPS':
                continue

            self.LSTMS[branch_name] = nn.LSTM(input_size=branch_size, hidden_size=hidden_size, batch_first=True)
            self.SAMS[branch_name] = SimplifiedAttentionMechanism(hidden_size=hidden_size)

        # Si applicano i Fully connected layers --> ricevono in input il vettore c concatenato --> devo arrivare a 2 output, che rappresentano gli spostamenti nei due assi (possibile automatizzare aggiungendo più strati)
        input_size_FC = hidden_size * (len(network_config)-1)
        self.FC = nn.Sequential(
            nn.Linear(input_size_FC, input_size_FC//2),
            nn.ReLU(),
            nn.Linear(input_size_FC//2, 2))

        # Definizione struttura funzione di costo --> si sceglie una Euclidean Loss per restare coerenti con il paper
        self.loss_criterion = LogCoshLoss()

        # Definizione ottimizzatore per la backward propagation --> qui si usa il metodo Adams
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Si crea una funzione utile per modificare, in caso di necessità, l'ottimizzatore utilizzato --> utilizzato kwargs per poter usare metodi con numero di argomenti di input differenti
    def set_optimizer(self, new_optimizer, **kwargs):
        self.optimizer = new_optimizer(self.parameters(), **kwargs)

    # Si crea ora la funzione che gestisce la forward propagation della rete neurale
    def forward(self, data):

        # Si itera su ogni elemento presente nel dizionario dei dati per ramo
        context_vector_list = []
        for key, batch in data.items():

            if key == 'GPS':
                continue

            # 1. Passaggio negli LSTM
            LSTM_output, _ = self.LSTMS[key](batch)

            # 2. Passaggio attraverso il meccanismo di attenzione semplificato
            context_vector = self.SAMS[key](LSTM_output)

            # Salvataggio del context vector prodotto nella lista
            context_vector_list.append(context_vector)

        # 3. Concatenazione dei vettori prodotti
        c = torch.cat(context_vector_list, dim=1)

        # 4. Passaggio dai fully connected layers
        predicted_displacement = self.FC(c)

        return predicted_displacement

    # Si crea una funzione per l'aggiornamento dei pesi tramite backpropagation per ogni epoca di addestramento
    def backpropagation(self, data):

        # Fase di forward propagation della rete neurale --> viene eseguita e genera i valori obiettivo
        self.optimizer.zero_grad()                # Funzione che resetta tutti i gradienti dei tensori gestiti dall'optimizer all'inizio di ogni iterazione
        pos_predetta = self.forward(data)         # Richiamo la rete neurale sui dati (all'interno di appositi dataframe) da analizzare

        # Fase di backward optimization --> si ottimizzano i pesi introdotti
        pos_target = data['GPS']
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
