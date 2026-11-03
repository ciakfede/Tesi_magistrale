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

    # Si inizializza la struttura comune degli elementi appartenenti alla classe --> in input riceve 3 liste di tensori e i parametri di normalizzazione (utilizzati solamente per Lp norm)
    def __init__(self, tensor_dict, sensor_frequencies, test_missions, network_config, mode, yaw_dict=None, norm_parameters=None, window_size=1, window_stride=1):

        self.mode = mode
        self.sensor_frequencies = sensor_frequencies

        self.yaw_dict = yaw_dict                    # Dizionario degli yaw [rad] per missione --> serve alla loss per ruotare gli spostamenti da body a NED (solo training e validazione)
        self.norm_parameters = norm_parameters      # Dizionario dei parametri di normalizzazione --> serve per riportare in metri le velocità DVL
        self.window_size = window_size              # Numero di secondi consecutivi aggregati in una finestra (K) --> solo training e validazione
        self.window_stride = window_stride          # Passo [s] tra l'inizio di due finestre successive della stessa missione

        self.validation_missions_dict = {('0', '2'), ('1', '8H'), ('2', '11G'), ('3', '16C'), ('4', '12H'), ('5', '5A')}

        # Si richiama la funzione opportuna in base alla modalità prescelta --> si ottiene la lista delle batch unitarie
        if mode == 'train':

            self.training_batches_list = []                                                             # Nella forma [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], [...], ...] --> struttura ripetuta per tutti i secondi di ogni missione
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions, network_config)

        elif mode == 'validazione':

            self.validation_batches_list = []                                                           # Nella forma [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], [...], ...] --> struttura ripetuta per tutti i secondi di ogni missione
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions, network_config)

        elif mode == 'test':

            self.test_batches_dict = {}                                                                 # Nella forma {'Missione1': [[tensor(5_IMU), tensor(10_DVL), tensor(1_GPS)], ...], 'Missione2': []...} --> struttura ripetuta all'interno di ogni missione per tutti i secondi della missione e per tutte le missioni
            self.unitary_batches_building(tensor_dict, sensor_frequencies, test_missions, network_config)

    # Funzione che permette di ricevere in input il dizionario dei tensori prodotto e restituire una lista di liste, con le misurazioni normalizzate ricevute per singolo secondo di missione
    def unitary_batches_building(self, tensor_dict, sensors_frequencies, test_missions, network_config):

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

                elif (trajectory, combination) in test_missions and (self.mode == 'train' or self.mode == 'validazione'):
                    continue

                elif (trajectory, combination) not in test_missions and self.mode == 'test':
                    continue

                # Se la modalità non è validazione si saltano tutte le missioni non definite a questo scopo --> non cambia a livello di risultato ma si risparmia costo computazionale
                if (trajectory, combination) in self.validation_missions_dict and self.mode != 'validazione':
                    continue

                elif (trajectory, combination) not in self.validation_missions_dict and self.mode == 'validazione':
                    continue

                # Si itera per ogni secondo della missione
                mission_seconds_list = []
                for second in range(mission_time):

                    # Si inizializza un dizionario vuoto che conterrà n tensori --> un tensore per ogni blocco sensori contenente le relative misurazioni di 1 secondo
                    single_second_batches_list = {}

                    # Si inizializza la variabile booleana di controllo
                    batch_is_valid = True

                    # Si itera su ogni elemento presente all'interno del dizionario dei sensori, che rappresenta il tensore di ogni gruppo di misurazioni previsto
                    first_ref_encounter = True
                    temp_tensor_slice = torch.empty(0)
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
                            tensor_slice = sensor_tensor[batch_start_line:batch_end_line]

                            # Controllo di validità della batch --> serve che il numero sia sempre lo stesso o altrimenti si scarta l'intero secondo
                            if tensor_slice.shape[0] != sensor_frequency:
                                batch_is_valid = False
                                print(f'    [WARNING]: Al secondo {second} della missione {trajectory} - {combination}, il blocco di sensori {sensor_name} presenta un numero di valori diverso dalla frequenza prevista.')
                                break

                            # Si aggregano le batch associate ai sensori che misurano le azioni ideali di motori e velocità --> se usate appartengono allo stesso ramo --> si aggiunge un check relativo alla configurazione della rete (se REF non è presente si salta e non si inserisce)
                            if sensor_name == 'MOT' or sensor_name == 'V_ref':

                                if network_config['IMU'] == 3 or network_config['IMU'] == 4:
                                    sensor_name = 'REF'
                                    if sensor_name not in network_config:
                                        continue
                                elif network_config['IMU'] == 9 or network_config['IMU'] == 10:
                                    sensor_name = 'IMU'
                                    temp_tensor_slice = single_second_batches_list[sensor_name]
                                    first_ref_encounter = False

                                if first_ref_encounter:
                                    temp_tensor_slice = tensor_slice
                                    first_ref_encounter = False
                                    continue
                                else:
                                    tensor_slice = torch.cat([temp_tensor_slice, tensor_slice], dim=1)

                            # Si aggiunge la porzione del sensore alla lista di batch per il secondo di iterazione
                            single_second_batches_list[sensor_name] = tensor_slice

                        except Exception as e:
                            print(f"    [WARNING] Errore nella fase di creazione del gruppo di batch unitarie per il secondo {second} --> {e}")
                            batch_is_valid = False
                            continue

                    # Per training e validazione si aggiungono lo yaw del secondo e lo spostamento DVL (in metri, assi body) --> servono ai termini phi_d2 e phi_s della loss
                    if batch_is_valid and self.mode != 'test' and self.yaw_dict is not None:

                        # Si estraggono le colonne associate ai parametri di normalizzazione e ai valori DVL normalizzati delle componenti lungo x e y
                        dvl_mean = torch.tensor(self.norm_parameters['DVL']['media'][:2])
                        dvl_std = torch.tensor(self.norm_parameters['DVL']['deviazione standard'][:2])
                        dvl_velocities = single_second_batches_list['DVL'][:, :2] * dvl_std + dvl_mean

                        # Si aggiungono al batch del singolo secondo la componente di spostamento calcolato tramite integrazione delle velocità e l'angolo di yaw corrispondente alla misura GPS target (si usa un offset di 1 s)
                        single_second_batches_list['DVL_DISP'] = dvl_velocities.sum(dim=0) / sensors_frequencies['DVL']
                        single_second_batches_list['YAW'] = torch.tensor(self.yaw_dict[trajectory][combination][second + 1], dtype=torch.float32)

                    # Si aggiunge alla lista completa delle batch la sottolista contenente i dati dei sensori per il secondo di iterazione se le dimensioni singole sono corrette
                    if batch_is_valid and (trajectory, combination) not in self.validation_missions_dict and self.mode == 'train':
                        mission_seconds_list.append((second, single_second_batches_list))
                    elif batch_is_valid and (trajectory, combination) in self.validation_missions_dict and self.mode == 'validazione':
                        mission_seconds_list.append((second, single_second_batches_list))
                    elif batch_is_valid and (trajectory, combination) in test_missions and self.mode == 'test':
                        self.test_batches_dict[(trajectory, combination)].append(single_second_batches_list)
                    else:
                        continue

                # Per training e validazione i secondi validi della missione vengono raggruppati in finestre di secondi consecutivi
                if self.mode != 'test':
                    self.windows_building(mission_seconds_list)

                print(f'  Batch unitarie per la missione {trajectory} - {combination} definite correttamente.')

        return self

    # Funzione che raggruppa i secondi di una missione in finestre di window_size secondi consecutivi (passo window_stride) --> ogni finestra ha forma [K, ...] per ogni blocco di sensori, YAW [K] e DVL_DISP [K, 2]
    def windows_building_overlap(self, mission_seconds_list):

        # Si itera per tutti i secondi che rappresentano l'istante di inizio di una nuova finestra
        for start in range(0, len(mission_seconds_list) - self.window_size + 1, self.window_stride):

            window = mission_seconds_list[start:start + self.window_size]

            # Se un secondo è stato scartato nel mezzo della finestra questa non è consecutiva e viene saltata --> altrimenti cumsum e smoothness della loss non avrebbero significato
            if window[-1][0] - window[0][0] != self.window_size - 1:
                continue

            stacked_window = {key: torch.stack([single_second[key] for _, single_second in window]) for key in window[0][1]}

            if self.mode == 'train':
                self.training_batches_list.append(stacked_window)
            elif self.mode == 'validazione':
                self.validation_batches_list.append(stacked_window)

    # Funzione che raggruppa i secondi di una missione in finestre di window_size secondi consecutivi --> ogni finestra ha forma [K, ...] per ogni blocco di sensori, YAW [K] e DVL_DISP [K, 2]
    def windows_building(self, mission_seconds_list):

        # Si itera per tutti i secondi che rappresentano l'istante di inizio di una nuova finestra
        for start in range(0, len(mission_seconds_list) - self.window_size + 1, self.window_size):

            window = mission_seconds_list[start:start + self.window_size]

            # Se un secondo è stato scartato nel mezzo della finestra questa non è consecutiva e viene saltata --> altrimenti cumsum e smoothness della loss non avrebbero significato
            if window[-1][0] - window[0][0] != self.window_size - 1:
                continue

            stacked_window = {key: torch.stack([single_second[key] for _, single_second in window]) for key in window[0][1]}

            if self.mode == 'train':
                self.training_batches_list.append(stacked_window)
            elif self.mode == 'validazione':
                self.validation_batches_list.append(stacked_window)

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


# Classe che implementa la loss function ispirata a Wei & Sun (2022) --> misfit dati + termini di regolarizzazione Lp (smallness rispetto al DVL e smoothness temporale) su finestre di K secondi consecutivi
class MixedLpNavLoss(nn.Module):

    # Si inizializzano media e std del GPS (per riportare in metri output e target) e i pesi dei termini --> i pesi sono iperparametri da tarare sulla validazione
    def __init__(self, mu, sigma, w_cum=0.1, a_s=0.05, a_t=0.05, p=1.0, q=1.0, w_dir=0.0, eps=1e-3):

        super().__init__()

        self.register_buffer('mu', mu, persistent=False)            # persistent=False --> non compaiono nello state_dict, quindi i pesi già salvati restano compatibili
        self.register_buffer('sigma', sigma, persistent=False)
        self.w_cum, self.a_s, self.a_t, self.p, self.q, self.w_dir, self.eps = w_cum, a_s, a_t, p, q, w_dir, eps
        self.terms = {}                                             # Valori degli ultimi termini calcolati --> utili per tarare i pesi

    # Utilizzo della norma p elevata al grado p --> come specificato nel lavoro di Fournier-Oldenburg (versione smussata con eps per avere gradiente definito in 0)
    def lp(self, x, p):
        return ((x**2 + self.eps**2) ** (p / 2)).mean()

    # Funzione interna che ruota uno spostamento dagli assi body (X, Y) agli assi NED (E, N) --> stessa convenzione di body_to_NED in PreProcessing, ma lavora su tensori [window size, window stride,2] e su yaw [window size, window stride]
    def body_to_NED(self, displacement, yaw):

        # Si estraggono i valori di spostamento body --> usata forma che permette di tenere conto di qualsiasi forma precedente all'indice colonna
        X_body = displacement[..., 0]
        Y_body = displacement[..., 1]

        # Si effettua la conversione
        Delta_N = X_body * torch.cos(yaw) - Y_body * torch.sin(yaw)
        Delta_E = X_body * torch.sin(yaw) + Y_body * torch.cos(yaw)

        return torch.stack([Delta_E, Delta_N], dim=-1)

    # Funzione di esecuzione del calcolo della loss effettivo --> y_hat, y: [B,K,2] normalizzati (body: X, Y); yaw: [B,K] rad; v_dvl: [B,K,2] spostamento DVL in metri (body)
    def forward(self, y_hat, y, yaw, v_dvl):

        # Conversione di output target e prodotto dalla rete in metri (de-normalizzazione)
        m_hat, m = y_hat * self.sigma + self.mu, y * self.sigma + self.mu

        sigma2 = (self.sigma**2).mean()       # Fattore che rende adimensionali i termini espressi in metri --> è la varianza

        phi_d1 = F.mse_loss(y_hat, y)                                                              # Misfit per passo (attuale)
        phi_d2 = F.mse_loss(self.body_to_NED(m_hat, yaw).cumsum(1), self.body_to_NED(m, yaw).cumsum(1)) / sigma2
        phi_s   = self.lp((m_hat - v_dvl) / self.sigma, self.p)                                     # Smallness rispetto allo spostamento DVL
        phi_t   = self.lp(y_hat[:, 1:] - y_hat[:, :-1], self.q)                                     # Smoothness temporale
        cross   = (m_hat[..., 0]*v_dvl[..., 1] - m_hat[..., 1]*v_dvl[..., 0]).pow(2).mean() / sigma2**2     # Allineamento di direzione con il DVL (prodotto vettoriale)

        self.terms = {'phi_d': phi_d1.detach(), 'phi_cum': phi_d2.detach(), 'phi_s': phi_s.detach(), 'phi_t': phi_t.detach(), 'cross': cross.detach()}
        #self.terms = {'phi_d': phi_d1.detach(), 'phi_s': phi_s.detach(),'phi_t': phi_t.detach(), 'cross': cross.detach()}

        return phi_d1 + self.w_cum*phi_d2 + self.a_s*phi_s + self.a_t*phi_t + self.w_dir*cross


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
    def __init__(self, network_config, hidden_size, FC_layers):

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

        # Si definisce una lista di dimensioni del layer --> composta da elementi input_size, input_size//2, input_size//4,...,2 (in base a quanti layer sono previsti)
        input_size_FC = hidden_size * (len(network_config) - 1)
        layers_size = [input_size_FC] + [input_size_FC // (2*i) for i in range(1, FC_layers)] + [2]

        # Si definisce una lista di layer composti dal modulo lineare seguito da una funzione ReLu --> si rimuove l'ultimo elemento che sarebbe la ReLu di troppo una volta raggiunta la dimensionalità finale della rete
        layers = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.pop()

        # Si associa la struttura definita alla variabile FC
        self.FC = nn.Sequential(*layers)

        # Definizione struttura funzione di costo --> si sceglie una Euclidean Loss per restare coerenti con il paper
        self.loss_criterion = nn.MSELoss()

        # Definizione ottimizzatore per la backward propagation --> qui si usa il metodo Adams
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Si crea una funzione utile per modificare, in caso di necessità, l'ottimizzatore utilizzato --> utilizzato kwargs per poter usare metodi con numero di argomenti di input differenti
    def set_optimizer(self, new_optimizer, **kwargs):
        self.optimizer = new_optimizer(self.parameters(), **kwargs)

    # Si crea una funzione per sostituire la loss MSE con la loss mista --> richiede media e std del GPS (per riportare in metri) e i pesi dei termini tramite kwargs
    def set_loss_function(self, loss_function, GPS_mean, GPS_std, **kwargs):

        if loss_function == MixedLpNavLoss:
            self.loss_criterion = MixedLpNavLoss(GPS_mean, GPS_std, **kwargs)
        else:
            self.loss_criterion = loss_function()

    # Si crea ora la funzione che gestisce la forward propagation della rete neurale
    def forward(self, data):

        # Si itera su ogni elemento presente nel dizionario dei dati per ramo
        context_vector_list = []
        for key, batch in data.items():

            if key == 'GPS' or key == 'YAW' or key == 'DVL_DISP':
                continue

            # Si memorizzano le dimensioni di raggruppamento e si uniscono in un'unica dimensione --> ogni secondo diventa una sequenza indipendente [N, freq, feat] per l'LSTM (qualsiasi sia il numero di dimensioni precedenti)
            lead_shape = batch.shape[:-2]
            batch = batch.reshape(-1, *batch.shape[-2:])

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

        return predicted_displacement.reshape(*lead_shape, -1)
        #return predicted_displacement

    # Si crea una funzione per l'aggiornamento dei pesi tramite backpropagation per ogni epoca di addestramento
    def backpropagation(self, data):

        # Fase di forward propagation della rete neurale --> viene eseguita e genera i valori obiettivo
        self.optimizer.zero_grad()                # Funzione che resetta tutti i gradienti dei tensori gestiti dall'optimizer all'inizio di ogni iterazione
        pos_predetta = self.forward(data)         # Richiamo la rete neurale sui dati (all'interno di appositi dataframe) da analizzare

        # Fase di backward optimization --> si ottimizzano i pesi introdotti
        pos_target = data['GPS']

        # Calcolo della perdita associata alla predizione rispetto al target
        if isinstance(self.loss_criterion, MixedLpNavLoss):
            loss = self.loss_criterion(pos_predetta, pos_target, data['YAW'], data['DVL_DISP'])
        else:
            loss = self.loss_criterion(pos_predetta, pos_target)

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
