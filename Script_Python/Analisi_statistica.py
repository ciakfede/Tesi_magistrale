"""           ANALISI STATISTICA
              FEDERICO CECCHINI
            ANNO ACCADEMICO 2025/26             """

import pandas as pd
import seaborn as sns                   # Modulo per la generazione di una heatmap
import matplotlib.pyplot as plt         # Modulo per gestire le impostazioni di una figura come dimensioni e salvataggio
import os                               # Modulo necessario per creare cartelle direttamente da Python
import numpy as np                      # Modulo per le funzioni matematiche
import re

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler


class StatisticalAnalysis:

    def __init__(self):

        self.correlations_dict = {}              # Dizionario nella forma {'resampling': {'pearson': {'chiave1': [], 'chiave2': []...}, 'spearman': {...}}, 'interpolazione': {...}]

        self.mean_correlations_dict = {}         # Dizionario nella forma {'resampling': {'pearson': {'chiave1': [], 'chiave2': []...}, 'spearman': {...}}, 'interpolazione': {...}]

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

        self.warn_count = 0

        self.files_initialized = set()

    # ======================================================
    #         ANALISI DI CORRELAZIONE PER MISSIONE
    # ======================================================

    # Funzione interna per il setup e il salvataggio delle heatmap raffiguranti le matrici di correlazione
    def heatmap_definition(self, correlation_method, correlation_matrix, processing_method, trajectory, combination, ROOT_DIR):

        try:
            # Impostazioni per la formattazione delle heatmap
            plt.figure(figsize=(26, 12))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f"Matrice di Correlazione di {correlation_method} per missione {trajectory} - {combination}", fontdict=self.font_titolo, loc="center", pad=10)

            # Definizione nome cartella in cui salvare i grafici
            output_dir_path = os.path.join(ROOT_DIR, f"Risultati_analisi_statistica/Analisi_{processing_method}/Analisi_{correlation_method}/Traiettoria_{trajectory}")

            # Verifica che la cartella non sia già stata creata in precedenza, altrimenti la genera
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            # Definizione del percorso del file aggiungendo alla cartella il nome dell'immagine
            output_file_path = os.path.join(output_dir_path, f"Heatmap_missione_{trajectory}_{combination}_{correlation_method}.pdf")

            # Salvataggio del file
            plt.savefig(output_file_path, bbox_inches='tight', dpi=300)

            # Chiusura della figura necessaria per liberare RAM
            plt.close()

        except Exception as e:
            print(f"    [WARNING] Errore nella generazione della heatmap associata alla missione {trajectory} - {combination} --> {e}")
            self.warn_count += 1
            return

    # Funzione interna per la formattazione dei coefficienti generati dall'analisi e la creazione di un file di testo riassuntivo
    def correlation_matrix_to_txt(self, correlation_method, couples_list, processing_method, trajectory, combination, ROOT_DIR):

        # Si definisce il percorso della cartella di output
        output_dir_path = os.path.join(ROOT_DIR, f"Risultati_analisi_statistica/Analisi_{processing_method}/Analisi_{correlation_method}/Traiettoria_{trajectory}")

        # Si verifica che la cartella non sia già stata creata in precedenza
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        # Si definisce il percorso del file di testo complessivo di tutte le missioni per ogni traiettoria
        output_file_path = os.path.join(output_dir_path, f"Report_analisi_{correlation_method}.txt")

        try:

            # Controllo esistenza file di output --> se è già presente nel set usa la modalità append (sono già alla seconda combinazione di una traiettoria), altrimenti usa write, in modo da eliminare le informazioni vecchie
            mode = "a" if output_file_path in self.files_initialized else "w"
            self.files_initialized.add(output_file_path)

            # Apertura file di testo in modalità append --> si vogliono aggiungere al file i nuovi dati di missione ogni volta
            with open(output_file_path, mode, encoding="utf-8") as f:

                # Intestazione del file --> da fare solo alla prima apertura
                if f.tell() == 0:
                    f.write("=" * 80 + "\n")
                    f.write(f"REPORT DI ANALISI TELEMETRIA X300 CON {processing_method} - METODO: {correlation_method.upper()}\n")
                    f.write(f"Coppie con correlazione superiore a 0.33 (o inferiore a -0.33):\n")
                    f.write("=" * 80 + "\n\n")

                # Intestazione per la missione di iterazione
                f.write(f"MISSIONE {trajectory} - {combination}:\n\n")

                # Si scrivono tutte le coppie con il relativo coefficiente di correlazione per la singola missione
                for (s1, s2), value in couples_list.items():

                    s1, s2 = sorted((s1, s2))
                    f.write(f"- {s1} <--> {s2}: {value:.3f}\n")

                # Separatore tra missioni
                f.write("\n" + "=" * 60 + "\n\n")

            return self

        except Exception as e:
            print(f"    [WARNING] Errore nella fase di scrittura file di testo per la missione {trajectory} - {combination}--> {e}")
            self.warn_count += 1
            return self

    # Funzione interna per l'aggiornamento del dizionario
    def correlation_dict_update(self, correlation_method, couples_list, processing_method, trajectory, combination, optimum_lag):

        # Creazione della chiave del metodo di formattazione dati di iterazione
        if processing_method not in self.correlations_dict:
            self.correlations_dict[processing_method] = {}

        # Creazione della chiave del metodo di analisi di correlazione
        if correlation_method not in self.correlations_dict[processing_method]:
            self.correlations_dict[processing_method][correlation_method] = {}

        # Si itera su ogni elemento della lista di correlazioni individuate
        for (s1, s2), value in couples_list.items():

            s1, s2 = sorted((s1, s2))
            couple_key = f"{s1} <--> {s2}"

            try:

                # Creazione della chiave associata alla coppia di valori
                if couple_key not in self.correlations_dict[processing_method][correlation_method]:
                    self.correlations_dict[processing_method][correlation_method][couple_key] = []

                # Necessario trattare separatamente questo caso perché è presente un'informazione aggiuntiva, ovvero il lag ottimale
                if correlation_method == 'cross-correlazione':

                    self.correlations_dict[processing_method][correlation_method][couple_key].append(((trajectory, combination), optimum_lag, value))

                else:

                    self.correlations_dict[processing_method][correlation_method][couple_key].append(((trajectory, combination), value))

            except Exception as e:
                print(f"    [WARNING] Errore nella fase di aggiornamento del dizionario di correlazione associato a {processing_method} - {correlation_method} - {couple_key} nella missione {trajectory} - {combination} --> {e}")
                self.warn_count += 1

        return self

    # Funzione incaricata di gestire l'analisi statistica vera e propria --> riceve in input i dizioni contenenti i dataframe processati per ogni missione e una lista dei metodi di correlazione prescelti
    def correlation_analysis(self, correlation_methods, processed_dataframes_dict, processing_methods, ROOT_DIR):

        # Si trasformano gli input in liste con un solo elemento in caso siano input di semplici stringhe --> possibile per come è strutturato il main, con più possibilità di scelta utente
        if isinstance(correlation_methods, str):
            correlation_methods = [correlation_methods]
        if isinstance(processing_methods, str):
            processing_methods = [processing_methods]

        # Si itera su ogni metodo di processing telemetrie selezionato
        for processing_method in processing_methods:

            dataset_dict = processed_dataframes_dict[processing_method].drop(columns=['UTM_North [m]', 'UTM_East [m]'])

            # Si itera sui metodi di analisi prescelti
            for correlation_method in correlation_methods:

                # Si itera su ogni elemento presente all'interno dei dizionari di partenza
                for trajectory, missions_dict in dataset_dict.items():

                    print(f'\n  Esecuzione analisi correlazione di {correlation_method.upper()}, sul database con {processing_method.upper()}, per le missioni associate alla traiettoria {trajectory} in corso:')

                    self.warn_count = 0
                    for combination, mission_dataframe in missions_dict.items():

                        # -----------------------------------------------------
                        #         CALCOLO COEFFICIENTI DI CORRELAZIONE
                        # -----------------------------------------------------
                        try:
                            # Calcolo dei coefficienti di correlazione del metodo di iterazione --> tramite fillna sostituisco tutti i valori NaN (relazioni non trovate) con uno 0
                            mission_dataframe_copy = mission_dataframe.drop(columns=['timestamp'])
                            correlation_matrix = mission_dataframe_copy.corr(method=correlation_method)
                        except Exception as e:
                            print(f"    [WARNING] Errore nel calcolo dei coefficienti di correlazione per il metodo di {correlation_method} nella missione {trajectory} - {combination} --> {e}")
                            self.warn_count += 1
                            continue

                        # -----------------------------------------------------
                        #   RAPPRESENTAZIONE GRAFICA MATRICE DI CORRELAZIONE
                        # -----------------------------------------------------
                        self.heatmap_definition(correlation_method, correlation_matrix, processing_method, trajectory, combination, ROOT_DIR)

                        # -----------------------------------------------------
                        #         SCOMPOSIZIONE MATRICE DI CORRELAZIONE
                        # -----------------------------------------------------
                        try:
                            # si trasforma la matrice in una lista di coppie --> si isola prima la matrice triangolare superiore (valori sono specchiati in quella inferiore) --> successivamente si trasforma la matrice in una lista di coppie "sensore - valore"
                            condition = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)  # Matrice di dimensioni pari alla matrice di correlazione, con valori True in ogni cella che vale 1 (triangolare superiore) e False nel resto
                            corr_matrix_triu = correlation_matrix.where(condition)
                            couples_list = corr_matrix_triu.unstack().dropna()
                        except Exception as e:
                            print(f"    [WARNING] Errore nella conversione da matrice a lista dei coefficienti di correlazione per la missione {trajectory} - {combination} --> {e}")
                            self.warn_count += 1
                            continue

                        # -----------------------------------------------------
                        #      AGGIORNAMENTO DIZIONARIO E FILE TESTUALE
                        # -----------------------------------------------------

                        # Si richiama la funzione che permette di elaborare la matrice di correlazione e aggiornare il dizionario con lo storico di tutte le coppie
                        self.correlation_matrix_to_txt(correlation_method, couples_list, processing_method, trajectory, combination, ROOT_DIR)

                        # Si richiama la funzione che aggiorna il dizionario in cui sono contenute tutte le coppie correlate individuate per ogni metodo
                        self.correlation_dict_update(correlation_method, couples_list, processing_method, trajectory, combination, 0)

                    print(f"    Analisi di correlazione eseguita con {self.warn_count} warning per la traiettoria {trajectory}.")

        return self

    # ======================================================
    #        CALCOLO CORRELAZIONE MEDIATA SU MISSIONI
    # ======================================================

    # Si inizializza una funzione per il calcolo del valore medio
    def mean_coefficients_calculation(self, correlation_method, correlations_dict, processing_method):

        # Necessario caso separato per poter gestire diversamente il contenuto dello storico
        if correlation_method == 'cross-correlazione':

            # Si itera per ogni coppia che compare nel dizionario per il metodo di iterazione
            mean_values_dict = {}
            for couple_key, couple_data in correlations_dict[processing_method][correlation_method].items():

                # Si estrapolano i singoli valori del coefficiente dalla tupla definita per il dizionario
                coefficients = [t for m, v, t in couple_data]
                optimum_lag = [v for m, v, t in couple_data]

                if couple_key not in mean_values_dict:
                    mean_values_dict[couple_key] = {}

                # Si verifica che siano presenti dei valori --> in caso non siano registrati coefficienti per una determinata coppia si imposta tutto a 0
                if len(coefficients) > 0:

                    # Si aggiorna il dizionario con i valori medi di lag e coefficiente
                    mean_values_dict[couple_key]['lag_ottimali_medi'] = np.mean(optimum_lag)
                    mean_values_dict[couple_key]['coefficienti_medi'] = np.mean(coefficients)

                else:

                    mean_values_dict[couple_key]['lag_ottimali_medi'] = 0
                    mean_values_dict[couple_key]['coefficienti_medi'] = 0
                    print(f'  [WARNING]: La coppia {couple_key} non presenta coefficienti di correlazione calcolati per il metodo {correlation_method}')

        else:

            # Si itera per ogni coppia che compare nel dizionario per il metodo di iterazione
            mean_values_dict = {}
            for couple_key, couple_data in correlations_dict[processing_method][correlation_method].items():

                # Si estrapolano i singoli valori del coefficiente dalla tupla definita per il dizionario
                coefficients = [v for m, v in couple_data]

                # Si verifica che siano presenti dei valori --> in caso non siano registrati coefficienti per una determinata coppia si imposta il valore medio a 0
                if len(coefficients) > 0:

                    mean_values_dict[couple_key] = np.mean(coefficients)

                else:

                    mean_values_dict[couple_key] = 0
                    print(f'    [WARNING]: La coppia {couple_key} non presenta coefficienti di correlazione calcolati per il metodo {correlation_method}')
                    self.warn_count += 1

        return mean_values_dict

    # Si inizializza una funzione per il setup e il salvataggio delle heatmap dei coefficienti medi
    def mean_heatmap_definition(self, correlation_method, mean_values_series, processing_method, ROOT_DIR):

        try:

            # -----------------------------------------------------
            #          CREAZIONE MATRICE DI VALORI MEDI
            # -----------------------------------------------------

            # Si estraggono i nomi univoci di tutte le misurazioni contenute nella serie di dati validi
            measures_names = set()
            for couple_key in mean_values_series.index:
                s1, s2 = couple_key.split(" <--> ")
                measures_names.update([s1, s2])

            # Si genera un dataframe vuoto avente come indici e colonne i nomi di tutte le misurazioni associate a correlazioni sopra la soglia di sbarramento
            measures_names_array = sorted(list(measures_names))
            n = len(measures_names_array)
            mean_correlation_natrix = pd.DataFrame(np.zeros((n, n)), index=measures_names_array, columns=measures_names_array)

            # Si riempe la diagonale principale del dataframe con soli valori unitari
            for i in range(n):
                mean_correlation_natrix.iat[i, i] = 1.0

            # Si riempono le celle della matrice con i coefficienti di correlazione medi calcolati
            for couple_key, couple_value in mean_values_series.items():
                s1, s2 = couple_key.split(" <--> ")
                mean_correlation_natrix.loc[s1, s2] = couple_value
                mean_correlation_natrix.loc[s2, s1] = couple_value

            # -----------------------------------------------------
            #          GENERAZIONE E SALVATAGGIO GRAFICO
            # -----------------------------------------------------

            plt.figure(figsize=(16, 10))
            sns.heatmap(mean_correlation_natrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title(f"Heatmap Medie Significative ({correlation_method})\nMetodo: {processing_method}", fontdict=self.font_titolo, loc='center', pad=10)

            # Definizione nome cartella in cui salvare i grafici
            output_dir_path = os.path.join(ROOT_DIR, f"Risultati_analisi_statistica/Analisi_{processing_method}/Analisi_{correlation_method}/Analisi_{correlation_method}_media")

            # Verifica che la cartella non sia già stata creata in precedenza, altrimenti la genera
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            # Definizione del percorso del file aggiungendo alla cartella il nome dell'immagine
            output_file_path = os.path.join(output_dir_path, f"Heatmap_medie_{correlation_method}.pdf")

            plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"    [WARNING] Errore nella fase di generazione della heatmap associata al dataset generato per {processing_method} e analizzato con {correlation_method} --> {e}")
            self.warn_count += 1

        return self

    # Funzione interna per l'aggiornamento del dizionario
    def mean_correlation_dict_update(self, correlation_method, mean_values_series, processing_method):

        try:

            # Si ritrasforma la lista in un dizionario per poter
            mean_values_dict = mean_values_series.to_dict()

            # Creazione della chiave del metodo di formattazione dati di iterazione
            if processing_method not in self.mean_correlations_dict:
                self.mean_correlations_dict[processing_method] = {}

            # Creazione della chiave del metodo di iterazione
            if correlation_method not in self.mean_correlations_dict[processing_method]:
                self.mean_correlations_dict[processing_method][correlation_method] = []

            # Aggiornamento del dizionario complessivo con il dizionario risultante
            self.mean_correlations_dict[processing_method][correlation_method] = mean_values_dict

        except Exception as e:
            print(f"    [WARNING] Errore nella fase di aggiornamento del dizionario di correlazione medio associato a {processing_method} - {correlation_method} --> {e}")
            self.warn_count += 1

        return self

    # Si inizializza una funzione che analizza tutte le coppie individuate, ne calcola il valore medio su tutte le missioni e stampa l'apposito file di testo
    def mean_correlation_analysis(self, correlation_methods, processing_methods, ROOT_DIR, threshold=0.33):

        # Si trasformano gli input in liste con un solo elemento in caso siano input di semplici stringhe --> possibile per come è strutturato il main, con più possibilità di scelta utente
        if isinstance(correlation_methods, str):
            correlation_methods = [correlation_methods]
        if isinstance(processing_methods, str):
            processing_methods = [processing_methods]

        # Si itera su ogni metodo di processing telemetrie selezionato
        for processing_method in processing_methods:

            # Si itera sui metodi di analisi prescelti
            for correlation_method in correlation_methods:
                
                print(f"\n  Calcolo e salvataggio dei coefficienti di correlazione mediati su tutte le missioni, trattate con {processing_method.upper()} per il metodo {correlation_method.upper()} in corso:")
                self.warn_count = 0

                # Si richiama il nome dei file generati all'interno della funzione correlation_matrix_to_txt
                output_dir_path = os.path.join(ROOT_DIR, f"Risultati_analisi_statistica/Analisi_{processing_method}/Analisi_{correlation_method}/Analisi_{correlation_method}_media")

                # Si verifica che la cartella non sia già stata creata in precedenza
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                output_file_path = os.path.join(output_dir_path, f"Report_analisi_{correlation_method}_valori_medi.txt")

                try:

                    # Si apre il file in modalità append e si genera l'intestazione della sezione dei valori medi
                    with open(output_file_path, "w", encoding="utf-8") as f:

                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"RIASSUNTO MEDIE DI TUTTE LE MISSIONI CON {processing_method} - METODO: {correlation_method.upper()}\n")
                        f.write("=" * 80 + "\n\n")

                        # Si definisce la copia del dizionario delle coppie per il metodo analizzato
                        correlations_dict_copy = self.correlations_dict.copy()

                        if correlation_method == 'cross-correlazione':

                            mean_values_dict = self.mean_coefficients_calculation(correlation_method, correlations_dict_copy, processing_method)

                            # Itera sulle chiavi del dizionario restituito (le coppie di variabili)
                            for couples_key, mean_value in mean_values_dict.items():

                                f.write(f"Relazione: {couples_key}\n")
                                f.write(f"  - Lag Medio: {mean_value['lag_ottimali_medi']:.2f} campioni\n")
                                f.write(f"  - Correlazione Max Media: {mean_value['coefficienti_medi']:.4f}\n")
                                f.write(f"  - Numero missioni: {len(correlations_dict_copy[processing_method][correlation_method][couples_key])}\n")
                                f.write("-" * 40 + "\n")

                        else:

                                # Si richiama la funzione per il calcolo dei coefficienti medi --> restituito un dizionario nella forma {'chiave_coppia': valore, ....}
                                mean_values_dict = self.mean_coefficients_calculation(correlation_method, correlations_dict_copy, processing_method)

                                # Si trasforma il dizionario in una serie pandas per poter fare operazioni vettoriali come la verifica della soglia
                                mean_values_series = pd.Series(mean_values_dict)

                                # Si filtrano i coefficienti medi complessivi per determinare quali superano il valore soglia e si ordinano in ordine crescente
                                valid_mean_values_series = mean_values_series[mean_values_series.abs() > threshold].sort_values(ascending=False)

                                # Si richiama la nuova funzione per la rappresentazione della heatmap associata ai coefficienti medi, se ci sono correlazioni valide
                                if not valid_mean_values_series.empty:
                                    self.mean_heatmap_definition(correlation_method, valid_mean_values_series, processing_method, ROOT_DIR)

                                # Si aggiorna il dizionario delle correlazioni medie
                                self.mean_correlation_dict_update(correlation_method, valid_mean_values_series, processing_method)

                                # Iterazione sugli elementi presenti nella lista dei coefficienti medi rilevanti --> a scopo di valutazione dettagliata dei dati
                                couple_mean_value: float
                                for couple_key, couple_mean_value in valid_mean_values_series.items():

                                    # Si definisce l'elenco di missioni per cui il valore supera la soglia --> spesso il coefficiente medio supera la soglia ma non vale lo stesso per tutte le singole missioni
                                    couple_values = correlations_dict_copy[processing_method][correlation_method][couple_key]       # Lista completa di pacchetti di dati associati alla coppia di misurazioni di iterazione --> sono nella forma ((traiettoria, combinazione), valore)
                                    valid_missions_array = [str(m) for m, v in couple_values if abs(v) > threshold]                 # Lista che contiene le tuple identificative di ogni missione in cui la correlazione della coppia è maggiore della soglia
                                    valid_missions_list = ", ".join(valid_missions_array)                                           # Si genera una unica stringa contenente la lista completa di tuple identificative delle missioni

                                    # Si definisce l'elenco delle missioni in cui il coefficiente è inferiore al valore medio
                                    under_mean_missions = [str(m) for m, v in couple_values if abs(v) < abs(couple_mean_value)]
                                    elenco_missioni_sotto_media = ", ".join(under_mean_missions)

                                    # Si scrivono le informazioni all'interno del file di testo
                                    f.write(f"- {couple_key}: Media = {couple_mean_value:.3f} --> nel dettaglio:\n")
                                    f.write(f"  - Correlazione sopra la soglia rilevata in {len(valid_missions_array)} missioni: {valid_missions_list} --> Nelle missioni {elenco_missioni_sotto_media} il coefficiente risulta inferiore al valore medio.\n")
                                    f.write(f"  - Correlazione inferiore al valore medio Nelle missioni {elenco_missioni_sotto_media}.\n\n")

                        f.write("\nFINE REPORT\n")

                        print(f'    Calcolo e formattazione completati con {self.warn_count} warnings rilevati.')

                except Exception as e:
                    print(f"    [WARNING] Errore nella fase di calcolo e formattazione su file di testo dei coefficienti medi per il metodo {correlation_method} applicato al database generato con {processing_method} --> {e}")
                    self.warn_count += 1

        return self




# Sotto-funzione per il calcolo della cross-correlazione tra due misurazioni specifiche --> tiene conto dello sfasamento temporale che ci può essere
def analizza_cross_correlazione(metodo_gestione_dati, metodo_correlazione, tabella_telemetria, colonna1, colonna2, numero_missione, max_lag):

    # Si definisce il nome della cartella di salvataggio
    nome_cartella_output = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_{metodo_correlazione}"

    # Verifica che la cartella non sia già stata creata in precedenza
    if not os.path.exists(nome_cartella_output):
        # Generazione della cartella
        os.makedirs(nome_cartella_output)

    # Si verifica che la deviazione standard sulla colonna di analisi sia nulla --> in questo caso si restituiscono subito valori nulli
    if tabella_telemetria[colonna1].std() < 1e-6 or tabella_telemetria[colonna2].std() == 0:
        return 0, 0.0

    # Si definisce una lista con tutti i possibili valori di lag equispaziati e si inizializza la lista con le correlazioni calcolate per ogni lag
    lags = range(-max_lag, max_lag + 1)
    correlazioni = []

    # Si itera per ogni valore di lag considerato
    for lag in lags:
        # Si calcola la correlazione traslando la colonna segnalata come 2 (quella non associata alla posizione) e si esegue la correlazione
        correlazione = tabella_telemetria[colonna1].corr(tabella_telemetria[colonna2].shift(lag))
        correlazioni.append(correlazione)

    # Si identifica il massimo valore di correlazione individuato --> di conseguenza anche il valore di lag ottimale
    index_max = np.argmax(np.abs(correlazioni))
    best_lag = lags[index_max]
    max_corr = correlazioni[index_max]

    # Si puliscono i nomi per i file
    def pulisci_nome(nome):
        return re.sub(r'[^a-zA-Z0-9]', '_', nome).strip('_')

    # --- SALVATAGGIO SU FILE DI TESTO ---
    file_report = os.path.join(nome_cartella_output, f"Report_Cross_Correlazione_M{numero_missione}.txt")
    with open(file_report, "a", encoding="utf-8") as f:
        f.write(f"Analisi: {colonna1} vs {colonna2}\n")
        f.write(f"  - Lag Ottimale: {best_lag} campioni\n")
        f.write(f"  - Correlazione Massima: {max_corr:.4f}\n")
        f.write("-" * 40 + "\n")

    return best_lag, max_corr


# Funzione principale necessaria per eseguire l'analisi PCA --> restituisce gli scree plot, le heatmap, i biplot e un file testuale associati a ogni missione
def analisi_pca(metodo_gestione_dati, tabella_telemetria, numero_missione):

    print(f"\nEsecuzione PCA per missione {numero_missione} in corso:")

    # Si identificano i sensori che producono le informazioni di riferimento per il risultato della rete neurale --> tutte quelli legati alla posizione
    parole_chiave_target = ['NED', 'body', 'X_ist', 'Y_ist', 'Z_ist', 'Depth [m]']
    colonne_target = [c for c in tabella_telemetria.columns if any(x in c for x in parole_chiave_target)]

    # Si produce la lista dei dati delle feature (sensori non target)
    dati_feature = tabella_telemetria.drop(columns=['timestamp'])                                                                   # Si rimuove la colonna legata ai timestamp
    dati_feature = dati_feature.select_dtypes(include=[np.number]).dropna()                                                         # Si pulisce il dataset eliminando tutte le colonne non numeriche (intestazioni) e quelle vuote
    dati_feature = dati_feature.loc[:, dati_feature.std() > 1e-6]                                                                   # Si pulisce il dataset eliminando tutte le colonne caratterizzate da deviazione standard quasi nulla (togliendo ad esempio DVL_Lock o rollio nelle missioni in cui non varia)
    dati_feature = dati_feature.drop(columns=[c for c in colonne_target if c in tabella_telemetria.columns], errors='ignore')       # Si rimuovono tutte le colonne considerate come target
    nomi_sensori = dati_feature.columns                                                                                             # Si definiscono i nomi dei sensori da utilizzare per grafici e file di testo

    # Si crea una sotto-tabella contenente solo le colonne target, allineate con i dati di feature (tramite .loc)
    dati_target = tabella_telemetria[colonne_target].loc[dati_feature.index]

    # Si effettua un'operazione di standardizzazione tramite il pacchetto StandardScaler --> necessario poiché si hanno dati con unità di misura e modulo molto diverso
    dati_feature_scalati = StandardScaler().fit_transform(dati_feature)

    # Si applica nel concreto la funzione per l'analisi tramite il pacchetto PCA --> si genera una matrice
    pca = PCA()
    componenti_principali = pca.fit_transform(dati_feature_scalati)

    # Si fa un'analisi di correlazione tra i componenti principali calcolati e i vari target di riferimento
    n_componenti_totali = componenti_principali.shape[1]                                                                                # Si memorizzano il numero delle componenti principali calcolate con PCA
    tabella_componenti_principali = pd.DataFrame(componenti_principali, columns=[f'PC{i + 1}' for i in range(n_componenti_totali)])     # Si crea una tabella con i componenti principali calcolati dalla PCA e si creano delle intestazioni

    # Si generano gli scree plot --> permettono di valutare per ogni missione quanti componenti partecipano alla varianza complessiva --> si produce la variabile varianza_spiegata
    varianza_spiegata = generazione_scree_plot_pca(metodo_gestione_dati, numero_missione, pca)

    # Si genera il file di testo contenente le informazioni riassuntive per ogni missione
    loadings_df = generazione_file_testo_pca(metodo_gestione_dati, numero_missione, pca, nomi_sensori, colonne_target, dati_target, tabella_componenti_principali)
    
    # Si generano le heatmap che permettono di rappresentare graficamente il dataframe generato per il file testuale
    generazione_heatmap_pca(metodo_gestione_dati, numero_missione, loadings_df)

    # Si generano i grafici biplot
    generazione_biplot_pca(metodo_gestione_dati, numero_missione, pca, nomi_sensori, componenti_principali, varianza_spiegata)

    print(f'Analisi PCA terminata e grafici generati.')

    return pca.explained_variance_ratio_


# Sotto-funzione per generare gli scree plot --> si restituisce la variabile varianza_spiegata che rappresenta .....
def generazione_scree_plot_pca(metodo_gestione_dati, numero_missione, pca):

    # Si crea la cartella di output apposita in cui inserire i grafici
    nome_cartella = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_PCA/Scree_plot"
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)

    # Si genera un array contenente la percentuale sulla varianza complessiva di ogni componente principale individuata dalla PCA
    varianza_spiegata = pca.explained_variance_ratio_

    # Si genera il grafico
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(varianza_spiegata) + 1), varianza_spiegata, alpha=0.7, label='Varianza individuale')
    plt.step(range(1, len(varianza_spiegata) + 1), np.cumsum(varianza_spiegata), where='mid', label='Varianza cumulata')
    plt.xlabel('Componenti Principali')
    plt.ylabel('Rapporto di Varianza Spiegata')
    plt.title(f'Scree Plot - Missione {numero_missione}')
    plt.legend(loc='best')
    plt.savefig(f"{nome_cartella}/ScreePlot_M{numero_missione}.png")
    plt.close()
    
    return varianza_spiegata


# Sotto-funzione per generare il file di testo riassuntivo con tutte le informazioni per ogni missione
def generazione_file_testo_pca(metodo_gestione_dati, numero_missione, pca, nomi_sensori, colonne_target, dati_target, tabella_componenti_principali):
    
    # Si crea la cartella di output apposita in cui inserire i grafici
    nome_cartella = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_PCA/File_testuali"
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)

    # Si definisce come dataframe per l'analisi quello composto da i vari componenti come colonne (per quello si usa il trasposto) e i sensori importanti in ogni componente come righe (si assegnano nomi dei sensori usati fino ad ora)
    n_componenti = pca.n_components_
    nomi_componenti = [f'PC{i + 1}' for i in range(n_componenti)]
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(n_componenti)], index=nomi_sensori)

    # Si aggiungono al dataframe le colonne legate ai target prescelti
    for col in colonne_target:
        tabella_componenti_principali[col] = dati_target[col].values

    # Analisi di Spearman che valuta i coefficienti di correlazione tra le componenti principali (che riassumono le features) e i target prescelti
    matrice_corr_target = tabella_componenti_principali.corr(method='spearman').loc[colonne_target, [f'PC{i + 1}' for i in range(15)]]

    # Si definisce il percorso di salvataggio di ogni file di output
    percorso_txt = f"{nome_cartella}/Analisi_spearman_componenti_M{numero_missione}.txt"
    with open(percorso_txt, 'w') as f:

        # Sezione legata alla varianza percentuale di ogni componente sul totale
        f.write("============================================================\n")
        f.write(f"ANALISI PCA GLOBALE - TUTTE LE {n_componenti} COMPONENTI\n")
        f.write("============================================================\n\n")

        for pc in nomi_componenti:
            f.write(f"--- {pc} (Varianza spiegata: {pca.explained_variance_ratio_[nomi_componenti.index(pc)] * 100:.2f}%) ---\n")

            # Prende i sensori più importanti per questa PC (valore assoluto)
            top_features = loadings_df[pc].abs().sort_values(ascending=False).head(10)
            for sensore, valore in top_features.items():
                segno = "+" if loadings_df.loc[sensore, pc] > 0 else "-"
                f.write(f"   * {sensore:<20} | Peso: {valore:.4f} ({segno})\n")
            f.write("-" * 60 + "\n")

        # Sezione legata alla scrittura delle 3 migliori componenti principali per ogni target
        f.write("="*60 + "\n")
        f.write(f"GUIDA ALLA SELEZIONE FEATURE PER RETE NEURALE - M{numero_missione}\n" + "=" * 60 + "\n")

        # Per ogni target si itera e si cerca la componente con correlazione maggiore
        for target in colonne_target:

            f.write(f"\nTARGET: {target}\n")

            '''
            miglior_pc = matrice_corr_target.loc[target].abs().idxmax()
            corr_val = matrice_corr_target.loc[target, miglior_pc]
            f.write(f"  -> PC più predittivo: {miglior_pc} (Correlazione: {corr_val:.3f})\n")

            # Elenca i principali sensori fisici che compongono quella componente principale
            top_sensori = loadings_df[miglior_pc].abs().sort_values(ascending=False).head(3)
            f.write(f"  -> Sensori fisici chiave da usare come input:\n")
            for s, peso in top_sensori.items():
                f.write(f"     * {s:25} (Importanza: {peso:.3f})\n")
            f.write("-" * 40 + "\n") '''

            top_3_pcs = matrice_corr_target.loc[target].abs().sort_values(ascending=False).head(3)

            for pc, corr_val in top_3_pcs.items():
                f.write(f"  -> {pc} (Correlazione Assoluta: {corr_val:.3f})\n")

                # Elenca i sensori chiave per questa specifica PC
                top_sensori = loadings_df[pc].abs().sort_values(ascending=False).head(3)
                for s, peso in top_sensori.items():
                    f.write(f"     * {s:25} (Importanza: {peso:.3f})\n")
            f.write("-" * 40 + "\n")

    return loadings_df


# Sotto-funzione per generare le heatmap, che rappresentano visivamente i dataframe con le componenti principali e i sensori importanti in ognuno
def generazione_heatmap_pca(metodo_gestione_dati, numero_missione, loadings_df):

    # Si crea la cartella di output apposita in cui inserire i grafici
    nome_cartella = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_PCA/Heatmap"
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)

    # Si costruiscono le heatmap mostrando tutte le componenti principali individuate
    plt.figure(figsize=(14, 10))
    n_pc_plot = min(10, len(loadings_df.columns))
    sns.heatmap(loadings_df.iloc[:, :n_pc_plot], cmap='RdBu_r', center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Influenza'})
    plt.title(f"Heatmap dei Caricamenti (Loadings) - M{numero_missione}\n(Blu = Correlazione Positiva, Rosso = Negativa)", fontsize=15)
    plt.ylabel("Sensori Telemetria")
    plt.xlabel("Componenti Principali")
    plt.savefig(f"{nome_cartella}/Heatmap_Loadings_M{numero_missione}.png", bbox_inches='tight', dpi=300)
    plt.close()

    return 0


# Sotto-funzione per generare i grafici biplot
def generazione_biplot_pca(metodo_gestione_dati, numero_missione, pca, nomi_sensori, componenti_principali, varianza_spiegata):

    # Si crea la cartella di output apposita in cui inserire i grafici
    nome_cartella = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_PCA/Biplot"
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)

    # Si genera nel concreto la figura
    plt.figure(figsize=(12, 10))

    # Plot dei punti (campioni della telemetria)
    plt.scatter(componenti_principali[:, 0], componenti_principali[:, 1], alpha=0.2, c='#2c3e50', s=2)

    # Plot delle frecce (loading dei sensori)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, var in enumerate(nomi_sensori):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.8, head_width=0.02)
        plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var, color='g', ha='center', va='center', fontsize=9)

    plt.xlabel(f'PC1 ({varianza_spiegata[0]:.2%})')
    plt.ylabel(f'PC2 ({varianza_spiegata[1]:.2%})')
    plt.title(f'PCA Biplot - Missione {numero_missione}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{nome_cartella}/Biplot_M{numero_missione}.png", bbox_inches='tight', dpi=300)
    plt.close()

    return 0


# Funzione principale per la valutazione media delle componenti principali uscenti dalla PCA
def analisi_pca_globale(metodo_gestione_dati, tabella_telemetria):

    print("\nAvvio Analisi PCA globale su tutte le missioni:")

    # Si crea la cartella di output apposita in cui inserire i grafici
    nome_cartella_output = f"Risultati_analisi/Analisi_con_{metodo_gestione_dati}/Analisi_PCA/Analisi_globale"
    if not os.path.exists(nome_cartella_output):
        os.makedirs(nome_cartella_output)

    # Si identificano i sensori che producono le informazioni di riferimento per il risultato della rete neurale --> tutte quelli legati alla posizione
    parole_chiave_target = ['NED', 'body', 'X_ist', 'Y_ist', 'Z_ist', 'Depth [m]']
    colonne_target = [c for c in tabella_telemetria.columns if any(x in c for x in parole_chiave_target)]

    # Si produce la lista dei dati delle feature (sensori non target)
    dati_feature = tabella_telemetria.drop(columns=['timestamp', 'DVL_Lock'])  # Si rimuove la colonna legata ai timestamp e al DVL_Lock (in missione 15 da problemi)
    dati_feature = dati_feature.select_dtypes(include=[np.number]).dropna()  # Si pulisce il dataset eliminando tutte le colonne non numeriche (intestazioni) e quelle vuote
    dati_feature = dati_feature.loc[:, dati_feature.std() > 0]  # Si pulisce il dataset eliminando tutte le colonne caratterizzate da deviazione standard quasi nulla (togliendo ad esempio DVL_Lock o rollio nelle missioni in cui non varia)
    dati_feature = dati_feature.drop(columns=[c for c in colonne_target if c in tabella_telemetria.columns], errors='ignore')  # Si rimuovono tutte le colonne considerate come target
    nomi_sensori = dati_feature.columns  # Si definiscono i nomi dei sensori da utilizzare per grafici e file di testo

    # Si crea una sotto-tabella contenente solo le colonne target, allineate con i dati di feature (tramite .loc)
    dati_target = tabella_telemetria[colonne_target].loc[dati_feature.index]

    # Si effettua un'operazione di standardizzazione tramite il pacchetto StandardScaler --> necessario poiché si hanno dati con unità di misura e modulo molto diverso
    dati_feature_scalati = StandardScaler().fit_transform(dati_feature)

    # Si applica nel concreto la funzione per l'analisi tramite il pacchetto PCA --> si genera una matrice
    pca = PCA()
    componenti_principali = pca.fit_transform(dati_feature_scalati)

    # Si crea il dataframe dei loadings --> colonne i componenti principali definiti da pca e righe i sensori presenti
    n_componenti = pca.n_components_
    nomi_componenti = [f'PC{i + 1}' for i in range(n_componenti)]
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(n_componenti)], index=nomi_sensori)

    # Si aggiungono al dataframe le colonne legate ai target prescelti
    tabella_componenti_principali = pd.DataFrame(componenti_principali, columns=nomi_componenti, index=dati_feature.index)      # Dataframe delle sole componenti principali
    for col in colonne_target:
        tabella_componenti_principali[col] = dati_target[col].values

    # Analisi di Spearman che valuta i coefficienti di correlazione tra le componenti principali (che riassumono le features) e i target prescelti
    matrice_corr_target = tabella_componenti_principali.corr(method='spearman').loc[colonne_target, [f'PC{i + 1}' for i in range(min(n_componenti, 15))]]

    # Si genera una heatmap con la matrice di correlazione generata
    plt.figure(figsize=(26, 12))
    sns.heatmap(matrice_corr_target, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Matrice di Correlazione di {matrice_corr_target} per PCA globale", fontsize=22,
              fontweight='bold', pad=20)
    plt.savefig(f"{nome_cartella_output}/Heatmap_Correlazioni_Globale.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Si genera una heatmap con i valori globali
    plt.figure(figsize=(20, 12))
    sns.heatmap(loadings.iloc[:, :20], annot=True, cmap='RdBu_r', fmt=".2f")
    plt.title(f"Heatmap Globale - Prime 20 Componenti (di {n_componenti})")
    plt.savefig(f"{nome_cartella_output}/Heatmap_Componenti_Globale.png", bbox_inches='tight')
    plt.close()

    # Si procede con la scrittura delle informazioni all'interno del file di testo
    with open(f"{nome_cartella_output}/Riassunto_testuale.txt", "w", encoding='utf-8') as f:

        f.write("============================================================\n")
        f.write(f"ANALISI PCA GLOBALE - TUTTE LE {n_componenti} COMPONENTI\n")
        f.write("============================================================\n\n")

        for pc in nomi_componenti:
            f.write(f"--- {pc} (Varianza spiegata: {pca.explained_variance_ratio_[nomi_componenti.index(pc)] * 100:.2f}%) ---\n")

            # Prende i sensori più importanti per questa PC (valore assoluto)
            top_features = loadings[pc].abs().sort_values(ascending=False).head(10)
            for sensore, valore in top_features.items():
                segno = "+" if loadings.loc[sensore, pc] > 0 else "-"
                f.write(f"   * {sensore:<20} | Peso: {valore:.4f} ({segno})\n")
            f.write("-" * 60 + "\n")

        f.write("=" * 60 + "\n")
        f.write(f"GUIDA ALLA SELEZIONE FEATURE PER RETE NEURALE - GLOBALE\n" + "=" * 60 + "\n")

        # Per ogni target si itera e si cerca la componente con correlazione maggiore
        for target in colonne_target:

            f.write(f"\nTARGET: {target}\n")

            '''miglior_pc = matrice_corr_target.loc[target].abs().idxmax()
            corr_val = matrice_corr_target.loc[target, miglior_pc]
            f.write(f"  -> PC più predittivo: {miglior_pc} (Correlazione: {corr_val:.3f})\n")

            # Elenca i principali sensori fisici che compongono quella componente principale
            top_sensori = loadings[miglior_pc].abs().sort_values(ascending=False).head(3)
            f.write(f"  -> Sensori fisici chiave da usare come input:\n")
            for s, peso in top_sensori.items():
                f.write(f"     * {s:25} (Importanza: {peso:.3f})\n")
            f.write("-" * 40 + "\n")'''

            top_3_pcs = matrice_corr_target.loc[target].abs().sort_values(ascending=False).head(3)

            for pc, corr_val in top_3_pcs.items():
                f.write(f"  -> {pc} (Correlazione Assoluta: {corr_val:.3f})\n")

                # Elenca i sensori chiave per questa specifica PC
                top_sensori = loadings[pc].abs().sort_values(ascending=False).head(3)
                for s, peso in top_sensori.items():
                    f.write(f"     * {s:25} (Importanza: {peso:.3f})\n")
            f.write("-" * 40 + "\n")

    print(f"Analisi globale completata, creati il file testuale e l'heatmap.")

    return 0





