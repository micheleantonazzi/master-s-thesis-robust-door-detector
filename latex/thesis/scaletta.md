# Scaletta

## Capitolo 2: State of the Art

* Inizio introducendo il problema del riconoscere le porte, cito qualche articolo
* Spiego perchè è utile per la robotica (map segmentation e cito qualche lavoro)
* Dico che gran parte di questi lavori usano moduli di computer vision end to end
* spiego che in questo task è molto usata la compouter vision (object detection in particolare) e cito le milestone in questo ambito (YOLO; Transformer ecc) e porto le metriche più famose e utilizzate per valutare tali modelli
* spiego che l'applicazione del deep learning alla robotica presenta delle criticità e dei problemi ancora aperti (cito il survey)
* il survey cita anche il problema di acquisire dai in simulazione (per la loro differenza dai reali e per la presenza di imprecisioni), posso collegarmi e citare alcuni ambienti di simulazione e alcuni dataset di ambienti
* Concludo dicendo che nella tesi si cerca di trovare superare alcune delle criticità citate sopra. Si propone modello per riconoscere porte utlizzando un modello end-to-end (DETR) e un metodo per aumentare le sue performance sfruttando il principio del wayfinding (*lo cito prima o qui??*) e offrendo un metodo di valutazione del modello più esaustivo e adatto ad un contesto di mobile robotics.

## Capitolo 3: Problem Formulation

* Piccolo preambolo in cui riscrivo lo scopo della tesi (formulazione di un metodo per aumentare le performance di un modello deep che riconosce le porte, sfruttando il principio di wayfinding e offrendo una metrica più esaustiva di quelle classiche usate in computer vision).

### Motivazioni e goal

Qui si possono scrivere le motivazioni della tesi. Scrivo in modo più esaustivo le difficoltà di applicare deep learning alla robotica, quali sono le limitazioni (prendo spunto dal survey) e quali tento di risolvere con questa tesi. Specifico quindi quali sono gli obiettivi.

### Definizioni

* Definisco cos'è una porta per me (implicita esplicita)
* definisco i dati che utilizzo (RGB, bounding box)
* definisco cosè un modello end-to-end (formule)
* definisco un modello deep per l'object detection
* definisco le metriche che vengono più utilizzare (COCO, Pascal VOC, AP, mAP con le formule)

### Formulazione del problema

* Spiego che utilizziamo un metodo di machine learning supervisionato
* Definisco cos'è un problema di machine learning supervisionato, un modello e cos'è la funzione di loss ecc (review machine learning principles)
* Spiego che per aumentare le perfomance di questo classificatore è possibile fare fine tune in quanto le porte saranno simili (wayfinding), cos'è fine tune e come si fa (magari con formule)
* Spiego perchè le metriche classiche per valutare un modello di object detection non sono sufficienti.

### Assunzioni

Riporto le assunzioni che faccio:

* sul robot (il fatto che è simulato e che si teletrasporta nello spazio, altezza della camera, risoluzione della camera)
* su gibson, quindi sull'ambiente si simulazione (assumo che i dati RGB acquisiti siamo realistici anche se sono processati da un modello deep che riempie i buchi, che simuli correttamente luce e ombre ecc. )
* sugli ambienti (devono essere indoor, sono statici, sono di vario tipo case, uffici, laboratori ecc)
* sull'algoritmo di acquisizione dei dati (assumo che l'algoritmo scelga location plausibili per il robot e non incontri mai ostacoli come muri, mobili o scale)
* sui dati (assumo che siamo stai utilizzati tutti quelli raccolti e che le label siamo corrette)

## Capitolo 4: Soluzione logica

### Data acquisition

* Spiego come ho acquisito i dati. Dico che ho usato la simulazione e perchè (devo acquisire tanti dati da molti ambienti diversi, più economico e veloce ecc)
* Spiego bene il problema di acquisire dati da gibson e matterport (il robot simulato non riesce a navigare)
* Spiego in linea generale come l'ho risolto, creando un algoritmo che data la mappa 2D di un ambiente ne estratta un grafo di possibili posizioni che il robot può acquisire, che verranno poi campionate utilizzando una valore che indica la distanza l'una dall'altra

### Data manipulation and correction

* i dati acquisiti sono labellati a mano a causa dell'imprecisione di Gibson e matterport e dei dati semantici

### Model evaluation

Spiego come è costruito il dataset e come si puo utilizzare per addestrare DETR e fare gli esperimenti. Descrivo la tecnica che ho adottato per aumentare le performance del calssificatore facendo un fine tune incrementale con esempi raccolti dall'ambiente in questione (**Potremmo chiamare la tecnica incremental fine-tune(?))**

## Capitolo 5: Proposed solution

Faccio una overview generale del sistema e di tutte le parti software della tesi (simulatore, pose estimator, software per gestire il dataset, acquisizione dati, modello end-to-end, modulo per valure i modelli ottenuti)

Per ogni modulo faccio una sotto-sezione in cui spiego il suo funzionamento (eventuali algoritmi implementati) e le tecnologie utilizzate.

* Per il simulatore spiego dettagliatamente il suo funzionamento e le modifiche che ho apportato a Gibson linkando il codice aggiornato
* Per il pose estimator spiego esattamente l'algoritmo per estrarre le posizioni da cui acquisire i dati e i parametri utilizzati
* Per il modello end-to-end spiego il suo funzionamento, l'architettura e i principi matematici che lo caratterizzano (formule, losses ecc)
* per il modulo di valutazione spiego in dettaglio le metriche adottate con le opportune formule

## Capitolo 6: System architecure

In questo capitolo verranno descritti più a basso livello i vari componenti software elencati precedentemente. In particolare, verranno descritte le tecnologie utilizzare, le funzioni di libreria impiegate negli algoritmi riportati e i loro parametri per garantire la riproducibilità degli esperimenti

## Capitolo 7

In questo capitolo verranno riportati i dettagli degli esperimenti

### Data acquisition

Verranno specificati i mondi di matterport utilizzati (quanti e quali) e la procedure per acquisire le immagini (dataset)

### Image dataset

Viene descritto in dettaglio il dataset di immagini negative e positive che verrà utilizzato per gli esperimenti finali (numero di immagini ecc).

### A preliminary test of DETR

Verrà definito il procedimento di valutazione di DETR come modello per riconoscere porte. Attraverso un dataset di immagini di porte già ampiamente conosciuto (deep doors 2), verrà valutato se DETR è adatto per queto tipo di task. Verrà quindi addestrato e valutato il modello utilizzando questo dataset esterno (la valutazione si può fare utilizzando le stesse metriche utilizzate per latesi o le metriche utilizzate nell'articolo del dataset, ma devo capire quanto è complicato utilizzare il loro metro di valutazione).

### Training and testing procedure

Verrà spiegato dettagliatamente la procedura per effettuare il training dei modelli utilizzando il dataset da me raccolto. Poi si porcederà a illustrare la procedure di testing dei modelli addestrati

### Results

In questa sezione verranno riportati i risultati derivanti dal training e testing

* training: tempo di addestramento (sia del modello generico sia dei vari fine tune), grafico losses
* testing: valori delle metriche, grafici AP per ogni label

## Capitolo 8 

Qui verranno riportati i fatti deducibili dalla fase di sperimentazione e possibili future direzioni di ricerca per migliorare questo lavoro (evidenziandone quindi le lacune)

## 

