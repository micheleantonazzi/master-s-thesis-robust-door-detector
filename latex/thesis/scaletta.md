# Scaletta

## Capitolo 2: State of the Art

* Inizio spiegando che il riconoscimento di porte puo essre utile per un mobile robot.
* Cito qualche lavoro di door detection con feature based method
* spiego che in object detection i metodi deep hanno soppiantato i vecchi metodi feature based
* Faccio una carrellata delle milestone più importanti in object detection per definirne un po la storia evolutiva (ho due survey da cui prendere spunto). Il primo arriva fino a retinanet, il secondo parla di objet detection usando transformer.
* Procedo col citare alcuni lavori dei object detection che utlizzano metodi deep
* spiego che l'applicazione del deep learning alla robotica presenta delle criticità e dei problemi ancora aperti (cito il survey)
* il survey cita anche il problema di acquisire dai in simulazione (per la loro differenza dai reali e per la presenza di imprecisioni), posso collegarmi e citare alcuni ambienti di simulazione e alcuni dataset di ambienti

## Capitolo 3: Problem Formulation 

### Motivazioni e goal

Qui descrivo lo scopo della tesi: creazione di un modulo per il riconoscimento di porte applicato ad un mobile robot. Spiego brevemente che un robot può essere spostato da un ambiente a uno diverso, l'assunzione è che avvenga con bassa frequenza e nel momento di un deployment iniziale (i concetti di deployment e impiego a lungo/medio termine servono per spiegare bene il contributo della tesi, dovrai riprenderli alla fine quando scrivi l'intro) robot andrà ad operare in uno stesso ambiente, che in un ambiente le porte sono simili e che quindi si può sfruttare il wayfinding per aumentare le performance del modello. Riporto inoltre brevemente quali sono alcune criticità di applicare modelli deep in robotica (come l'assenza di metriche e l'assenza di dataset per la robotica). Spiego che il dataset lo acquisisco da me in modo da costruirne uno adatto ad un task robotico e spiego che voglio offrire un miglior metodo di valutazione.

### Descrizione del problema

* Definisco cos'è una porta per me (implicita esplicita)
* definisco i dati che utilizzo (RGB, bounding box)
* sugli ambienti (devono essere indoor, sono statici, sono di vario tipo case, uffici, laboratori ecc)
* definisci anche che cosa vuoi che il robot faccia, quale è lo scenario ideale che hai in mente. Il robot è deployato, si sposta per fare qualcosa, riconosce una porta (aperta/chiusa), agisce di conseguenza etc E sottolinea che sono una serie di scelte che caratterizzano la tua tesi

## Capitolo 4: Proposed solution in detail

### Preliminaries and work overview

Spiego come ho risolverò il problema:

* Definisco (molto brevemente) cos'è un problema di machine learning supervisionato, un modello e cos'è la funzione di loss ecc (review machine learning principles) e il fine tuning.
* spiego che per sfruttare il fatto che le porte sono simili (wayfinding) si farà un fine tune del modello solo con immagini acquisite e labellate manualmente da quel mondo. Dico che il fine tuning sarà incrementale per valutare la quantità di immagini da utilizzare (**qui definisco che la mia tecnica si chiamerà incremental-learning**). In questo modo si aumentano le performance del classificatore
* Dico che mi sono costruito il dataset con l'obiettivo di ottenere immagini di porte che siamo viste da più angolazioni e da posizioni plausibili per il robot
* Il modulo per riconoscere le porte sarà un modello deep utilizzato sviluppato per object detection. Dico che si tratta di DETR, un modello basato sui transformer, così si effettua anche un esperimento per valutare questi modelli sul task delle porte, cosa che in letteratura ancora non c'è
* spiego che per migliorare il metodo di valutazione è importante andare a valutare anche le immagini negative (senza porte)

Faccio poi una panoramica di come hai sviluppato il lavoro. Qui stai generico dicendo: serve un dataset e un modo per costruirlo, serve un metodo di clasificazione etc (in parte come già hai scritto nella scaletta) e metriche piu esaustive.

### Proposed solution

Qui descrivo quali sono le parti principali del mio sistema. Faccio prima una overview generale del sistema con tutte le sue parti software. Per ognuna poi ci sarà una sezione dedicata:

#### Simulazione

Qui scrivo che i dati sono stati acquisiti in simulazione (per averne tanti e per ridurre i costi di tempo). Spiego quale simulatore ho usato (Gibson) e il dataset di mondi (Matterport). Spiego in dettaglio il funzionaneto del simulatore, il fatto che il robot non naviga e le modifiche che ho apportato per risolvere il problema.

#### Pose estimator

Spiego che ho realizzato un modulo per la stima delle posizioni del robot (dato che non usa il suo navigation stack) e spiego come funziona

#### Image dataset

Viene descritto in dettaglio il dataset di immagini negative e positive che verrà utilizzato per gli esperimenti finali (numero di immagini ecc). Spiego in dettaglio il funzionamento del software che o scritto per gestire ed elaborare il dataset

#### DETR

Riporto il modello scelto per riconoscere le porte. Ne descrivo il funzionamento (implementazione e loss functions). Riporto anche l'esperimento che ho fatto con il dataset di porte trovato su internet  (deep doors 2) e spiego che mi è servito sia per imparare a fine-tunare questo enorme modello sia per capire se è adatto per il task di door detection.

#### Model evaluator

qui scrivo esattamente come si svolge la tecnica dell'incremental learning. Spiego inoltre in modo dettagliato come è implementata la nuova metrica delle immagini negative.

## Capitolo 7: Results

### Experimental setting

Qui riporto in dettaglio quali esperimenti, quali ambienti, che modalità di valutazione, quali parametri fai variare, insomma la descrizione degli esperimenti e il loro funzionamento.

### Results

Qui vengono riportati i risultati della fase sperimentale (grafici e tabelle, risultati di training, testing ecc) e verranno riportate le considerazioni e i fatti deducibili dai risultati. *Scrivo il future work*

