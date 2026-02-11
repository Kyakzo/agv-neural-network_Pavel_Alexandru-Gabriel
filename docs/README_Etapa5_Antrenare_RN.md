# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pavel Alexandru-Gabriel  
**Link Repository GitHub:** https://github.com/Kyakzo/agv-neural-network_Pavel_Alexandru-Gabriel
**Data predării:** 19/12/2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

---

## 1. Antrenarea Modelului

Modelul a fost antrenat folosind datele colectate și preprocesate. S-a utilizat un set de date împărțit stratificat pentru a asigura o reprezentare corectă a stărilor robotului.

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [X] **State Machine** definit și documentat în `docs/state_machine.*`
- [X] **Contribuție ≥40% date originale** în `data/generated/` (verificabil - toate datele sunt generate si plasate in raw)
- [X] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [X] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`) - (datele sunt generate intr-un CSV)
- [X] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [X] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare**                                                                                                              |
|--------------------|------|------------------------------------------------------------------------------------------------------------------------------|
| Learning rate | 0.001 | Valoare standard pentru optimizatorul Adam; asigură o convergență rapidă și stabilă fără a "sări" peste minimul global.      |
| Batch size | 32 | Compromis optim între viteza de calcul și stabilitatea gradientului pentru un set de date de dimensiune medie (~3000 mostre) |
| Number of epochs | 50 | Suficient pentru ca modelul să învețe pattern-urile simple (urmarire linie) fără a intra în overfitting (supra-antrenare).   |
| Optimizer | Adam | Algoritm cu rată de învățare adaptivă, foarte eficient pentru rețele neurale simple și date cu zgomot.                       |
| Loss function | MSE (Mean Squared Error) | Regresie: Dorim să minimizăm eroarea pătratică dintre viteza prezisă și cea reală a motoarelor (valori continue).            |
| Activation functions | ReLU (hidden), Linear (output) | ReLU în straturile ascunse pentru a învăța relații non-liniare; Linear la ieșire pentru a permite valori negative (mers înapoi) și pozitive.                                                             |

**Justificare detaliată batch size:**
```
Am ales batch_size=32 pentru că reprezintă un echilibru optim între mai mulți factori importanți în antrenarea rețelelor neurale:
1. Eficiență computațională:
   * Procesoarele (CPU și mai ales GPU) sunt optimizate pentru operații vectoriale paralele. Procesarea a 32 de mostre simultan permite o utilizare eficientă a resurselor hardware, accelerând semnificativ timpul de antrenare pe epocă în comparație cu dimensiuni de batch foarte mici (e.g., 1, 2 sau 4).
   * Este suficient de mare pentru a beneficia de paralelism, dar nu atât de mare încât să ducă la sub-utilizarea anumitor unități de calcul sau la timpi morți.
2. Stabilitate gradient (batch prea mic → zgomot mare în gradient)
3. Generalizare:
   * Deși batch-urile mai mici (e.g., 1, 2) pot introduce mai mult "zgomot" în procesul de antrenare, ajutând teoretic la evitarea minimelor locale superficiale și la o mai bună generalizare, o dimensiune de 32 este încă
     suficient de mică pentru a menține o anumită varianță în estimarea gradientului. Această varianță moderată poate contribui la o generalizare robustă, fără a sacrifica stabilitatea antrenamentului.
   * Batch-urile foarte mari, pe de altă parte, pot duce la o convergență către minime mai "ascuțite" și pot reduce capacitatea de generalizare a modelului.
4. Memorie: dimensiunea de 32 este relativ mică și permite antrenarea modelului chiar și pe hardware cu resurse de memorie (RAM/VRAM) limitate, prevenind erorile de tip "Out of Memory".
```

### 2. Rezultate și metrici (test set)

Evaluarea finală a fost realizată pe setul de test, care nu a fost văzut de model în timpul antrenării.

| Metrica | Valoare | Observații |
|---------|---------|------------|
| **Accuracy** | **63.86%** | Procentul de predicții corecte în raport cu o marjă de eroare acceptabilă. |
| **F1-score (macro)** | **0.26** | Scorul reflectă dificultatea modelului de a gestiona clasele minoritare (virajele). |
| **Test Loss (MSE)** | 0.1558 | Eroarea medie pătratică pe setul de test. |
| **Test MAE** | 0.3571 | Eroarea medie absolută (eroarea medie per motor). |

*Date extrase din `results/test_metrics.json` și `results/training_history.csv`.*

---

### Graficul Loss (antrenare vs validare)

Modelul a prezentat o curbă de învățare descendentă, cu o convergență clară spre epoca 29, unde procesul a fost întrerupt de mecanismul de Early Stopping pentru a preveni supra-antrenarea.
Vizualizare: `docs/loss_curve.png`

---

## 3. Integrare și Demonstrație UI

Modelul antrenat (`models/trained_model.keras`) a fost încărcat în modulul de control al robotului (`src/app/main.py`). Spre deosebire de etapa 4 unde se folosea un model dummy, acum robotul ia decizii bazate pe pattern-urile învățate din date.

**Screenshot Inferență reală:**
O demonstrație a modelului în funcțiune poate fi găsită în: `docs/screenshots/inference_real.png`

---

## 4. Analiză erori in context industrial

### 1. Pe ce clase greșește cel mai mult modelul?
Modelul are dificultăți în principal la virajele strânse. Deoarece dataset-ul este dominat de mersul înainte (linie dreaptă), modelul tinde să aibă un bias către această acțiune, confundând adesea începutul unui viraj cu mersul rectiliniu.

### 2. Ce caracteristici ale datelor cauzează erori?
Zgomotul redus din simulare face ca modelul să fie foarte sensibil la variații mici ale senzorilor. De asemenea, lipsa datelor de tip "recovery" (când robotul este deja pe jumătate ieșit de pe linie) face ca rețeaua să nu știe cum să revină corect pe trasă în situații extreme.

### 3. Ce implicații are pentru aplicația industrială?
O acuratețe de ~64% înseamnă că AGV-ul ar putea pierde linia în 1 din 3 situații complexe. Pentru un mediu industrial real, acest lucru necesită implementarea unor sisteme redundante de siguranță sau re-antrenarea modelului cu un set de date mai diversificat (balansat).

### 4. Ce măsuri corective propuneți?
1.  **Oversampling pentru viraje:** Dublarea sau triplarea mostrelor în care robotul execută viraje pentru a forța rețeaua să acorde importanță acestor stări.
2.  **Data augmentation:** Adăugarea de jitter temporal și zgomot gaussian pe valorile senzorilor pentru a simula imperfecțiunile hardware-ului real.
3.  **Tuning arhitectural:** Experimentarea cu straturi de tip Dropout pentru a reduce dependența de anumite caracteristici și a îmbunătăți generalizarea pe date noi.

## Verificare consistență cu State Machine (etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în etapa 4.

**Exemplu pentru monitorizare vibrații lagăr:**

| **Stare din etapa 4** | **Implementare în etapa 5**                                                                                          |
|-----------------------|----------------------------------------------------------------------------------------------------------------------|
| `INITIALIZATION`      | Încărcarea configurărilor (config.py) și a resurselor (imagini traseu, model neural).                                |
| `DATA_COLLECTION`     | Utilizatorul conduce robotul; la fiecare ciclu se citesc senzorii și se scriu datele în CSV.                         |
| `TRAINING_PIPELINE`   | Stare offline în care se prelucrează CSV-ul și se antrenează modelul Keras.                                          |
| `AUTONOMOUS_CONTROL`  | Starea principală de funcționare a SIA. Robotul citește senzorii, execută inferența RN și aplică comanda la motoare. |
| `REMOTE_LOGGING`      | În paralel cu controlul, robotul trimite starea curentă către serverul Web via HTTP POST.                            |

## Structura repository-ului la finalul etapei 5

agv-neural-network/
  ├── README.md
  ├── docs/
  │   ├── state_machine.png             
  │   ├── loss_curve.png
  │   ├── datasets/
  │   ├── README_Etapa4_Arhitectura_SIA.md
  │   ├── README_Etapa5_Antrenare_RN.md
  │   └── screenshots/
  │       ├── inference_real.png         
  │       └── ui_demo.png                
  │
  ├── data/                               
  │   ├── raw/
  │   │   └── date_robot.csv             
  │   ├── processed/       #Versiuni mai vechi procesate  
  │   ├── README.md
  │   └── train/ #Nimic aici momentan                         
  │
  ├── src/
  │   ├── data_acquisition/              
  │   │   └── simulator.py               
  │   ├── neural_network/
  │   │   └── antrenare.py               
  │   └── app/
  │       ├── app.py                     
  │       └── server.py                  
  │
  ├── models/
  │   └── trained_model.keras           
  │
  ├── results/                            
  │   ├── training_history.csv           
  │   ├── test_metrics.json              
  │   └── hyperparameters.yaml          
  │
  ├── config/
  │   └── config.py                    
  ├── .gitignore
  └── requirements.txt

## Checklist final – Bifați totul inainte de predare

### Prerequisite Etapa 4 (verificare)
- [X] State Machine există și e documentat în `docs/state_machine.*`
- [X] Contribuție ≥40% date originale verificabilă în `data/raw/`
- [X] Cele 3 module din etapa 4 funcționale

### Preprocesare și Date
- [X] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [X] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [X] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`) - nu folosesc.

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [X] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [X] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [X] Tabel hiperparametri + justificări completat în acest README
- [ ] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [X] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [X] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [X] Model ANTRENAT încărcat în UI din etapa 4 (nu model dummy) (se incarca in simulator)
- [X] UI face inferență REALĂ cu predicții corecte
- [X] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [X] Verificat: predicțiile sunt diferite față de etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [X] Early stopping implementat și documentat în cod
- [X] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [X] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [X] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [X] Analiză erori în context industrial completată (4 întrebări răspunse)
- [ ] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [ ] `requirements.txt` actualizat cu toate bibliotecile noi
- [X] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [X] Cod nou comentat în limba română sau engleză (minimum 15%)
- [X] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [X] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare state machine (etapa 4)
- [X] Fluxul de inferență respectă stările din State Machine
- [X] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [X] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [x] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [X] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [X] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [X] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [X] Push: `git push origin main --tags`
- [X] Repository accesibil (public sau privat cu acces profesori)

## Predare și contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**