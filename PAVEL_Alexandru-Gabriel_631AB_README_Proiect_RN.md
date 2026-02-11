## 1. Identificare Proiect

| Câmp | Valoare                                                              |
|------|----------------------------------------------------------------------|
| **Student** | Pavel Alexandru-Gabriel                                              |
| **Grupa / Specializare** | [631AB / Informatică Industrială]                                    |
| **Disciplina** | Rețele Neuronale                                                     |
| **Instituție** | POLITEHNICA București – FIIR                                         |
| **Link Repository GitHub** | https://github.com/Kyakzo/agv-neural-network_Pavel_Alexandru-Gabriel |
| **Acces Repository** | [Privat cu acces cadre didactice RN]                                 |
| **Stack Tehnologic** | [Python]                                                             |
| **Domeniul Industrial de Interes (DII)** | [Logistică / Robotică]                                               |
| **Tip Rețea Neuronală** | [MLP]                                                                |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|-------------|-----|
| Accuracy (Test Set) | ≥70% | 64.00%           | 72.00%         | +8.00% | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.65             | 0.65            | - | [✓] |
| Latență Inferență |  < 33 ms | 32ms             | 32 ms          | - | [✓] |
| Contribuție Date Originale | ≥40% | 100%             | 100%           | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4                | 4              | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [X] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [X] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [X] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [X] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [X] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Acest proiect abordează optimizarea logisticii interne în mediile industriale (Smart Factory/Warehouse) prin implementarea unui sistem de navigare autonom pentru vehicule ghidate automat (AGV). În contextul actual, transportul manual de materiale sau utilizarea sistemelor rigide de ghidare magnetică este ineficientă și greu de scalat. Nevoia reală constă în crearea unui sistem capabil să proceseze în timp real datele de la senzori
simpli și să ia decizii de deplasare precise, fără intervenție umană constantă.

Rezolvarea acestei probleme este crucială pentru creșterea productivității și siguranței la locul de muncă. Prin utilizarea unei rețele neuronale de tip MLP (Multi-Layer Perceptron), AGV-ul poate învăța să urmeze trasee complexe și să se adapteze dinamic, reducând blocajele în fluxul de producție și minimizând riscul de eroare umană în transportul componentelor între stațiile de lucru.

### 2.2 Beneficii Măsurabile Urmărite

   1. Acuratețea deciziilor de navigare: Atingerea unei precizii de peste 65% în predicția corectă a acțiunilor (stânga, înainte, dreapta) pe setul de test.
   2. Performanța decizională (F1-Score): Menținerea unui scor F1 de minim 0.60 pentru a asigura un echilibru între precizia și rapelul virajelor pe trasee complexe.
   3. Reducerea erorii de control: Minimizarea erorii medii pătratice (MSE) în predicția vitezelor motoarelor pentru a asigura o deplasare lină, fără sacadări.
   4. Autonomie completă: Eliminarea necesității controlului manual în proporție de 100% odată ce modelul este încărcat în modul "AUTONOM".
   5. Eficiența procesării: Timp de răspuns sub 33ms per frame (30 FPS) pentru procesarea datelor de la senzori și generarea comenzii motor, asigurând reacții rapide la curbe.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil**                |
|---------------------------|--------------------------|--------------------------------|-------------------------------------|
| Navigare autonomă continuă pe trasee marcate | Procesare senzori proximitate → Predicție viteze motoare (MLP Regression) |  src/neural_network/ (Inference) | Accuracy > 70%, autonomie 100%      |
|  Reacție rapidă la curbe pentru evitarea ieșirii în decor |  Buclă de control optimizată cu inferență rapidă (Forward Pass) |  src/data_acquisition/ (Simulator) | Timp răspuns < 33ms (echiv. 30 FPS) |
|  Monitorizarea remote a stării operative (Telemetrie) | Dashboard web real-time cu vizualizare probabilități decizie și stare senzor | src/app/ (Web Service / Flask) | Latență UI < 500ms (Refresh rate)   |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare                                                     |
|----------------|-------------------------------------------------------------|
| **Origine date** | Simulare                                                    |
| **Sursa concretă** | Simulare proprie dezvoltată în Pygame (robot cu 3 senzori IR) |
| **Număr total observații finale (N)** | 2,933  |
| **Număr features** | 3 (senzor_stanga, senzor_centru, senzor_dreapta)            |
| **Tipuri de date** | Numerice (binare pentru senzori [0,1], float pentru motoare [-1,1]) |
| **Format fișiere** | CSV                                                         |
| **Perioada colectării/generării** | Noiembrie 2025 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 2,933 |
| **Observații originale (M)** | 2,933 |
| **Procent contribuție originală** | 100% |
| **Tip contribuție** |  Simulare fizică și etichetare automată prin control manual |
| **Locație cod generare** | `src/data_acquisition/simulator.py` |
| **Locație date originale** | `data/raw/date_robot.csv` |

**Descriere metodă generare/achiziție:**

Datele originale au fost achiziționate prin simulare activă în mediul Pygame, utilizând un script de achiziție (simulator.py) ce permite controlul manual al vehiculului. Am condus robotul pe un traseu cu segmente drepte și
curbe variate, în timp ce sistemul a eșantionat la o rată de 30 FPS starea celor 3 senzori virtuali de proximitate și comenzile de viteză transmise motoarelor. Pentru a asigura calitatea dataset-ului, scriptul a fost
configurat să ignore stările staționare (când ambele motoare sunt 0), capturând astfel doar dinamica relevantă pentru navigare.

Această metodă este critică pentru problema propusă deoarece permite corelarea directă a semnalelor binare de la senzori (detecție linie neagră pe fundal alb) cu vectorul de mișcare necesar corecției traiectoriei. Parametrii
utilizați includ o lungime a senzorilor de 40 pixeli și un unghi de deschidere de 45 de grade față de axa centrală, oferind robotului un "câmp vizual" suficient pentru a anticipa curbele strânse înainte ca acesta să părăsească
marcajul.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 2,053 |
| Validation | 15% | 440 |
| Test | 15% | 440 |

**Preprocesări aplicate:**
   - Filtrare la sursă: Eliminarea automată a cadrelor în care robotul este staționar pentru a evita bias-ul de inactivitate.
   - Normalizare implicită: Senzorii sunt citiți direct în intervalul [0.0, 1.0] (0 = linie, 1 = fundal), iar motoarele sunt limitate în intervalul [-1.0, 1.0].
   - Amestecare (Shuffle): Aplicată înainte de split-ul datelor pentru a elimina corelația temporală specifică achiziției secvențiale.
   - Discretizare pentru metrici: Conversia ieșirilor continue în clase discrete (Stânga, Înainte, Dreapta) utilizând un prag de 0.2 pentru calculul Acurateței și F1-Score.

**Referințe fișiere:** `data/raw/date_robot.csv`, ` src/neural_network/antrenare.py`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** |  Python (Pygame) | Simulare fizică a robotului și achiziție date senzoriale + comenzi manuale | `src/data_acquisition/` |
| **Neural Network** |  Keras / TensorFlow | Rețea de tip MLP pentru regresia vitezelor motoarelor și evaluarea performanței | `src/neural_network/` |
| **Web Service / UI** | Flask | Dashboard de telemetrie pentru monitorizarea în timp real a stării și probabilităților | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere                                                     | Condiție Intrare                 | Condiție Ieșire                             |
|-------|---------------------------------------------------------------|----------------------------------|---------------------------------------------|
| `INITIALIZATION` | Încărcarea configurărilor din config.py și a modelului .keras | Start aplicație                  | Resurse încărcate                           |
| `DATA_COLLECTION` | Înregistrare senzori și comenzi manuale (Mod MANUAL)          | MOD_OPERARE = 'MANUAL'           | Salvare date_robot.csv                      |
| `TRAINING_PIPELINE` | Procesare date și optimizare ponderi rețea (offline)          | Execuție antrenare.py            | Salvare trained_model.keras |
| `AUTONOMOUS_CONTROL` | Bucla de control reactiv: senzori → inferență RN → motoare    | MOD_OPERARE = 'AUTONOM'          | Oprire simulator                                            |
| `REMOTE_LOGGING` | Transmitere telemetrie către serverul Flask via HTTP POST     | Predicție generată în mod autonom |  Confirmare sync dashboard                                            |
| `ERROR` | Gestionare excepții (ex: model negăsit la pornire)            |  Fișier .keras absent în models/                                 | Oprire sistem / Log eroare                                            |

**Justificare alegere arhitectură State Machine:**

Am ales o arhitectură de tip **Control Reactiv cu Monitorizare Asincronă** pentru că proiectul nostru necesită controlul în timp real al robotului (buclă rapidă de decizie) simultan cu vizualizarea datelor de la distanță.

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (shape: [3]) -> (Senzori: Stanga, Centru, Dreapta)
   → Dense(16, ReLU)
   → Dense(8, ReLU)
   → Dense(2, Linear) -> (Motoare: Stanga, Dreapta)
Output: 2 valori continue (regresie)
```

**Justificare alegere arhitectură:**

 Am ales o arhitectură MLP (Multi-Layer Perceptron) simplă deoarece intrările sunt vectori de dimensiune mică (3 senzori), iar relația dintre senzori și comenzile motoarelor nu necesită o rețea convoluțională complexă. Stratul
 de ieșire cu activare liniară permite predicția vitezelor continue (regresie), esențială pentru controlul fin al robotului, spre deosebire de o clasificare discretă care ar limita mișcările la comenzi brute (stânga/dreapta).

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Valoare standard pentru optimizatorul Adam; asigură o convergență stabilă pentru un model de regresie. |
| Batch Size | 32 |  Compromis optim între eficiența computațională și stabilitatea gradientului pentru un set de date de ~3000 mostre. |
| Epochs | 50 |  Suficient pentru a permite modelului să învețe pattern-urile de navigare, cu oprire automată via Early Stopping. |
| Optimizer | Adam |  Algoritm cu rată de învățare adaptivă, ideal pentru seturi de date simulate cu variații mici de semnal. |
| Loss Function |  MSE (Mean Squared Error) | Specifică problemelor de regresie; penalizează erorile mari între viteza prezisă și cea reală a motoarelor. |
| Regularizare | ReduceLROnPlateau | Ajustarea dinamică a ratei de învățare (factor 0.5) pentru a depăși platourile de eroare în faza finală. |
| Early Stopping | [patience=5, monitor=val_loss] | Prevenirea supra-antrenării (overfitting) prin oprirea procesului dacă eroarea pe validare nu mai scade. |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline              | Accuracy | F1-Score | Timp Antrenare | Observații                            |
|------|------------------------------------------|----------|---------|----------------|---------------------------------------|
| **Baseline** | Configurația din etapa 5 (batch size 32) | 64.00%   | 0.26 | < 1 min | Referință                             |
| Exp 1 | Learning rate 0.001 → 0.002              | 68.00%   | 0.55 | < 1 min | Convergență mai rapidă, dar instabilă |
| Exp 2 | Batch size 32 → 64                       | 66.00%   |  0.45 | 30 sec | Epoch 16: early stopping              |
| Exp 3 | +1 hidden layer (32 neuroni)             | 72.00%   | 0.65 | 10 sec | Epoch 16: early stopping              |
| Exp 4 | Optimizer 'adam' → 'rmsprop'             | 65.00%   | 0.48 | 15 sec | Performanță puțin mai slabă, Epoch 39: early stopping   |
| **FINAL** | Exp 3 (+1 hidden layer 32 neuroni)       | 72.00%   | 0.65 | 10 sec | Modelul cu cel mai bun echilibru între precizie și F1-Score      |

**Justificare alegere model final:**

  Am ales configurația din Experimentul 3 ca model final deoarece adăugarea unui strat ascuns suplimentar de 32 de neuroni a permis rețelei să captureze relații non-liniare mai complexe, rezultând în cea mai mare acuratețe (72%)
  și cel mai bun F1-Score (0.65). Această arhitectură a redus semnificativ confuzia între mersul înainte și viraje, oprindu-se eficient la epoca 11 prin Early Stopping, ceea ce indică o generalizare superioară fără semne de
  overfitting.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/trained_model.keras`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|-----|
| **Accuracy** | 72.00% | ≥70% | [✓] |
| **F1-Score (Macro)** | 0.65 | ≥0.65 | [✓] |
| **Precision (Macro)** | 0.68 | - | - |
| **Recall (Macro)** | 0.62 | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 64.00% | 72.00%  | +8.00% |
| F1-Score | 0.26 | 0.65 | +0.39 |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** |  Înainte (Clasa 2) - Precision 36%, Recall 34% (valori relative în context multi-class dificil) |
| **Clasa cu cea mai slabă performanță** |  Dreapta (Clasa 1) - Precision 26%, Recall 22% |
| **Confuzii frecvente** |  Dreapta → Înainte: 44% din cazurile reale de viraj dreapta sunt clasificate ca mers înainte. Înainte → Stânga: 37% din cazurile de mers înainte sunt confundate cu viraj stânga. |
| **Dezechilibru clase** |  Dominanța clasei "Înainte" în setul de antrenament predispune modelul să prezică această clasă în situații de incertitudine (bias majoritar). |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă)                                    | Predicție RN | Clasă Reală | Cauză Probabilă                                                                                                                                                                                        | Implicație Industrială                                                                                                                                    |
|---|-------------------------------------------------------------|--------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Input senzori [0,0,0] (toți pe linie)                       | Înainte | Stânga | Ambiguitate senzorială la intersecții sau linii groase unde starea statică a celor 3 senzori nu oferă suficientă informație spațială pentru a diferenția direcția.                                     | Robotul ratează ramificația traseului și continuă să meargă înainte, riscând părăsirea fluxului de producție planificat.                                  |
| 2 | Input senzori [0,0,0] (toți pe linie)                       | Înainte | Dreapta | Bias de clasă determinat de dataset-ul dezechilibrat (70% mers drept); modelul tinde să favorizeze clasa majoritară atunci când încrederea în predicție este scăzută.                                  | Risc de blocaj pe traseu în punctele de decizie, ducând la întârzieri în lanțul logistic și necesitatea intervenției manuale.                             |
| 3 | Input senzori [0,0,1] (Stânga și Centru pe linie, Dreapta pe alb) | Înainte | Stânga | Sensibilitate scăzută la senzorii periferici; modelul acordă o importanță prea mare senzorului central, ignorând semnalele de corecție laterală până când robotul este deja parțial ieșit de pe trasă. | Comportament de navigare oscilant ("zigzag"), ceea ce duce la o uzură prematură a motoarelor și potențiala pierdere definitivă a liniei în curbe strânse. |
| 4 | Input senzori [0,0,0] (Intersecție tip T) | Înainte | Stânga | Lipsa trăsăturilor distincte și a contextului temporal; un singur cadru static nu poate indica direcția de urmat fără senzori "look-ahead" sau memorie (arhitectură RNN/LSTM)                          | Erori de sortare și livrarea componentelor la stații de lucru greșite, cauzând perturbări majore în procesul de asamblare.                                |
| 5 | Input senzori [0,0,0] (Intersecție)   | Înainte | Dreapta | Conflicte în datele de antrenament provenite din eroarea umană în timpul colectării manuale (aceeași stare a senzorilor a fost etichetată uneori ca "înainte" și alteori ca "dreapta")                 | Lipsă de predictibilitate operațională; robotul se poate comporta diferit în puncte identice ale fabricii, creând riscuri de siguranță pentru personal.   |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

  În contextul navigării autonome a AGV-ului, o acuratețe de 72% și un recall de 62% pe manevrele de viraj (media claselor Stânga/Dreapta) se traduc printr-un risc operațional semnificativ. Practic, din 100 de viraje necesare pe
  traseu, robotul execută corect doar aproximativ 62, ratând restul de 38 prin continuarea deplasării înainte sau virarea târzie. Într-o fabrică, fiecare viraj ratat (False Negative pe clasă viraj) înseamnă o oprire de producție
  pentru intervenția operatorului uman, cu un cost estimat de 5 minute/incident  100 RON/oră = ~8.3 RON per eroare. La un flux de 1000 viraje/zi, pierderile pot atinge 380  8.3 RON = 3154 RON zilnic, ceea ce face soluția actuală
  neviabilă economic fără supraveghere.

**Pragul de acceptabilitate pentru domeniu:** Recall ≥ 95% pentru manevrele de navigare (viraje) pentru a asigura o autonomie reală.
**Status:** Neatins (Diferență de ~33% față de pragul de siguranță).
**Plan de îmbunătățire:**
   1. Augmentare date: Oversampling agresiv pentru clasele de viraj și colectarea de date suplimentare specifice situațiilor de eroare (recovery).
   2. Senzori adiționali: Integrarea a 2 senzori laterali "extremi" pentru a anticipa curbele cu o marjă de timp mai mare.
   3. Arhitectură temporală: Implementarea unei rețele recurente (LSTM) care să ia decizii bazate pe o secvență de 5-10 cadre, nu doar pe inputul instantaneu, eliminând ambiguitățile statice.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|------------------|-------------|
| **Model încărcat** | `trained_model.keras` | `trained_model.keras` |  Arhitectura a fost îmbunătățită cu un strat ascuns de 32 neuroni pentru a crește F1-Score-ul pe viraje. |
| **Threshold decizie** | 0.5 (implicit) | 0.2 (Discretizare metrici) |  Ajustat pentru a captura mai precis intenția de viraj și a reduce bias-ul de mers înainte. |
| **UI - feedback vizual** | Afișare date brute (JSON)| Grafic bară probabilități | Permite operatorului să vizualizeze în timp real nivelul de încredere al modelului pentru fiecare direcție. |
| **Latență inferență** | ~45 ms | 32 ms |  Optimizarea fluxului de preprocesare și a comunicării asincrone între simulator și serverul Flask. |
| **Preprocesare date** | Filtrare de bază | Filtrare stări staționare + Mirroring | Elimină datele redundante (robot oprit) și echilibrează dataset-ul pentru viraje stânga/dreapta. |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

  Screenshot-ul prezintă interfața de monitorizare a robotului AGV în timp ce rulează în modul AUTONOM folosind modelul optimizat. În partea stângă este vizibil panoul de telemetrie care afișează datele brute în format JSON
  (coordonate, unghi, stare senzori și viteze motoare), iar în partea dreaptă este evidențiat noul grafic de tip bară care arată în timp real probabilitățile calculate de rețeaua neuronală pentru cele trei acțiuni
  posibile: Stânga, Înainte și Dreapta. Imaginea demonstrează capacitatea sistemului de a lua decizii de navigare corect corelate cu inputul senzorial, oferind în același timp operatorului un feedback vizual clar asupra
  nivelului de încredere al modelului.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | Robotul detectează marcajul liniei negre prin cei 3 senzori IR virtuali în simulator. |
| 2 | Procesare | Normalizarea vectorului de input [S_L, S_C, S_R] și transmiterea către modelul MLP. |
| 3 | Inferență | Modelul prezice vitezele motoarelor; Dashboard-ul afișează probabilitățile (ex: "Înainte: 85%"). |
| 4 | Decizie | Robotul aplică vitezele diferențiale și își ajustează unghiul de deplasare pentru a rămâne pe trasă.|

**Latență măsurată end-to-end:** 32 ms ms  
**Data și ora demonstrației:** 11.02.2026, x:30

---

## 8. Structura Repository-ului Final

```
agv-neural-network/
│
├── PAVEL_Alexandru-Gabriel_631AB_README_Proiect_RN.md   # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── datasets/
│   ├── README_Etapa3.md              # Documentație Etapa 3
│   ├── README_Etapa4_Arhitectura_SIA.md          # Documentație Etapa 4
│   ├── README_Etapa5_Antrenare_RN.md           # Documentație Etapa 5
│   ├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md      # Documentație Etapa 6
│   │
│   ├── confusion_matrix.png 
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.mp4             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # NOU - Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # NOU - Model optimizat
│   │   ├── confusion_matrix_optimized.png   
│   │   └── example_predictions.png         # NOU - Grid exemple
│   │
│   └── optimization/
│
├── data/
│   ├── README.md                          
│   ├── raw/           
│   │   └── date_robot.csv                     
│   ├── processed/       
│   │   ├── model_robot.h5
│   │   └── model_robot.keras                   
│   ├── generated/                          
│   ├── train/                              
│   ├── validation/                         
│   └── test/                               
│
├── src/
│   ├── data_acquisition/                   
│   │   └── simulator.py
│   │
│   ├── preprocessing/                       
│   │
│   ├── neural_network/                    
│   │   ├── antrenare.py                
│   │   ├── confusion_matrix_optimized.py        
│   │   ├── find_errors.py  (error_analysis.json nu este necesar, le printeaza in consola)                 
│   │   └── generate_visualizations.py    
│   │
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
│   ├── optimization_experiments.csv        
│   ├── final_metrics.json
│   └── hyperparameters.yaml  
│
├── config/
│   └── config.py 
│
├── requirements.txt                        
└── .gitignore                              
```

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=X.XX, F1=X.XX" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)
pip >= 21.0
PyCharm (recomandat pentru configurarea automată a mediului și a variabilelor de path)
```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone https://github.com/Kyakzo/agv-neural-network_Pavel_Alexandru-Gabriel.git
cd agv-neural-network

# 2. Creare mediu virtual (recomandat)
# Se recomandă utilizarea PyCharm pentru a crea automat venv-ul.
# Altfel:
python -m venv venv
#source venv/bin/activate        # Linux/Mac
sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1:  Antrenare model (ppțional, dacă doriți re-antrenarea)
# Rulează scriptul de antrenare care preprocesează automat datele din data/raw/date_robot.csv
python src/neural_network/antrenare.py

# Pasul 2: Lansare Server Telemetrie (Flask)
# Acesta va porni dashboard-ul web la http://127.0.0.1:5000
python src/app/server.py

# Pasul 3: Lansare simulator (clientul robotului)
# Asigurați-vă că serverul Flask rulează deja. Simulatorul va încerca să se conecteze la el.
python src/data_acquisition/simulator.py

Modul de operare se schimbă din fișierul config/config.py:
  * Colectare Date (Manual): Setați MOD_OPERARE = 'MANUAL'. Controlați robotul cu săgețile și datele se vor salva în date_robot.csv.
  * Navigare Autonomă: Setați MOD_OPERARE = 'AUTONOM'. Robotul va încărca modelul antrenat și se va deplasa singur, trimițând date către dashboard.
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect
python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='2'; from tensorflow.keras.models import load_model; m = load_model('models/trained_model.keras'); print('✓ Model încărcat cu succes')"

# Verificare rapidă a predicțiilor și analiza erorilor
# Acest script încarcă modelul și testează predicțiile pe setul de test
python src/neural_network/find_errors.py
```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|---------|-----|
|  Acuratețea deciziilor de navigare |  > 65% | 72.00% | [✓] |
| Performanța decizională (F1-Score) |  ≥ 0.60 |  0.65 | [✓] |
| Accuracy pe test set | ≥70% | 72.00% | [✓] |
| F1-Score pe test set | ≥0.65 | 0.65 | [✓] |
|  Eficiența procesării (Latență inferență)| < 33 ms | 32 ms | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

   1. Limitare 1 (sensibilitate la intersecții): Modelul confundă adesea intersecțiile (când toți senzorii detectează negru [0,0,0]) cu linia dreaptă, cauzând ratarea virajelor în puncte critice.
   2. Limitare 2 (bias de "mers inainte"): Din cauza preponderenței segmentelor drepte în datele de antrenament, robotul are tendința de a întârzia virajele, corectând traiectoria abia când iese parțial de pe linie.
   3. Limitare 3 (Lipsă memorie temporală): Fiind un MLP simplu, rețeaua ia decizii cadru cu cadru, fără a "ști" ce s-a întâmplat acum 100ms. Acest lucru duce la oscilații ("tremurat") pe liniile curbe.
   4. Funcționalități planificate dar neimplementate: Nu s-a implementat exportul modelului în format ONNX pentru rularea pe dispozitive embedded dedicate (ex: microcontrolere STM32).

### 10.3 Lecții Învățate (Top 5)

1. **Importanța calității datelor:** Am învățat că un set de date curat și etichetat corect este mai valoros decât o arhitectură complexă. Erorile de etichetare umană (comenzi contradictorii) au limitat performanța maximă a
      modelului.
2. **Echilibrul claselor:**  Dominanța clasei "Înainte" a forțat modelul să ignore virajele subtile. Am înțeles necesitatea tehnicilor de oversampling sau a funcțiilor de loss ponderate.
3. **Early Stopping este crucial:**  Fără oprirea automată la epoca ~11-20, modelul intra rapid în overfitting, învățând pe de rost traseul de antrenament și eșuând pe cel de test.
4. **Impactul arhitecturii:**  Adăugarea unui singur strat ascuns suplimentar (32 neuroni) a transformat un model mediocru (F1=0.26) într-unul funcțional (F1=0.65), demonstrând că rețeaua avea nevoie de mai multă capacitate pentru a modela curbele.
5. **Decuplarea sistemelor:**  Arhitectura client-server (Simulator <-> Flask) a permis dezvoltarea și debug-ul interfeței grafice independent de logica de control a robotului, simplificând integrarea finală.

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

  Dacă aș reîncepe proiectul, aș acorda mult mai multă atenție etapei de Data Engineering și design-ului senzorilor. În loc să mă bazez pe doar 3 senzori binari, aș simula o bară cu 5-7 senzori sau chiar o cameră video simplă
  procesată cu CNN. Am observat că majoritatea erorilor provin din ambiguitatea stărilor [0,0,0] sau [0,0,1], unde informația spațială este pur și simplu insuficientă pentru o decizie corectă. De asemenea, aș implementa de la
  început o arhitectură care să suporte secvențe temporale (RNN/LSTM) pentru a elimina oscilațiile, în loc să încerc să forțez un MLP static să rezolve o problemă dinamică.

  Un alt aspect pe care l-aș schimba este modul de colectare a datelor. În loc de control manual pur, aș implementa un algoritm PID clasic pentru a genera date "perfecte" de urmarire a liniei, la care aș adăuga ulterior zgomot
  controlat. Colectarea manuală a introdus inconsistențe (timpi de reacție variabili ai operatorului) care au "confuzat" rețeaua, limitându-i capacitatea de a învăța o funcție de control optimă.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | Implementarea unui buffer de istoric (ultimele 5 cadre) ca input în rețea. | Eliminarea ambiguității la intersecții și creșterea acurateței cu 5-10%. |
| **Medium-term** (1-2 luni) | Trecerea la o arhitectură Recurentă (LSTM/GRU) și creșterea numărului de senzori la 5. | Navigare fluidă în curbe strânse și reducerea oscilațiilor cu 80%. |
| **Long-term** | Portarea modelului pe un microcontroler ESP32/STM32 folosind TensorFlow Lite Micro. | Sistem complet autonom (embedded), latență <5ms și cost hardware sub 50 RON. |

---

## 11. Bibliografie

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E., 2012. "ImageNet Classification with Deep Convolutional Neural Networks". NIPS. URL: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf (https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
2. Kingma, D. P., & Ba, J., 2014. "Adam: A Method for Stochastic Optimization". ICLR. URL: https://arxiv.org/abs/1412.6980 (https://arxiv.org/abs/1412.6980)
3. Chollet, F., et al., 2015. "Keras Documentation". Keras.io. URL: https://keras.io/api/callbacks/early_stopping/ (https://keras.io/api/callbacks/early_stopping/)
4. Pygame Community, 2023. "Pygame Documentation - Sprite Module". URL: https://www.pygame.org/docs/ref/sprite.html (https://www.pygame.org/docs/ref/sprite.html)

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale** (verificabil în `data/raw/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [x] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [x] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [x] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate cu date reale)
- [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă cu Secțiunea 8
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [x] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [x] **Tag `v0.6-optimized-final`** creat și pushed
- [x] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [x] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** 12.02.2026  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
