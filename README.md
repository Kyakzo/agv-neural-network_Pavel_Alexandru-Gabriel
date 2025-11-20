# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pavel Alexandru-Gabriel
**Data:** 20.11.2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
agv-neural-network/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   └── train/             # set de instruire
├── src/
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date generate sintetic folosind un simulator propriu dezvoltat în Python cu libraria Pygame
* **Modul de achiziție:** Simulare
* **Perioada / condițiile colectării:** Sesiune de colectare realizată pe trasee cu linii curbe (stânga/dreapta) și linii drepte, simulând condiții ideale de contrast.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** ~10,000 frame-uri (eșantioane temporale)
* **Număr de caracteristici (features):** 3 intrări (senzori) + 2 ieșiri (motoare)
* **Tipuri de date:** Numerice (Float)
* **Format fișiere:** CSV (date_robot.csv)

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere**                                                  | **Domeniu valori** |
|--------------------|---------|-------------|----------------------------------------------------------------|--------------------|
| senzor_stanga      | numeric | adimensional | Intensitate semnal senzor stânga (0=negru/linie, 1=alb/fundal) | 0.0 – 1.0 |
| senzor_centru      | numeric | adimensional | Intensitate semnal senzor centru                               | 0.0 – 1.0 |
| senzor_dreapta     | numeric | adimensional | Intensitate semnal senzor dreapta                              | 0.0 – 1.0 |
| motor_stanga       | numeric | adimensional | Comanda de viteză aplicată motorului stâng                     | -1.0 – 1.0 |
| motor_dreapta      | numeric | adimensional | Comanda de viteză aplicată motorului drept                     | -1.0 – 1.0 |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate
* **Distribuția Comenzilor**: S-au generat histograme pentru valorile motor_stanga și motor_dreapta pentru a vedea ponderea mersului drept vs. viraje.
* **Corelații**: Matricea de corelație între senzori și motoare (ex: senzor_stanga activat ar trebui să fie corelat negativ cu motor_stanga pentru viraj).

### 3.2 Analiza calității datelor

* **Valori lipsă**: Nu există valori lipsă (NaN) deoarece simulatorul generează date complete la fiecare frame.
* **Date staționare**: S-au identificat numeroase rânduri unde motor_stanga=0 și motor_dreapta=0 (robot oprit). Aceste date nu sunt utile pentru învățarea navigării.

### 3.3 Probleme identificate

* **Dezechilibru de clasă (Bias de mers drept)**: Majoritatea datelor (aprox. 70%) reprezintă mersul în linie dreaptă. Acest lucru poate face ca rețeaua să învețe să meargă doar înainte.
* **Zgomot redus**: Fiind o simulare simplă, datele sunt "prea curate", ceea ce poate duce la o generalizare slabă dacă s-ar adăuga zgomot ulterior.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminarea stărilor staționare:** S-au șters din setul de date rândurile unde ambele comenzi de motor sunt 0 (robotul nu se mișcă).
* **Tratarea outlierilor:** Nu a fost necesară (datele sunt limitate prin cod la intervalul [-1, 1]).

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Datele senzorilor sunt deja normalizate de simulator în intervalul [0, 1]. Comenzile motoarelor sunt de asemenea în intervalul [-1, 1], compatibil cu funcția de activare **tanh**.
* **Augmentare:** Pentru a combate bias-ul de mers drept, se poate folosi "Oglindirea" (Mirroring) datelor: inversarea senzorilor stânga/dreapta și a comenzilor motoarelor pentru a dubla datele de viraj.

### 4.3 Structurarea seturilor de date

**Împărțire:**
* 80% – train: Utilizat pentru optimizarea ponderilor RN
* 20% – validation: Utilizat pentru monitorizarea erorii (Loss) și prevenirea overfitting-ului în timpul antrenării.
* Nota: Testarea finală se face prin rularea simulatorului în mod AUTONOM.

**Principii respectate:**
* `random_state=42` pentru reproductibilitatea împărțirii datelor.
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi
* Shuffle (amestecare) înainte de split pentru a rupe corelația temporală a frame-urilor consecutive.

### 4.4 Salvarea rezultatelor preprocesării

[//]: # (* Date preprocesate în `data/processed/`)
[//]: # (* Seturi train/val/test în foldere dedicate)
* Modelul RN nu necesită salvarea intermediară a fișierelor CSV preprocesate, preprocesarea (filtrarea zerourilor) se face "on-the-fly" în scriptul antrenare.py înainte de a intra în rețea.
* Modelul antrenat final este salvat ca `model_robot.h5`.

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/` – set final
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train generate
- [ ] Documentație actualizată în README + `data/README.md`

---
