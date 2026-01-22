# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Pavel Alexandru-Gabriel
**Link Repository GitHub: https://github.com/Kyakzo/agv-neural-network_Pavel_Alexandru-Gabriel**
**Data:** 04/12/2025 
---

[//]: # (## Scopul Etapei 4)

[//]: # ()
[//]: # (Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.)

[//]: # ()
[//]: # (**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială &#40;SIA&#41;. In acest stadiu modelul RN este doar definit și compilat &#40;fără antrenare serioasă&#41;.**)

[//]: # ()
[//]: # (### IMPORTANT - Ce înseamnă "schelet funcțional":)

[//]: # ()
[//]: # ( **CE TREBUIE SĂ FUNCȚIONEZE:**)

[//]: # (- Toate modulele pornesc fără erori)

[//]: # (- Pipeline-ul complet rulează end-to-end &#40;de la date → până la output UI&#41;)

[//]: # (- Modelul RN este definit și compilat &#40;arhitectura există&#41;)

[//]: # (- Web Service/UI primește input și returnează output)

[//]: # ()
[//]: # ( **CE NU E NECESAR ÎN ETAPA 4:**)

[//]: # (- Model RN antrenat cu performanță bună)

[//]: # (- Hiperparametri optimizați)

[//]: # (- Acuratețe mare pe test set)

[//]: # (- Web Service/UI cu funcționalități avansate)

[//]: # ()
[//]: # (**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.)

[//]: # ()
[//]: # (---)

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)

[//]: # (Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți &#40;metrici măsurabile obligatoriu&#41;:)

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|---------------------------------|--------------------------------|
| Automatizarea transportului de materiale în depozite pe trasee fixe | Robot autonom (Line Follower) cu timp de reacție la curbe < 50ms | `src/neural_network` + `simulator.py` |
| Generarea rapidă a scenariilor de testare (curbe, erori) fără risc fizic | Simulare fizică (Pygame) a robotului și senzorilor pentru colectare date | `src/data_acquisition/simulator.py` |
| Monitorizarea de la distanță a parametrilor operaționali (viteză, direcție) | Dashboard Web accesibil în rețea cu latență redusă (< 1s) | `app.py` (Web Service) |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

#### Cum se calculează 40%: generez toate datele (simulare, senzori proprii, etichetare manuală).

[//]: # (#### Declarație obligatorie în README:)

### Contribuția originală la setul de date:

**Total observații finale:** ~10,000 (după etapa 3 + etapa 4)
**Observații originale:** ~10,000 (100%)

**Tipul contribuției:** Date generate prin simulare fizică

**Descriere detaliată:**
Am dezvoltat un simulator propriu ("AGV Simulator") folosind biblioteca **Pygame** în Python. Acesta modelează fizica unui robot cu tracțiune diferențială și o bară de senzori cu 3 puncte de detecție.
Datele sunt generate prin sesiuni de "conducere manuală", unde utilizatorul controlează robotul pe trasee desenate digital. La fiecare frame al simulării, se salvează starea senzorilor (input: 0-1 în funcție de culoarea pixelilor de sub senzori) și comanda dată motoarelor (label: viteză motor stânga/dreapta).
Această metodă asigură date perfect etichetate și permite generarea de scenarii dificile (ieșiri de pe traseu, curbe strânse) esențiale pentru antrenarea rețelei.

**Locația codului:** `src/data_acquisition/simulator.py`
**Locația datelor:** `data/raw/date_robot.csv`

**Dovezi:**
- Cod funcțional simulator: `src/data_acquisition/simulator.py`
- Fișier CSV generat: `data/raw/date_robot.csv`

---

### 3. Diagrama State Machine a intregului sistem

[//]: # (**Cerințe:**)

[//]: # (- **Minimum 4-6 stări clare** cu tranziții între ele)

[//]: # (- **Formate acceptate:** PNG/SVG, pptx, draw.io )

[//]: # (- **Locație:** `docs/state_machine.*` &#40;orice extensie&#41;)

[//]: # (- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?")

**Stări tipice pentru un SIA:**
```
IDLE → ACQUIRE_DATA → PREPROCESS → INFERENCE → DISPLAY/ACT → LOG → [ERROR] → STOP
                ↑______________________________________________|
```

[//]: # (**Legendă obligatorie &#40;scrieți în README&#41;:**)

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **Control Reactiv cu Monitorizare Asincronă** pentru că proiectul nostru necesită controlul în timp real al robotului (buclă rapidă de decizie) simultan cu vizualizarea datelor de la distanță.

Stările principale sunt:
1.  **INITIALIZATION**: Încărcarea configurărilor (`config.py`) și a resurselor (imagini traseu, model neural).
2.  **DATA_COLLECTION (Manual)**: Utilizatorul conduce robotul; la fiecare ciclu se citesc senzorii și se scriu datele în CSV.
3.  **TRAINING_PIPELINE**: Stare offline în care se prelucrează CSV-ul și se antrenează modelul Keras.
4.  **AUTONOMOUS_CONTROL**: Starea principală de funcționare a SIA. Robotul citește senzorii, execută inferența RN și aplică comanda la motoare.
5.  **REMOTE_LOGGING**: În paralel cu controlul, robotul trimite starea curentă către serverul Web via HTTP POST.

Tranzițiile critice sunt:
- **Manual → Train**: Se face prin oprirea simulatorului și rularea scriptului de antrenare.
- **Train → Autonomous**: Se face prin schimbarea flag-ului `MOD_OPERARE` în config și restartarea simulatorului.
- **Autonomous → Error**: Dacă modelul `.h5` nu este găsit la pornire.

---

### 4. Scheletul complet al celor 3 module cerute la curs (slide 7)

[//]: # (Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.)
[//]: # ()
[//]: # (| **Modul** | **Python**                                      | **Cerință minimă funcțională &#40;la predare&#41;** |)

[//]: # (|-----------|-------------------------------------------------|----------------------------------------------|)

[//]: # (| **1. Data Logging / Acquisition** | `src/data_acquisition/`                         | **MUST:** Produce CSV cu datele voastre &#40;inclusiv cele 40% originale&#41;. Cod rulează fără erori și generează minimum 100 samples demonstrative. |)

[//]: # (| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună &#40;poate avea weights random/inițializați&#41;. |)

[//]: # (| **3. Web Service / UI** | Flask | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |)

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [x] Cod rulează fără erori: `python src/data_acquisition/simulator.py`
- [x] Generează CSV în format compatibil cu preprocesarea din etapa 3
- [x] Include minimum 40% date originale în dataset-ul final (100% în acest caz)
- [x] Documentație în cod: datele sunt `senzor_stanga, senzor_centru, senzor_dreapta, motor_stanga, motor_dreapta`

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [x] Arhitectură RN definită și compilată fără erori (`src/neural_network/antrenare.py`)
- [x] Model poate fi salvat (`model.save`) și reîncărcat (`load_model`)
- [x] Include justificare pentru arhitectura aleasă (Dense layers pentru clasificare/regresie simplă)
- [x] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [x] Propunere Interfață ce primește input de la user (Flask API endpoint `/api/robot/status`)
- [x] Includeți un screenshot demonstrativ în `docs/screenshots/`

[//]: # (**Ce NU e necesar în Etapa 4:**)

[//]: # (- UI frumos/profesionist cu grafică avansată)

[//]: # (- Funcționalități multiple &#40;istorice, comparații, statistici&#41;)

[//]: # (- Predicții corecte &#40;modelul e neantrenat, e normal să fie incorect&#41;)

[//]: # (- Deployment în cloud sau server de producție)

[//]: # ()
[//]: # (**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.)


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

[//]: # (**Verificare consistență cu Etapa 3:**)

```
agv-neural-network/
├── data/
│   ├── raw/               
│   ├── processed/         
│   ├── train/
│   └── README.md
├── src/
│   ├── data_acquisition/
│   │   └── simulator.py   
│   ├── neural_network/
│   │   ├── antrenare.py 
│   │   └── graph.png
│   └── app/               
├── app.py                 
├── server.py             
├── config/
│   └── config.py         
├── docs/
│   ├── state_machine.png
│   ├── screenshots/ 
│   │   └── ui_demo.png
│   └── datasets/
├── README.md             
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [x] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [x] Cod generare/achiziție date funcțional și documentat
- [x] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.png` (Trebuie generată imaginea)
- [x] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [x] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/data_acquisition/simulator.py`)
- [x] Produce minimum 40% date originale din dataset-ul final
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [x] Documentație în `src/data_acquisition/simulator.py`

### Modul 2: Neural Network
- [x] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [x] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [x] Propunere interfață ce pornește fără erori (comanda de lansare testată)
- [x] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [x] README/Instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`
