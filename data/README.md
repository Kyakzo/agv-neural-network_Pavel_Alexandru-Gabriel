# 📂 Documentație set de date: Control Robot Mobil (AGV)

---

## Introducere

Acest director conține datele utilizate pentru antrenarea și validarea Rețelei Neuronale (RN) în cadrul proiectului de Behavioral Cloning. Datele reprezintă înregistrări ale comportamentului unui operator uman care controlează un robot mobil simulat pe un traseu cu linii.

---

# 1. Descriere generală

Setul de date captează relația cauzală dintre percepție (citirile senzorilor de linie) și acțiune (comenzile motoarelor). Scopul este ca RN să învețe funcția de mapare $f(x) = y$, unde $x$ sunt senzorii și $y$ sunt motoarele.

* **Fișier Principal**: data/raw/date_robot.csv
* **Format**: CSV (Comma Separated Values)
* **Dimensiune tipică**: ~5,000 - 10,000 rânduri (frame-uri de simulare)
* **Frecvență eșantionare**: ~30 Hz (limitat de FPS-ul simulatorului Pygame)

# 2. Structura fișierului CSV