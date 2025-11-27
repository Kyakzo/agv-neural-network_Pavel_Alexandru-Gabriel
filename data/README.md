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

| **Caracteristică** | **Tip** | **Unitate** | **Descriere**                                                  | **Domeniu valori** |
|--------------------|---------|-------------|----------------------------------------------------------------|--------------------|
| senzor_stanga      | numeric | adimensional | Intensitate semnal senzor stânga (0=negru/linie, 1=alb/fundal) | 0.0 – 1.0 |
| senzor_centru      | numeric | adimensional | Intensitate semnal senzor centru                               | 0.0 – 1.0 |
| senzor_dreapta     | numeric | adimensional | Intensitate semnal senzor dreapta                              | 0.0 – 1.0 |
| motor_stanga       | numeric | adimensional | Comanda de viteză aplicată motorului stâng                     | -1.0 – 1.0 |
| motor_dreapta      | numeric | adimensional | Comanda de viteză aplicată motorului drept                     | -1.0 – 1.0 |