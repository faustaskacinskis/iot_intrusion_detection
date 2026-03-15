# IoT tinklo atakų aptikimas – BoT-IoT Dataset

Mašininio mokymosi projektas, skirtas automatiškai aptikti kenkėjišką IoT tinklo srautą naudojant realius **BoT-IoT** duomenis.

---

## Projekto struktūra

```
iot-attack-detection/
│
├── main.py                  # Pagrindinis paleidimo skriptas
├── data_preprocessing.py    # Duomenų įkėlimas, valymas, kodavimas, normalizavimas
├── models.py                # Modelių apibrėžimas, mokymas, vertinimas
├── visualization.py         # Grafikų generavimas
├── requirements.txt         # Python priklausomybės
├── README.md                # Šis failas
│
└── results/                 # Automatiškai sukuriamas
    ├── 01_class_distribution.png
    ├── 02_model_comparison.png
    ├── 03_confusion_matrices.png
    ├── 04_roc_curves.png
    ├── 05_feature_importance.png
    ├── 06_training_times.png
    └── summary.json
```

---

##  Įdiegimas

```bash
# 1. Klonuokite arba atsisiųskite projektą
cd iot-attack-detection

# 2. Sukurkite virtualią aplinką (rekomenduojama)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Įdiekite priklausomybes
pip install -r requirements.txt
```

---

## Paleidimas

### Greitas startas (sintetiniai duomenys)
```bash
python main.py
```

### Su realiais BoT-IoT duomenimis
```bash
# Atsisiunčiami duomenys iš: https://research.unsw.edu.au/projects/bot-iot-dataset
# Failai: UNSW_2018_IoT_Botnet_Dataset_1.csv ... _74.csv

python main.py --data UNSW_2018_IoT_Botnet_Dataset_1.csv
```

### Visos parinktys
```bash
python main.py \
  --data   kelias/iki/failo.csv \  # CSV failo kelias
  --rows   100000 \                # Eilučių skaičius (greičiui)
  --mode   binary \                # 'binary' arba 'multiclass'
  --seed   42                      # Atsitiktinumo sėkla
```

---

##  Naudojami modeliai
Random Forest
SVM
KNN
MLP Naleur Network


##  Vertinimo metrikos

- **Accuracy** – bendras tikslumas
- **Precision** – kiek teigiamų prognozių yra iš tikro teigiamos
- **Recall** – kiek tikrų atakų aptikta
- **F1-score** – harmoninis Precision ir Recall vidurkis
- **Confusion Matrix** – detali klasifikacijos vizualizacija
- **ROC / AUC** – modelio diskriminavimo geba

---

## 🔬 BoT-IoT duomenų rinkinys

**Šaltinis:** UNSW Canberra, 2018  
**Atsisiuntimas:** https://research.unsw.edu.au/projects/bot-iot-dataset  
**Dydis:** ~74 CSV failai, ~73 mln. įrašų  
**Atakų tipai:** DoS, DDoS, Reconnaissance, Data Theft, Normal

### Pagrindiniai požymiai
| Požymis | Aprašymas |
|---------|-----------|
| `pkts` | Paketų skaičius sraute |
| `bytes` | Baitų skaičius sraute |
| `dur` | Srauto trukmė (s) |
| `rate` | Paketų perdavimo greitis |
| `proto` | Protokolas (TCP/UDP/ICMP) |
| `state` | Ryšio būsena |
| `spkts/dpkts` | Siuntėjo/gavėjo paketai |
| `sbytes/dbytes` | Siuntėjo/gavėjo baitai |
| `attack` | Etiketė: 0=normalus, 1=ataka |
| `category` | Atakos kategorija |

