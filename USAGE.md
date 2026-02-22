# Naudojimo Instrukcija


### 1. Instaliavimas

```bash
# Klonuokite projektą
git clone <repository-url>
cd iot_intrusion_detection

# Sukurkite virtualią aplinką
python -m venv venv
source venv/bin/activate  # Linux/Mac
# arba
venv\Scripts\activate     # Windows

# Įdiekite priklausomybes
pip install -r requirements.txt
```

### 2. Demo Vykdymas

Paprasčiausias būdas išbandyti sistemą:

```bash
python main.py --generate-demo
```

Ši komanda:
- Sugeneruos 5000 sintetinių IoT tinklo įrašų
- Išmokys visus 4 modelius
- Sukurs visas vizualizacijas
- Išsaugos modelius ir rezultatus

### 3. Darbas su Tikrais Duomenimis

#### BoT-IoT duomenų rinkinys

```bash


# 1. Vykdykite:
python main.py --data data/iot_network_data.csv
```
