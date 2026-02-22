"""
Duomenų paruošimo modulis IoT atakų aptikimo sistemai.

Šis modulis atsakingas už:
- Duomenų įkėlimą
- Trūkstamų reikšmių tvarkymą
- Kategorinių kintamųjų kodavimą
- Duomenų normalizavimą
- Duomenų padalijimą mokymo/testavimo aibėms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Klasė duomenų paruošimui ir transformacijoms."""
    
    def __init__(self, dataset_type='botiot'):
        """
        Inicializuoja duomenų paruošimo objektą.
        
        Args:
            dataset_type (str): Duomenų rinkinio tipas ('botiot', 'unsw', 'cic')
        """
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, filepath, sample_size=None):
        """
        Įkelia duomenis iš CSV failo.
        
        Args:
            filepath (str): Kelias iki duomenų failo
            sample_size (int): Pavyzdžių skaičius (None = visi)
            
        Returns:
            pd.DataFrame: Įkelti duomenys
        """
        print(f"Įkeliami duomenys iš: {filepath}")
        
        try:
            # Įkeliame duomenis
            if sample_size:
                df = pd.read_csv(filepath, nrows=sample_size)
            else:
                df = pd.read_csv(filepath)
                
            print(f"✓ Sėkmingai įkelta {len(df)} įrašų")
            print(f"✓ Stulpelių skaičius: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"✗ Klaida įkeliant duomenis: {e}")
            return None
    
    def explore_data(self, df):
        """
        Atlieka pradinę duomenų analizę.
        
        Args:
            df (pd.DataFrame): Duomenų rinkinys
        """
        print("\n" + "="*60)
        print("DUOMENŲ APŽVALGA")
        print("="*60)
        
        print(f"\nForma: {df.shape}")
        print(f"\nPirmieji 5 įrašai:")
        print(df.head())
        
        print(f"\nDuomenų tipai:")
        print(df.dtypes.value_counts())
        
        print(f"\nTrūkstamų reikšmių statistika:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_percent = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Stulpelis': missing.index,
                'Trūkstama': missing.values,
                'Procentas': missing_percent.values
            })
            print(missing_df[missing_df['Trūkstama'] > 0])
        else:
            print("✓ Trūkstamų reikšmių nėra")
            
        # Tikslinės klasės pasiskirstymas
        if 'label' in df.columns:
            print(f"\nKlasių pasiskirstymas:")
            print(df['label'].value_counts())
            print(f"\nKlasių santykis:")
            print(df['label'].value_counts(normalize=True))
        elif 'attack' in df.columns:
            print(f"\nAtakų tipų pasiskirstymas:")
            print(df['attack'].value_counts())
            
    def handle_missing_values(self, df):
        """
        Tvarko trūkstamas reikšmes.
        
        Args:
            df (pd.DataFrame): Duomenų rinkinys
            
        Returns:
            pd.DataFrame: Apdoroti duomenys
        """
        print("\nTvarkomos trūkstamos reikšmės...")
        
        # Skaičių stulpeliams - užpildome medianos reikšmėmis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Kategoriniams stulpeliams - užpildome dažniausiais
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        print("✓ Trūkstamos reikšmės apdorotos")
        return df
    
    def remove_unnecessary_features(self, df):
        """
        Pašalina nereikalingus požymius.
        
        Args:
            df (pd.DataFrame): Duomenų rinkinys
            
        Returns:
            pd.DataFrame: Filtruoti duomenys
        """
        print("\nŠalinami nereikalingi požymiai...")
        
        # Bendri nereikalingi stulpeliai
        unnecessary_cols = []
        
        # ID tipo stulpeliai
        id_cols = [col for col in df.columns if col.lower() in 
                   ['id', 'index', 'flow_id', 'src_ip', 'dst_ip']]
        unnecessary_cols.extend(id_cols)
        
        # Konstantūs stulpeliai (viena unikali reikšmė)
        const_cols = [col for col in df.columns if df[col].nunique() == 1]
        unnecessary_cols.extend(const_cols)
        
        # Šaliname dublikatus iš sąrašo
        unnecessary_cols = list(set(unnecessary_cols))
        
        # Paliekame tikslinę klasę
        if 'label' in unnecessary_cols:
            unnecessary_cols.remove('label')
        if 'attack' in unnecessary_cols:
            unnecessary_cols.remove('attack')
            
        df = df.drop(columns=unnecessary_cols, errors='ignore')
        
        print(f"✓ Pašalinti {len(unnecessary_cols)} stulpeliai")
        if unnecessary_cols:
            print(f"  Pašalinti: {unnecessary_cols}")
            
        return df
    
    def encode_categorical_features(self, df):
        """
        Koduoja kategorinius požymius į skaičius.
        
        Args:
            df (pd.DataFrame): Duomenų rinkinys
            
        Returns:
            pd.DataFrame: Koduoti duomenys
        """
        print("\nKoduojami kategoriniai požymiai...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        # Nekoduojame tikslinės klasės dar
        categorical_cols = [col for col in categorical_cols 
                           if col not in ['label', 'attack', 'category']]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        print(f"✓ Užkoduoti {len(categorical_cols)} kategoriniai stulpeliai")
        
        return df
    
    def prepare_target_variable(self, df, binary_classification=True):
        """
        Paruošia tikslinę kintamąjį klasifikacijai.
        
        Args:
            df (pd.DataFrame): Duomenų rinkinys
            binary_classification (bool): Ar naudoti dvejetainę klasifikaciją
            
        Returns:
            tuple: (X, y) požymiai ir tikslas
        """
        print("\nRuošiamas tikslinė kintamasis...")
        
        # Nustatome tikslinį stulpelį
        if 'label' in df.columns:
            target_col = 'label'
        elif 'attack' in df.columns:
            target_col = 'attack'
        elif 'category' in df.columns:
            target_col = 'category'
        else:
            raise ValueError("Tikslinės klasės stulpelis nerastas!")
        
        # Dvejetainė klasifikacija: normalus vs ataka
        if binary_classification:
            # Konvertuojame į binary (0 = normal, 1 = attack)
            y = df[target_col].apply(
                lambda x: 0 if str(x).lower() in ['normal', '0', 'benign'] else 1
            )
            print("✓ Dvejetainė klasifikacija: Normal (0) vs Attack (1)")
        else:
            # Kelių klasių klasifikacija
            le = LabelEncoder()
            y = le.fit_transform(df[target_col])
            self.label_encoders['target'] = le
            print(f"✓ Kelių klasių klasifikacija: {len(le.classes_)} klasės")
            print(f"  Klasės: {le.classes_}")
        
        # Požymiai
        X = df.drop(columns=[target_col], errors='ignore')
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Požymių skaičius: {X.shape[1]}")
        print(f"✓ Pavyzdžių skaičius: {X.shape[0]}")
        
        return X, y
    
    def normalize_features(self, X_train, X_test=None):
        """
        Normalizuoja požymius (StandardScaler).
        
        Args:
            X_train (pd.DataFrame): Mokymo duomenys
            X_test (pd.DataFrame): Testavimo duomenys (optional)
            
        Returns:
            tuple: Normalizuoti X_train, X_test (jei pateikti)
        """
        print("\nNormalizuojami požymiai...")
        
        # Mokymo duomenims
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, 
                                       columns=X_train.columns,
                                       index=X_train.index)
        
        # Testavimo duomenims (jei pateikti)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled,
                                        columns=X_test.columns,
                                        index=X_test.index)
            print("✓ Normalizuoti mokymo ir testavimo duomenys")
            return X_train_scaled, X_test_scaled
        
        print("✓ Normalizuoti mokymo duomenys")
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Padalina duomenis į mokymo ir testavimo aibes.
        
        Args:
            X (pd.DataFrame): Požymiai
            y (pd.Series): Tikslas
            test_size (float): Testavimo aibės dalis
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"\nDalinama į mokymo ({1-test_size:.0%}) ir testavimo ({test_size:.0%}) aibes...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Išlaikome proporcijas
        )
        
        print(f"✓ Mokymo aibė: {X_train.shape[0]} pavyzdžiai")
        print(f"✓ Testavimo aibė: {X_test.shape[0]} pavyzdžiai")
        
        # Patikriname klasių balansą
        print("\nKlasių pasiskirstymas mokymo aibėje:")
        print(pd.Series(y_train).value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath, binary_classification=True, 
                           test_size=0.2, sample_size=None):
        """
        Pilnas duomenų paruošimo proceso vykdymas.
        
        Args:
            filepath (str): Kelias iki duomenų failo
            binary_classification (bool): Dvejetainė ar kelių klasių
            test_size (float): Testavimo aibės dalis
            sample_size (int): Pavyzdžių skaičius (None = visi)
            
        Returns:
            tuple: X_train, X_test, y_train, y_test (normalizuoti)
        """
        print("="*60)
        print("DUOMENŲ PARUOŠIMO PROCESAS")
        print("="*60)
        
        # 1. Įkeliame duomenis
        df = self.load_data(filepath, sample_size)
        if df is None:
            return None
        
        # 2. Apžvelgiame duomenis
        self.explore_data(df)
        
        # 3. Tvarkomę trūkstamas reikšmes
        df = self.handle_missing_values(df)
        
        # 4. Šaliname nereikalingus požymius
        df = self.remove_unnecessary_features(df)
        
        # 5. Koduojame kategorinius požymius
        df = self.encode_categorical_features(df)
        
        # 6. Ruošiame tikslinę kintamąjį
        X, y = self.prepare_target_variable(df, binary_classification)
        
        # 7. Dalijame duomenis
        X_train, X_test, y_train, y_test = self.split_data(
            X, y, test_size=test_size
        )
        
        # 8. Normalizuojame
        X_train, X_test = self.normalize_features(X_train, X_test)
        
        print("\n" + "="*60)
        print("DUOMENŲ PARUOŠIMAS BAIGTAS")
        print("="*60)
        
        return X_train, X_test, y_train, y_test


def main():
    """Demonstracinis pavyzdys."""
    
    # Sukuriame demo duomenis (jei nėra tikrų)
    print("Demonstracinis pavyzdys su sintetiniais duomenimis")
    
    # Generuojame sintetinius duomenis
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {
        'duration': np.random.exponential(5, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.randint(1, 1024, n_samples),
        'packets': np.random.poisson(50, n_samples),
        'bytes': np.random.exponential(1000, n_samples),
        'rate': np.random.uniform(0, 100, n_samples),
        'label': np.random.choice(['normal', 'attack'], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(demo_data)
    demo_path = '/home/claude/iot_intrusion_detection/data/demo_data.csv'
    df.to_csv(demo_path, index=False)
    
    # Vykdome paruošimo procesą
    preprocessor = DataPreprocessor(dataset_type='demo')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        demo_path,
        binary_classification=True,
        test_size=0.2,
        sample_size=None
    )
    
    print(f"\nGalutinės formos:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
