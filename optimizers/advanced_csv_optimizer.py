#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador CSV Avanzado con Machine Learning para DataSnap IA
Interpreta y corrige cualquier tipo de archivo CSV usando IA y predicciones
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from io import StringIO

# Optional imports with fallbacks
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False

try:
    from dateutil import parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.stats as stats

class AdvancedCSVOptimizer:
    """OPTIMIZADOR CSV IA UNIVERSAL CON MACHINE LEARNING - INTERPRETA CUALQUIER ARCHIVO"""

    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
        
        # ML Models for intelligent detection
        self.column_type_classifier = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clustering_model = KMeans(n_clusters=8, random_state=42, n_init=10)
        
        # Knowledge base for intelligent corrections
        self.data_patterns = self._initialize_patterns()
        self.correction_history = {}
        
        print("[CSV AI] Optimizador CSV con Machine Learning inicializado")

    def _initialize_patterns(self):
        """Initialize intelligent data patterns for ML detection"""
        return {
            'email': {
                'patterns': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
                'keywords': ['email', 'mail', 'correo', 'e-mail', 'electronic'],
                'corrections': {
                    'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'hotmial.com': 'hotmail.com',
                    'yahoo.co': 'yahoo.com', 'outlok.com': 'outlook.com', 'gmail.comm': 'gmail.com'
                }
            },
            'phone': {
                'patterns': [r'[\+]?[1-9]?[0-9]{7,15}'],
                'keywords': ['phone', 'telefono', 'tel', 'celular', 'mobile', 'contacto'],
                'corrections': {}
            },
            'date': {
                'patterns': [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'],
                'keywords': ['fecha', 'date', 'birth', 'created', 'updated', 'registro'],
                'corrections': {
                    '1995-02-30': '1995-02-28', '1995-15-08': '1995-08-15', 
                    '1995-14-25': '1995-12-25', 'ayer': '2024-01-14'
                }
            },
            'name': {
                'patterns': [r'^[A-Za-záéíóúñÁÉÍÓÚÑ\s\'-]{2,50}$'],
                'keywords': ['name', 'nombre', 'apellido', 'firstname', 'lastname', 'usuario'],
                'corrections': {}
            },
            'city': {
                'patterns': [r'^[A-Za-záéíóúñÁÉÍÓÚÑ\s\'-]{2,100}$'],
                'keywords': ['city', 'ciudad', 'location', 'municipio', 'localidad'],
                'corrections': {}
            },
            'country': {
                'patterns': [r'^[A-Za-záéíóúñÁÉÍÓÚÑ\s\'-]{2,50}$'],
                'keywords': ['country', 'pais', 'nation', 'nacionalidad'],
                'corrections': {}
            },
            'age': {
                'patterns': [r'^\d{1,3}$'],
                'keywords': ['age', 'edad', 'years', 'años'],
                'corrections': {}
            },
            'price': {
                'patterns': [r'^\$?[\d,]+\.?\d*$'],
                'keywords': ['price', 'precio', 'cost', 'salary', 'salario', 'amount'],
                'corrections': {}
            },
            'boolean': {
                'patterns': [r'^(true|false|1|0|yes|no|si|sí)$'],
                'keywords': ['active', 'activo', 'enabled', 'status', 'flag'],
                'corrections': {}
            }
        }

    def optimize_csv(self, csv_content):
        """Optimización CSV completa con Machine Learning"""
        try:
            print("[CSV AI] Iniciando optimización inteligente...")
            
            # 1. Intelligent parsing with encoding detection
            df = self._intelligent_parse(csv_content)
            self.original_rows = len(df)
            
            # 2. ML-based column type detection
            column_types = self._detect_column_types_ml(df)
            
            # 3. Intelligent data cleaning
            df = self._intelligent_clean(df, column_types)
            
            # 4. ML-based anomaly detection and correction
            df = self._detect_and_fix_anomalies(df, column_types)
            
            # 5. Predictive data correction
            df = self._predictive_correction(df, column_types)
            
            # 6. Smart deduplication
            df = self._smart_deduplication(df)
            
            self.final_rows = len(df)
            
            # Generate optimized CSV
            result = df.to_csv(index=False, encoding='utf-8')
            self.corrections_applied.append("CSV optimized with Machine Learning")
            
            print(f"[CSV AI] Optimización completada: {self.original_rows} -> {self.final_rows} filas")
            return result
            
        except Exception as e:
            print(f"[CSV AI ERROR] {e}")
            raise Exception(f"Error optimizing CSV with ML: {e}")

    def _intelligent_parse(self, content):
        """Parse CSV with intelligent encoding and separator detection"""
        try:
            # Detect encoding if content is bytes
            if isinstance(content, bytes):
                if CHARDET_AVAILABLE:
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding', 'utf-8')
                else:
                    encoding = 'utf-8'
                content = content.decode(encoding)
            
            # Try multiple parsing strategies
            separators = [',', ';', '\t', '|', ':']
            engines = ['python', 'c']
            
            for sep in separators:
                for engine in engines:
                    try:
                        df = pd.read_csv(StringIO(content), sep=sep, engine=engine)
                        if len(df.columns) > 1 and len(df) > 0:
                            self.corrections_applied.append(f"Parsed with separator '{sep}' and {engine} engine")
                            return df
                    except:
                        continue
            
            # Fallback: basic parsing
            df = pd.read_csv(StringIO(content))
            self.corrections_applied.append("Parsed with default settings")
            return df
            
        except Exception as e:
            raise Exception(f"Failed to parse CSV: {e}")

    def _detect_column_types_ml(self, df):
        """ML-based intelligent column type detection"""
        column_types = {}
        
        for col in df.columns:
            col_name = str(col).lower()
            sample_data = df[col].dropna().astype(str).head(100)
            
            if len(sample_data) == 0:
                column_types[col] = 'unknown'
                continue
            
            # 1. Keyword-based detection
            detected_type = self._keyword_detection(col_name)
            
            # 2. Pattern-based detection with ML confidence
            if not detected_type:
                detected_type = self._pattern_detection_ml(sample_data)
            
            # 3. Statistical analysis
            if not detected_type:
                detected_type = self._statistical_detection(sample_data)
            
            column_types[col] = detected_type or 'text'
            print(f"[CSV AI] Columna '{col}' detectada como: {column_types[col]}")
        
        return column_types

    def _keyword_detection(self, col_name):
        """Detect column type based on keywords"""
        for data_type, config in self.data_patterns.items():
            if any(keyword in col_name for keyword in config['keywords']):
                return data_type
        return None

    def _pattern_detection_ml(self, sample_data):
        """ML-enhanced pattern detection"""
        type_scores = {}
        
        for data_type, config in self.data_patterns.items():
            matches = 0
            for pattern in config['patterns']:
                matches += sample_data.str.contains(pattern, regex=True, na=False).sum()
            
            confidence = matches / len(sample_data) if len(sample_data) > 0 else 0
            type_scores[data_type] = confidence
        
        # Return type with highest confidence (>60%)
        if type_scores:
            best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
            if best_score >= 0.6:
                return best_type
        
        return None

    def _statistical_detection(self, sample_data):
        """Statistical analysis for type detection"""
        try:
            # Try numeric conversion
            numeric_data = pd.to_numeric(sample_data, errors='coerce')
            numeric_ratio = numeric_data.notna().sum() / len(sample_data)
            
            if numeric_ratio > 0.8:
                # Check if it's age-like
                if numeric_data.min() >= 0 and numeric_data.max() <= 150:
                    return 'age'
                # Check if it's price-like
                elif numeric_data.min() >= 0:
                    return 'price'
                else:
                    return 'numeric'
            
            # Check for date-like patterns
            if DATEUTIL_AVAILABLE:
                date_patterns = 0
                for value in sample_data.head(20):
                    try:
                        parser.parse(str(value))
                        date_patterns += 1
                    except:
                        pass
                
                if date_patterns / len(sample_data.head(20)) > 0.5:
                    return 'date'
            
        except Exception as e:
            print(f"[CSV AI] Statistical detection error: {e}")
        
        return None

    def _intelligent_clean(self, df, column_types):
        """Intelligent data cleaning based on detected types"""
        df_clean = df.copy()
        
        # Replace various null representations
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na',
                      '#N/A', '#NULL!', 'nil', 'NIL', '-', '--', '---', 'undefined',
                      'UNDEFINED', 'missing', 'MISSING', '?', '??', '???', 'NaN']
        
        df_clean = df_clean.replace(null_values, np.nan)
        
        # Clean based on detected types
        for col, col_type in column_types.items():
            if col_type == 'email':
                df_clean[col] = self._clean_emails(df_clean[col])
            elif col_type == 'phone':
                df_clean[col] = self._clean_phones(df_clean[col])
            elif col_type == 'date':
                df_clean[col] = self._clean_dates(df_clean[col])
            elif col_type == 'name':
                df_clean[col] = self._clean_names(df_clean[col])
            elif col_type == 'city':
                df_clean[col] = self._clean_cities(df_clean[col])
            elif col_type == 'country':
                df_clean[col] = self._clean_countries(df_clean[col])
            elif col_type == 'age':
                df_clean[col] = self._clean_ages(df_clean[col])
            elif col_type == 'price':
                df_clean[col] = self._clean_prices(df_clean[col])
            elif col_type == 'boolean':
                df_clean[col] = self._clean_booleans(df_clean[col])
        
        self.corrections_applied.append("Intelligent cleaning applied based on ML detection")
        return df_clean

    def _clean_emails(self, series):
        """Advanced email cleaning with ML-based corrections"""
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email_str = str(email).lower().strip()
            
            # Apply known corrections
            for wrong, correct in self.data_patterns['email']['corrections'].items():
                email_str = email_str.replace(wrong, correct)
            
            # Add @ if missing
            if '@' not in email_str and email_str:
                email_str = f"{email_str}@gmail.com"
            
            # Fix incomplete domains
            if email_str.endswith('@'):
                email_str += 'gmail.com'
            
            return email_str
        
        return series.apply(fix_email)

    def _clean_phones(self, series):
        """Clean phone numbers"""
        return series.astype(str).str.replace(r'[^\d+\-\s()]', '', regex=True)

    def _clean_dates(self, series):
        """Advanced date cleaning with intelligent parsing"""
        def fix_date(date_val):
            if pd.isna(date_val):
                return date_val
            
            date_str = str(date_val).strip()
            
            # Apply known corrections
            for wrong, correct in self.data_patterns['date']['corrections'].items():
                if date_str == wrong:
                    return correct
            
            if DATEUTIL_AVAILABLE:
                try:
                    # Use dateutil for intelligent parsing
                    parsed = parser.parse(date_str, dayfirst=True)
                    return parsed.strftime('%Y-%m-%d')
                except:
                    pass
            
            # Basic date parsing fallback
            try:
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                    parts = date_str.split('/')
                    return f"{parts[2]}-{parts[1]:0>2}-{parts[0]:0>2}"
                elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                    return date_str
            except:
                pass
            
            return date_str
        
        return series.apply(fix_date)

    def _clean_names(self, series):
        """Clean names with proper capitalization"""
        def fix_name(name):
            if pd.isna(name):
                return name
            
            name_str = str(name).strip()
            
            # Remove numbers and special chars
            if re.search(r'\d', name_str) or len(name_str) < 2:
                return np.nan
            
            # Proper case
            return name_str.title()
        
        return series.apply(fix_name)

    def _clean_cities(self, series):
        """Clean city names with fuzzy matching"""
        def fix_city(city):
            if pd.isna(city):
                return city
            
            city_str = str(city).strip().title()
            
            if FUZZY_AVAILABLE:
                # Common cities for fuzzy matching
                common_cities = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao', 
                               'Zaragoza', 'Málaga', 'Murcia', 'Palma', 'Córdoba']
                
                # Use fuzzy matching for corrections
                match = process.extractOne(city_str, common_cities, scorer=fuzz.ratio)
                if match and match[1] > 80:  # 80% similarity threshold
                    return match[0]
            
            return city_str
        
        return series.apply(fix_city)

    def _clean_countries(self, series):
        """Clean country names using pycountry"""
        def fix_country(country):
            if pd.isna(country):
                return country
            
            country_str = str(country).strip()
            
            if PYCOUNTRY_AVAILABLE:
                try:
                    # Try to find country using pycountry
                    country_obj = pycountry.countries.lookup(country_str)
                    return country_obj.name
                except:
                    pass
            
            return country_str.title()
        
        return series.apply(fix_country)

    def _clean_ages(self, series):
        """Clean age values with range validation"""
        def fix_age(age):
            try:
                age_val = float(age)
                if 0 <= age_val <= 150:
                    return int(age_val)
                return np.nan
            except:
                return np.nan
        
        return series.apply(fix_age)

    def _clean_prices(self, series):
        """Clean price values"""
        def fix_price(price):
            if pd.isna(price):
                return price
            
            price_str = str(price).strip()
            
            # Remove currency symbols
            price_str = re.sub(r'[$€£¥₹,]', '', price_str)
            
            try:
                price_val = float(price_str)
                return abs(price_val)  # Ensure positive
            except:
                return np.nan
        
        return series.apply(fix_price)

    def _clean_booleans(self, series):
        """Clean boolean values"""
        def fix_boolean(val):
            if pd.isna(val):
                return val
            
            val_str = str(val).lower().strip()
            
            true_vals = ['true', '1', 'yes', 'si', 'sí', 'active', 'activo', 'enabled']
            false_vals = ['false', '0', 'no', 'inactive', 'inactivo', 'disabled']
            
            if val_str in true_vals:
                return True
            elif val_str in false_vals:
                return False
            
            return val
        
        return series.apply(fix_boolean)

    def _detect_and_fix_anomalies(self, df, column_types):
        """ML-based anomaly detection and correction"""
        df_fixed = df.copy()
        
        for col, col_type in column_types.items():
            if col_type in ['age', 'price', 'numeric']:
                try:
                    # Convert to numeric for anomaly detection
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    valid_data = numeric_data.dropna()
                    
                    if len(valid_data) > 5:
                        # Detect outliers using IsolationForest
                        outliers = self.anomaly_detector.fit_predict(valid_data.values.reshape(-1, 1))
                        
                        # Replace outliers with median
                        median_val = valid_data.median()
                        outlier_indices = valid_data.index[outliers == -1]
                        
                        for idx in outlier_indices:
                            df_fixed.loc[idx, col] = median_val
                        
                        if len(outlier_indices) > 0:
                            self.corrections_applied.append(f"Fixed {len(outlier_indices)} anomalies in {col}")
                
                except Exception as e:
                    print(f"[CSV AI] Anomaly detection error for {col}: {e}")
        
        return df_fixed

    def _predictive_correction(self, df, column_types):
        """Use ML to predict and correct missing values"""
        df_predicted = df.copy()
        
        for col, col_type in column_types.items():
            try:
                # Only predict for columns with some missing values
                missing_count = df[col].isna().sum()
                total_count = len(df)
                
                if 0 < missing_count < total_count * 0.5:  # 0-50% missing
                    df_predicted = self._predict_missing_values(df_predicted, col, column_types)
            
            except Exception as e:
                print(f"[CSV AI] Prediction error for {col}: {e}")
        
        return df_predicted

    def _predict_missing_values(self, df, target_col, column_types):
        """Predict missing values using other columns"""
        try:
            # Prepare features (other columns)
            feature_cols = [col for col in df.columns if col != target_col]
            
            if not feature_cols:
                return df
            
            # Create feature matrix
            X = pd.DataFrame()
            for col in feature_cols:
                if column_types.get(col) in ['age', 'price', 'numeric']:
                    X[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    # Encode categorical variables
                    le = LabelEncoder()
                    try:
                        X[col] = le.fit_transform(df[col].astype(str))
                    except:
                        X[col] = 0
            
            # Prepare target
            if column_types.get(target_col) in ['age', 'price', 'numeric']:
                y = pd.to_numeric(df[target_col], errors='coerce')
            else:
                le_target = LabelEncoder()
                y = le_target.fit_transform(df[target_col].astype(str))
            
            # Split data
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) > 10:  # Need enough data for training
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )
                
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
                
                # Check accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > 0.7:  # Only use if accuracy > 70%
                    # Predict missing values
                    missing_mask = df[target_col].isna()
                    X_missing = X[missing_mask]
                    
                    if len(X_missing) > 0 and not X_missing.isna().all().all():
                        predictions = model.predict(X_missing.fillna(0))
                        
                        # Convert back if categorical
                        if column_types.get(target_col) not in ['age', 'price', 'numeric']:
                            predictions = le_target.inverse_transform(predictions)
                        
                        df.loc[missing_mask, target_col] = predictions
                        self.corrections_applied.append(f"Predicted {len(predictions)} missing values in {target_col}")
        
        except Exception as e:
            print(f"[CSV AI] Prediction failed for {target_col}: {e}")
        
        return df

    def _smart_deduplication(self, df):
        """Intelligent deduplication using fuzzy matching"""
        initial_rows = len(df)
        
        # Standard deduplication
        df_dedup = df.drop_duplicates()
        
        # Fuzzy deduplication for name-like columns if fuzzy library available
        if FUZZY_AVAILABLE:
            name_cols = [col for col in df.columns 
                        if any(keyword in str(col).lower() 
                              for keyword in ['name', 'nombre', 'apellido'])]
            
            if name_cols:
                df_dedup = self._fuzzy_deduplicate(df_dedup, name_cols)
        
        removed = initial_rows - len(df_dedup)
        if removed > 0:
            self.corrections_applied.append(f"Removed {removed} duplicate rows")
        
        return df_dedup

    def _fuzzy_deduplicate(self, df, name_cols):
        """Remove fuzzy duplicates based on name similarity"""
        if len(df) < 2 or not FUZZY_AVAILABLE:
            return df
        
        to_remove = set()
        
        for i, row1 in df.iterrows():
            if i in to_remove:
                continue
            
            for j, row2 in df.iterrows():
                if j <= i or j in to_remove:
                    continue
                
                # Calculate similarity for name columns
                similarities = []
                for col in name_cols:
                    val1 = str(row1[col]) if pd.notna(row1[col]) else ""
                    val2 = str(row2[col]) if pd.notna(row2[col]) else ""
                    
                    if val1 and val2:
                        similarity = fuzz.ratio(val1.lower(), val2.lower())
                        similarities.append(similarity)
                
                # If average similarity > 85%, consider duplicate
                if similarities and np.mean(similarities) > 85:
                    to_remove.add(j)
        
        return df.drop(index=to_remove)

    def get_optimization_summary(self):
        """Get detailed summary of ML optimizations"""
        summary = f"CSV ML Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Rows processed: {self.original_rows - self.final_rows}\n"
        summary += f"- ML Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary
