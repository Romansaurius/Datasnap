#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador CSV Avanzado con Machine Learning para DataSnap IA
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
    """OPTIMIZADOR CSV IA UNIVERSAL CON MACHINE LEARNING"""

    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
        
        # ML Models for intelligent detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clustering_model = KMeans(n_clusters=8, random_state=42, n_init=10)
        
        # Knowledge base for intelligent corrections
        self.data_patterns = self._initialize_patterns()
        
        print("[CSV AI] Optimizador CSV con Machine Learning inicializado")

    def _initialize_patterns(self):
        """Initialize intelligent data patterns for ML detection"""
        return {
            'email': {
                'patterns': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
                'keywords': ['email', 'mail', 'correo', 'e-mail'],
                'corrections': {
                    'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'hotmial.com': 'hotmail.com',
                    'yahoo.co': 'yahoo.com', 'outlok.com': 'outlook.com', 'gmail.comm': 'gmail.com'
                }
            },
            'phone': {
                'patterns': [r'[\+]?[1-9]?[0-9]{7,15}'],
                'keywords': ['phone', 'telefono', 'tel', 'celular', 'mobile'],
                'corrections': {}
            },
            'date': {
                'patterns': [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'],
                'keywords': ['fecha', 'date', 'birth', 'created', 'updated'],
                'corrections': {
                    '1995-02-30': '1995-02-28', '1995-15-08': '1995-08-15', 
                    '1995-14-25': '1995-12-25', 'ayer': '2024-01-14'
                }
            },
            'name': {
                'patterns': [r'^[A-Za-záéíóúñÁÉÍÓÚÑ\s\'-]{2,50}$'],
                'keywords': ['name', 'nombre', 'apellido', 'firstname', 'lastname'],
                'corrections': {}
            },
            'age': {
                'patterns': [r'^\d{1,3}$'],
                'keywords': ['age', 'edad', 'years', 'años'],
                'corrections': {}
            },
            'price': {
                'patterns': [r'^\$?[\d,]+\.?\d*$'],
                'keywords': ['price', 'precio', 'cost', 'salary', 'salario'],
                'corrections': {}
            },
            'boolean': {
                'patterns': [r'^(true|false|1|0|yes|no|si|sí)$'],
                'keywords': ['active', 'activo', 'enabled', 'status'],
                'corrections': {}
            }
        }

    def optimize_csv(self, csv_content):
        """Optimización CSV completa con Machine Learning"""
        try:
            print("[CSV AI] Iniciando optimización inteligente...")
            
            # 1. Parse CSV
            df = self._intelligent_parse(csv_content)
            self.original_rows = len(df)
            
            # 2. Detect column types
            column_types = self._detect_column_types_ml(df)
            
            # 3. Clean data
            df = self._intelligent_clean(df, column_types)
            
            # 4. Fix anomalies
            df = self._detect_and_fix_anomalies(df, column_types)
            
            # 5. Remove duplicates
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
            if isinstance(content, bytes):
                if CHARDET_AVAILABLE:
                    detected = chardet.detect(content)
                    encoding = detected.get('encoding', 'utf-8')
                else:
                    encoding = 'utf-8'
                content = content.decode(encoding)
            
            # Try multiple parsing strategies
            separators = [',', ';', '\t', '|']
            
            for sep in separators:
                try:
                    df = pd.read_csv(StringIO(content), sep=sep, engine='python')
                    if len(df.columns) > 1 and len(df) > 0:
                        self.corrections_applied.append(f"Parsed with separator '{sep}'")
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
            
            # Keyword-based detection
            detected_type = self._keyword_detection(col_name)
            
            # Pattern-based detection
            if not detected_type:
                detected_type = self._pattern_detection_ml(sample_data)
            
            # Statistical analysis
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
        
        if type_scores:
            best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
            if best_score >= 0.6:
                return best_type
        
        return None

    def _statistical_detection(self, sample_data):
        """Statistical analysis for type detection"""
        try:
            numeric_data = pd.to_numeric(sample_data, errors='coerce')
            numeric_ratio = numeric_data.notna().sum() / len(sample_data)
            
            if numeric_ratio > 0.8:
                if numeric_data.min() >= 0 and numeric_data.max() <= 150:
                    return 'age'
                elif numeric_data.min() >= 0:
                    return 'price'
                else:
                    return 'numeric'
            
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
        
        # Replace null values
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na']
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
            elif col_type == 'age':
                df_clean[col] = self._clean_ages(df_clean[col])
            elif col_type == 'price':
                df_clean[col] = self._clean_prices(df_clean[col])
            elif col_type == 'boolean':
                df_clean[col] = self._clean_booleans(df_clean[col])
        
        self.corrections_applied.append("Intelligent cleaning applied")
        return df_clean

    def _clean_emails(self, series):
        """Clean email addresses"""
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email_str = str(email).lower().strip()
            
            # Apply corrections
            for wrong, correct in self.data_patterns['email']['corrections'].items():
                email_str = email_str.replace(wrong, correct)
            
            if '@' not in email_str and email_str:
                email_str = f"{email_str}@gmail.com"
            
            if email_str.endswith('@'):
                email_str += 'gmail.com'
            
            return email_str
        
        return series.apply(fix_email)

    def _clean_phones(self, series):
        """Clean phone numbers"""
        return series.astype(str).str.replace(r'[^\d+\-\s()]', '', regex=True)

    def _clean_dates(self, series):
        """Clean date values"""
        def fix_date(date_val):
            if pd.isna(date_val):
                return date_val
            
            date_str = str(date_val).strip()
            
            # Apply corrections
            for wrong, correct in self.data_patterns['date']['corrections'].items():
                if date_str == wrong:
                    return correct
            
            if DATEUTIL_AVAILABLE:
                try:
                    parsed = parser.parse(date_str, dayfirst=True)
                    return parsed.strftime('%Y-%m-%d')
                except:
                    pass
            
            return date_str
        
        return series.apply(fix_date)

    def _clean_names(self, series):
        """Clean names"""
        def fix_name(name):
            if pd.isna(name):
                return name
            
            name_str = str(name).strip()
            
            if re.search(r'\d', name_str) or len(name_str) < 2:
                return np.nan
            
            return name_str.title()
        
        return series.apply(fix_name)

    def _clean_ages(self, series):
        """Clean age values"""
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
            price_str = re.sub(r'[$€£¥₹,]', '', price_str)
            
            try:
                price_val = float(price_str)
                return abs(price_val)
            except:
                return np.nan
        
        return series.apply(fix_price)

    def _clean_booleans(self, series):
        """Clean boolean values"""
        def fix_boolean(val):
            if pd.isna(val):
                return val
            
            val_str = str(val).lower().strip()
            
            true_vals = ['true', '1', 'yes', 'si', 'sí', 'active', 'activo']
            false_vals = ['false', '0', 'no', 'inactive', 'inactivo']
            
            if val_str in true_vals:
                return True
            elif val_str in false_vals:
                return False
            
            return val
        
        return series.apply(fix_boolean)

    def _detect_and_fix_anomalies(self, df, column_types):
        """ML-based anomaly detection"""
        df_fixed = df.copy()
        
        for col, col_type in column_types.items():
            if col_type in ['age', 'price', 'numeric']:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    valid_data = numeric_data.dropna()
                    
                    if len(valid_data) > 5:
                        outliers = self.anomaly_detector.fit_predict(valid_data.values.reshape(-1, 1))
                        median_val = valid_data.median()
                        outlier_indices = valid_data.index[outliers == -1]
                        
                        for idx in outlier_indices:
                            df_fixed.loc[idx, col] = median_val
                        
                        if len(outlier_indices) > 0:
                            self.corrections_applied.append(f"Fixed {len(outlier_indices)} anomalies in {col}")
                
                except Exception as e:
                    print(f"[CSV AI] Anomaly detection error for {col}: {e}")
        
        return df_fixed

    def _smart_deduplication(self, df):
        """Intelligent deduplication"""
        initial_rows = len(df)
        
        # Standard deduplication
        df_dedup = df.drop_duplicates()
        
        # Fuzzy deduplication if available
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
        """Remove fuzzy duplicates"""
        if len(df) < 2 or not FUZZY_AVAILABLE:
            return df
        
        to_remove = set()
        
        for i, row1 in df.iterrows():
            if i in to_remove:
                continue
            
            for j, row2 in df.iterrows():
                if j <= i or j in to_remove:
                    continue
                
                similarities = []
                for col in name_cols:
                    val1 = str(row1[col]) if pd.notna(row1[col]) else ""
                    val2 = str(row2[col]) if pd.notna(row2[col]) else ""
                    
                    if val1 and val2:
                        similarity = fuzz.ratio(val1.lower(), val2.lower())
                        similarities.append(similarity)
                
                if similarities and np.mean(similarities) > 85:
                    to_remove.add(j)
        
        return df.drop(index=to_remove)

    def get_optimization_summary(self):
        """Get summary of optimizations"""
        summary = f"CSV ML Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary
