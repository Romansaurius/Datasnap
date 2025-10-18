"""
🧹 ADVANCED CLEANING UTILITIES 🧹
Utilidades avanzadas de limpieza con IA
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

class AdvancedCleaner:
    """Limpiador avanzado con IA"""
    
    def __init__(self):
        # Patrones de detección de anomalías
        self.anomaly_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9][\d]{0,15}$',
            'price': r'^\d+(\.\d{1,2})?$',
            'name': r'^[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]{2,50}$'
        }
    
    def deep_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza profunda con IA"""
        
        # 1. Detectar y corregir encoding
        df = self._fix_encoding_issues(df)
        
        # 2. Normalizar espacios y caracteres especiales
        df = self._normalize_text(df)
        
        # 3. Detectar y corregir anomalías
        df = self._detect_and_fix_anomalies(df)
        
        # 4. Optimizar tipos de datos
        df = self._optimize_data_types(df)
        
        return df
    
    def _fix_encoding_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige problemas de encoding"""
        
        for col in df.select_dtypes(include=['object']).columns:
            # Corregir caracteres mal codificados
            df[col] = df[col].astype(str).str.replace('Ã¡', 'á')
            df[col] = df[col].str.replace('Ã©', 'é')
            df[col] = df[col].str.replace('Ã­', 'í')
            df[col] = df[col].str.replace('Ã³', 'ó')
            df[col] = df[col].str.replace('Ãº', 'ú')
            df[col] = df[col].str.replace('Ã±', 'ñ')
        
        return df
    
    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza texto"""
        
        for col in df.select_dtypes(include=['object']).columns:
            # Normalizar espacios
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remover caracteres de control
            df[col] = df[col].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
        
        return df
    
    def _detect_and_fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y corrige anomalías"""
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Detectar tipo de columna y aplicar validación
            if 'email' in col_lower:
                df[col] = self._fix_email_anomalies(df[col])
            elif 'phone' in col_lower or 'telefono' in col_lower:
                df[col] = self._fix_phone_anomalies(df[col])
            elif 'precio' in col_lower or 'price' in col_lower:
                df[col] = self._fix_price_anomalies(df[col])
            elif 'nombre' in col_lower or 'name' in col_lower:
                df[col] = self._fix_name_anomalies(df[col])
        
        return df
    
    def _fix_email_anomalies(self, series: pd.Series) -> pd.Series:
        """Corrige anomalías en emails"""
        
        def fix_email(email):
            if pd.isna(email) or email == 'nan':
                return email
            
            email = str(email).lower().strip()
            
            # Validar con regex
            if not re.match(self.anomaly_patterns['email'], email):
                # Intentar corregir
                if '@' not in email:
                    email += '@gmail.com'
                elif not email.split('@')[1]:
                    email = email.split('@')[0] + '@gmail.com'
            
            return email
        
        return series.apply(fix_email)
    
    def _fix_phone_anomalies(self, series: pd.Series) -> pd.Series:
        """Corrige anomalías en teléfonos"""
        
        def fix_phone(phone):
            if pd.isna(phone) or phone == 'nan':
                return phone
            
            phone = str(phone).strip()
            # Remover caracteres no numéricos excepto +
            phone = re.sub(r'[^\d\+]', '', phone)
            
            # Agregar código de país si falta
            if phone and not phone.startswith('+') and len(phone) == 9:
                phone = '+34' + phone
            
            return phone
        
        return series.apply(fix_phone)
    
    def _fix_price_anomalies(self, series: pd.Series) -> pd.Series:
        """Corrige anomalías en precios"""
        
        def fix_price(price):
            if pd.isna(price):
                return price
            
            # Convertir a string y limpiar
            price_str = str(price).strip()
            
            # Remover caracteres no numéricos excepto punto
            price_clean = re.sub(r'[^\d\.]', '', price_str)
            
            try:
                return float(price_clean)
            except:
                return np.nan
        
        return series.apply(fix_price)
    
    def _fix_name_anomalies(self, series: pd.Series) -> pd.Series:
        """Corrige anomalías en nombres"""
        
        def fix_name(name):
            if pd.isna(name) or name == 'nan':
                return name
            
            name = str(name).strip().title()
            
            # Validar longitud
            if len(name) < 2:
                return 'Usuario'
            elif len(name) > 50:
                return name[:50]
            
            return name
        
        return series.apply(fix_name)
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza tipos de datos"""
        
        for col in df.columns:
            # Intentar convertir a numérico si es posible
            if df[col].dtype == 'object':
                numeric = pd.to_numeric(df[col], errors='coerce')
                if not numeric.isna().all():
                    df[col] = numeric
        
        return df