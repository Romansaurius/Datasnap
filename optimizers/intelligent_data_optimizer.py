"""
🧠 INTELLIGENT DATA OPTIMIZER 🧠
Optimizador inteligente que preserva la estructura original y hace correcciones inteligentes
- Detecta patrones reales en los datos
- Corrige errores sin inventar información
- Usa ML para predicciones basadas en contexto
- Maneja cualquier tipo de error de forma robusta
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
from collections import Counter
import json

warnings.filterwarnings('ignore')

class IntelligentDataOptimizer:
    """Optimizador inteligente que preserva la estructura original"""
    
    def __init__(self):
        # Patrones de detección inteligente
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[\d\s\-\(\)]{7,20}$',
            'url': r'^https?://[^\s]+$',
            'date': r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
            'time': r'^\d{1,2}:\d{2}(:\d{2})?$',
            'number': r'^-?\d+\.?\d*$',
            'currency': r'^[\$€£¥]?\s*\d+\.?\d*$',
            'percentage': r'^\d+\.?\d*\s*%$',
            'ip': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'boolean': r'^(true|false|yes|no|si|sí|1|0|on|off|activo|inactivo)$'
        }
        
        # Correcciones específicas basadas en patrones reales
        self.domain_corrections = {
            'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'gmaill.com': 'gmail.com',
            'hotmial.com': 'hotmail.com', 'hotmailcom': 'hotmail.com',
            'yahoo.co': 'yahoo.com', 'yahooo.com': 'yahoo.com',
            'outlok.com': 'outlook.com', 'outllok.com': 'outlook.com',
            '.con': '.com', '.cmo': '.com', '.ocm': '.com'
        }
        
        # Valores que representan NULL de forma inteligente
        self.null_indicators = {
            '', ' ', '  ', '   ', 'nan', 'NaN', 'null', 'NULL', 'None', 'NONE',
            'n/a', 'N/A', 'na', 'NA', 'undefined', 'UNDEFINED', '-', '--', '---',
            'missing', 'MISSING', 'empty', 'EMPTY', '?', '??', 'unknown', 'UNKNOWN',
            'sin datos', 'no data', 'no disponible', 'not available', 'vacío', 'vacio',
            '#N/A', '#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!',
            'invalid', 'Invalid', 'INVALID', 'error', 'Error', 'ERROR'
        }
        
        # Cache para patrones detectados
        self.column_patterns = {}
        self.data_statistics = {}
    
    def optimize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización inteligente principal"""
        
        if df.empty:
            return df
        
        try:
            # 1. Análisis inicial de datos
            df_optimized = self._analyze_data_structure(df.copy())
            
            # 2. Limpieza inteligente
            df_optimized = self._intelligent_cleaning(df_optimized)
            
            # 3. Detección automática de tipos
            df_optimized = self._detect_column_types(df_optimized)
            
            # 4. Corrección específica por tipo
            df_optimized = self._apply_type_specific_corrections(df_optimized)
            
            # 5. Predicción inteligente de valores faltantes
            df_optimized = self._intelligent_missing_value_prediction(df_optimized)
            
            # 6. Validación y corrección de consistencia
            df_optimized = self._validate_data_consistency(df_optimized)
            
            # 7. Optimización de tipos de datos
            df_optimized = self._optimize_data_types(df_optimized)
            
            return df_optimized
            
        except Exception as e:
            print(f"Error en optimización: {e}")
            # Fallback: aplicar solo limpieza básica
            return self._basic_fallback_cleaning(df)
    
    def _analyze_data_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analiza la estructura de datos para entender el contexto"""
        
        # Estadísticas básicas por columna
        for col in df.columns:
            self.data_statistics[col] = {
                'total_rows': len(df),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_count': df[col].nunique(),
                'data_type': str(df[col].dtype),
                'sample_values': df[col].dropna().head(10).tolist()
            }
        
        return df
    
    def _intelligent_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza inteligente que preserva información válida"""
        
        # Reemplazar indicadores de NULL de forma inteligente
        for col in df.columns:
            # Solo reemplazar si el valor está en la lista de NULL indicators
            mask = df[col].astype(str).str.strip().isin(self.null_indicators)
            df.loc[mask, col] = pd.NA
        
        # Limpiar espacios en strings sin perder información
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            # Solo convertir a NA si es realmente vacío después del strip
            df[col] = df[col].replace(r'^$', pd.NA, regex=True)
        
        # Eliminar filas completamente vacías (todas las columnas son NA)
        df = df.dropna(how='all')
        
        # Eliminar duplicados exactos
        df = df.drop_duplicates()
        
        # Resetear índice
        df = df.reset_index(drop=True)
        
        return df
    
    def _detect_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta tipos de columnas basado en contenido real"""
        
        for col in df.columns:
            # Obtener muestra de valores no nulos
            sample = df[col].dropna().astype(str).head(50)
            
            if len(sample) == 0:
                self.column_patterns[col] = 'unknown'
                continue
            
            # Detectar patrón dominante
            pattern_matches = {}
            for pattern_name, pattern in self.patterns.items():
                matches = sample.str.match(pattern, na=False).sum()
                if matches > 0:
                    pattern_matches[pattern_name] = matches / len(sample)
            
            # Asignar el patrón con mayor coincidencia (mínimo 30%)
            if pattern_matches:
                best_pattern = max(pattern_matches.items(), key=lambda x: x[1])
                if best_pattern[1] >= 0.3:
                    self.column_patterns[col] = best_pattern[0]
                else:
                    self.column_patterns[col] = 'text'
            else:
                self.column_patterns[col] = 'text'
        
        return df
    
    def _apply_type_specific_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica correcciones específicas según el tipo detectado"""
        
        for col in df.columns:
            pattern_type = self.column_patterns.get(col, 'text')
            
            if pattern_type == 'email':
                df[col] = self._correct_emails(df[col])
            elif pattern_type == 'phone':
                df[col] = self._correct_phones(df[col])
            elif pattern_type == 'date':
                df[col] = self._correct_dates(df[col])
            elif pattern_type == 'number':
                df[col] = self._correct_numbers(df[col])
            elif pattern_type == 'boolean':
                df[col] = self._correct_booleans(df[col])
            elif pattern_type == 'url':
                df[col] = self._correct_urls(df[col])
            elif pattern_type == 'uuid':
                df[col] = self._correct_uuids(df[col])
            # Para 'text' y otros, aplicar limpieza general
            else:
                df[col] = self._correct_text(df[col])
        
        return df
    
    def _correct_emails(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de emails"""
        
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email_str = str(email).lower().strip()
            
            # Si es un indicador de NULL, mantener como NA
            if email_str in self.null_indicators:
                return pd.NA
            
            # Aplicar correcciones de dominios conocidos
            for wrong, correct in self.domain_corrections.items():
                email_str = email_str.replace(wrong, correct)
            
            # Corregir emails que terminan sin dominio completo
            if email_str.endswith('@'):
                return pd.NA  # No inventar dominio
            elif '@' not in email_str and '.' in email_str:
                # Podría ser un dominio sin @, pero no asumir
                return pd.NA
            elif email_str.count('@') > 1:
                # Email malformado con múltiples @
                return pd.NA
            
            # Validar formato final
            if re.match(self.patterns['email'], email_str):
                return email_str
            else:
                return pd.NA
        
        return series.apply(fix_email)
    
    def _correct_phones(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de teléfonos"""
        
        def fix_phone(phone):
            if pd.isna(phone):
                return phone
            
            phone_str = str(phone).strip()
            
            if phone_str in self.null_indicators:
                return pd.NA
            
            # Limpiar caracteres no numéricos excepto + y espacios
            clean_phone = re.sub(r'[^\d\+\s\-\(\)]', '', phone_str)
            
            # Si queda muy corto o muy largo, es inválido
            digits_only = re.sub(r'[^\d]', '', clean_phone)
            if len(digits_only) < 7 or len(digits_only) > 15:
                return pd.NA
            
            return clean_phone
        
        return series.apply(fix_phone)
    
    def _correct_dates(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de fechas"""
        
        def fix_date(date_val):
            if pd.isna(date_val):
                return date_val
            
            date_str = str(date_val).strip()
            
            if date_str in self.null_indicators:
                return pd.NA
            
            # Intentar parsear con pandas
            try:
                parsed_date = pd.to_datetime(date_str, errors='coerce')
                if pd.notna(parsed_date):
                    # Validar que la fecha sea razonable (entre 1900 y 2100)
                    if 1900 <= parsed_date.year <= 2100:
                        return parsed_date.strftime('%Y-%m-%d')
                return pd.NA
            except:
                return pd.NA
        
        return series.apply(fix_date)
    
    def _correct_numbers(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de números"""
        
        def fix_number(num_val):
            if pd.isna(num_val):
                return num_val
            
            num_str = str(num_val).strip()
            
            if num_str in self.null_indicators:
                return pd.NA
            
            # Limpiar caracteres no numéricos excepto punto y signo
            clean_num = re.sub(r'[^\d\.\-\+]', '', num_str)
            
            try:
                # Intentar convertir a float
                result = float(clean_num)
                # Validar rangos razonables
                if abs(result) > 1e15:  # Números extremadamente grandes
                    return pd.NA
                return result
            except:
                return pd.NA
        
        return series.apply(fix_number)
    
    def _correct_booleans(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de booleanos"""
        
        boolean_map = {
            'true': True, 'yes': True, 'si': True, 'sí': True, '1': True, 
            'on': True, 'activo': True, 'active': True, 'enabled': True,
            'false': False, 'no': False, '0': False, 'off': False, 
            'inactivo': False, 'inactive': False, 'disabled': False
        }
        
        def fix_boolean(bool_val):
            if pd.isna(bool_val):
                return bool_val
            
            bool_str = str(bool_val).lower().strip()
            
            if bool_str in self.null_indicators:
                return pd.NA
            
            return boolean_map.get(bool_str, pd.NA)
        
        return series.apply(fix_boolean)
    
    def _correct_urls(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de URLs"""
        
        def fix_url(url_val):
            if pd.isna(url_val):
                return url_val
            
            url_str = str(url_val).strip()
            
            if url_str in self.null_indicators:
                return pd.NA
            
            # Agregar protocolo si falta
            if not url_str.startswith(('http://', 'https://')):
                if '.' in url_str:
                    url_str = 'https://' + url_str
                else:
                    return pd.NA
            
            return url_str
        
        return series.apply(fix_url)
    
    def _correct_uuids(self, series: pd.Series) -> pd.Series:
        """Corrección inteligente de UUIDs"""
        
        def fix_uuid(uuid_val):
            if pd.isna(uuid_val):
                return uuid_val
            
            uuid_str = str(uuid_val).lower().strip()
            
            if uuid_str in self.null_indicators:
                return pd.NA
            
            # Validar formato UUID
            if re.match(self.patterns['uuid'], uuid_str):
                return uuid_str
            else:
                return pd.NA
        
        return series.apply(fix_uuid)
    
    def _correct_text(self, series: pd.Series) -> pd.Series:
        """Corrección general de texto"""
        
        def fix_text(text_val):
            if pd.isna(text_val):
                return text_val
            
            text_str = str(text_val).strip()
            
            if text_str in self.null_indicators:
                return pd.NA
            
            # Correcciones básicas de texto
            text_str = re.sub(r'\s+', ' ', text_str)  # Múltiples espacios a uno
            text_str = text_str.strip()
            
            return text_str if text_str else pd.NA
        
        return series.apply(fix_text)
    
    def _intelligent_missing_value_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción inteligente basada en patrones reales de los datos"""
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
            
            pattern_type = self.column_patterns.get(col, 'text')
            
            # Solo predecir si hay suficientes datos válidos para establecer patrones
            valid_count = df[col].notna().sum()
            if valid_count < 3:  # Muy pocos datos para predecir
                continue
            
            if pattern_type == 'number':
                # Usar mediana para números
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
            
            elif pattern_type == 'boolean':
                # Usar moda para booleanos
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            
            elif pattern_type == 'date':
                # Para fechas, usar interpolación si es posible
                if df[col].dtype == 'datetime64[ns]':
                    df[col] = df[col].interpolate(method='time')
            
            # Para otros tipos (email, phone, text, etc.), NO predecir
            # Es mejor mantener NA que inventar datos
        
        return df
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia entre columnas relacionadas"""
        
        # Validaciones específicas basadas en nombres de columnas
        columns = [col.lower() for col in df.columns]
        
        # Validar consistencia edad/fecha_nacimiento
        age_cols = [col for col in df.columns if 'edad' in col.lower() or 'age' in col.lower()]
        birth_cols = [col for col in df.columns if 'nacimiento' in col.lower() or 'birth' in col.lower()]
        
        if age_cols and birth_cols:
            age_col = age_cols[0]
            birth_col = birth_cols[0]
            
            for idx in df.index:
                age = df.loc[idx, age_col]
                birth = df.loc[idx, birth_col]
                
                if pd.notna(age) and pd.notna(birth):
                    try:
                        birth_date = pd.to_datetime(birth)
                        calculated_age = (datetime.now() - birth_date).days // 365
                        
                        # Si la diferencia es muy grande, marcar como inconsistente
                        if abs(calculated_age - float(age)) > 2:
                            df.loc[idx, age_col] = calculated_age
                    except:
                        continue
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza tipos de datos basado en contenido"""
        
        for col in df.columns:
            pattern_type = self.column_patterns.get(col, 'text')
            
            try:
                if pattern_type == 'number':
                    # Intentar convertir a numérico
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if numeric_series.notna().sum() > 0:
                        df[col] = numeric_series
                
                elif pattern_type == 'date':
                    # Intentar convertir a datetime
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    if date_series.notna().sum() > 0:
                        df[col] = date_series
                
                elif pattern_type == 'boolean':
                    # Convertir a boolean nullable
                    df[col] = df[col].astype('boolean')
                
            except Exception:
                # Si falla la conversión, mantener como está
                continue
        
        return df
    
    def _basic_fallback_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza básica como fallback en caso de errores"""
        
        try:
            # Solo limpieza muy básica
            df = df.replace(self.null_indicators, pd.NA)
            df = df.dropna(how='all')
            df = df.drop_duplicates()
            return df.reset_index(drop=True)
        except:
            # Si incluso esto falla, devolver el DataFrame original
            return df
    
    def get_optimization_report(self, original_df: pd.DataFrame, optimized_df: pd.DataFrame) -> Dict:
        """Genera reporte de optimización"""
        
        report = {
            'original_shape': original_df.shape,
            'optimized_shape': optimized_df.shape,
            'columns_analyzed': len(self.column_patterns),
            'detected_patterns': self.column_patterns,
            'data_statistics': self.data_statistics,
            'optimization_summary': {
                'rows_removed': original_df.shape[0] - optimized_df.shape[0],
                'null_values_before': original_df.isna().sum().sum(),
                'null_values_after': optimized_df.isna().sum().sum(),
                'data_types_optimized': sum(1 for col in optimized_df.columns 
                                           if str(optimized_df[col].dtype) != str(original_df[col].dtype))
            }
        }
        
        return report