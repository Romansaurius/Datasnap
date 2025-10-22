"""
🌍 UNIVERSAL GLOBAL AI ENGINE 🌍
IA GLOBAL que funciona con CUALQUIER archivo de CUALQUIER tipo
- Detecta automáticamente el contenido
- Corrige TODOS los errores posibles
- Optimiza PERFECTAMENTE cualquier dato
- Funciona con CUALQUIER estructura de datos
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import random
import string

class UniversalGlobalAI:
    """IA GLOBAL UNIVERSAL que funciona con CUALQUIER archivo"""
    
    def __init__(self):
        # Patrones UNIVERSALES de detección
        self.universal_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'[\+]?[\d\s\-\(\)]{7,20}',
            'url': r'https?://[^\s]+',
            'date': r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            'time': r'\d{1,2}:\d{2}(:\d{2})?',
            'number': r'-?\d+\.?\d*',
            'currency': r'[\$€£¥]\s*\d+\.?\d*|\d+\.?\d*\s*[\$€£¥]',
            'percentage': r'\d+\.?\d*\s*%',
            'ip': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            'credit_card': r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}',
            'postal_code': r'\d{5}(-\d{4})?|\d{5}',
            'boolean': r'(true|false|yes|no|si|sí|1|0|on|off|activo|inactivo)',
            'name': r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*'
        }
        
        # Correcciones UNIVERSALES
        self.universal_corrections = {
            'email_domains': {
                'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'gmaill.com': 'gmail.com',
                'hotmial.com': 'hotmail.com', 'hotmial.com': 'hotmail.com', 'hotmailcom': 'hotmail.com',
                'yahoo.co': 'yahoo.com', 'yahooo.com': 'yahoo.com', 'yaho.com': 'yahoo.com',
                'outlok.com': 'outlook.com', 'outllok.com': 'outlook.com', 'outlook.co': 'outlook.com',
                '.con': '.com', '.cmo': '.com', '.ocm': '.com', '.comm': '.com'
            },
            'boolean_values': {
                'si': True, 'sí': True, 'yes': True, 'true': True, '1': True, 'on': True, 'activo': True,
                'no': False, 'false': False, '0': False, 'off': False, 'inactivo': False
            },
            'common_typos': {
                'teh': 'the', 'adn': 'and', 'recieve': 'receive', 'seperate': 'separate',
                'definately': 'definitely', 'occured': 'occurred', 'begining': 'beginning'
            }
        }
        
        # Valores UNIVERSALES por defecto
        self.default_values = {
            'names': ['Usuario', 'Cliente', 'Persona', 'Individuo'],
            'cities': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza', 'Málaga'],
            'countries': ['España', 'México', 'Argentina', 'Colombia', 'Chile', 'Perú'],
            'companies': ['Empresa', 'Compañía', 'Organización', 'Corporación'],
            'categories': ['General', 'Estándar', 'Básico', 'Principal']
        }
    
    def process_any_data(self, data: Any) -> Any:
        """Procesa CUALQUIER tipo de datos de CUALQUIER archivo"""
        
        try:
            # 1. DETECTAR TIPO DE DATOS
            data_type = self._detect_data_type(data)
            
            # 2. CONVERTIR A FORMATO UNIVERSAL
            universal_data = self._convert_to_universal_format(data, data_type)
            
            # 3. APLICAR IA GLOBAL
            optimized_data = self._apply_global_ai(universal_data)
            
            # 4. CONVERTIR DE VUELTA AL FORMATO ORIGINAL
            final_data = self._convert_back_to_original_format(optimized_data, data_type)
            
            return final_data
            
        except Exception as e:
            # FALLBACK: Si falla todo, al menos devolver DataFrame básico
            if isinstance(data, pd.DataFrame):
                return self._apply_basic_corrections(data)
            else:
                # Crear DataFrame básico y aplicar correcciones
                basic_df = pd.DataFrame({'data': [str(data)]})
                return self._apply_basic_corrections(basic_df)
    
    def _apply_basic_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica correcciones básicas como fallback"""
        
        try:
            # Limpieza básica
            df = self._universal_cleaning(df)
            
            # Correcciones simples por columna
            for col in df.columns:
                col_lower = col.lower()
                
                if 'email' in col_lower:
                    df[col] = self._correct_emails(df[col])
                elif 'nombre' in col_lower or 'name' in col_lower:
                    df[col] = self._correct_names_simple(df[col])
                elif 'precio' in col_lower or 'price' in col_lower:
                    df[col] = self._correct_prices_simple(df[col])
                elif 'activo' in col_lower:
                    df[col] = self._correct_booleans(df[col])
            
            return df
            
        except Exception:
            # Si incluso esto falla, devolver el DataFrame original
            return df
    
    def _correct_names_simple(self, series: pd.Series) -> pd.Series:
        """Corrección simple de nombres"""
        def fix_name(name):
            if pd.isna(name) or str(name).lower() in ['nan', 'none', '']:
                return 'Usuario'  # Nombre por defecto
            
            name = str(name).strip().title()
            return name
        
        return series.apply(fix_name)
    
    def _correct_prices_simple(self, series: pd.Series) -> pd.Series:
        """Corrección simple de precios"""
        
        def fix_price(price):
            if pd.isna(price):
                return 100.0  # Precio por defecto
            
            try:
                if isinstance(price, str):
                    # Limpiar y convertir
                    clean_price = re.sub(r'[^\d\.]', '', str(price))
                    return float(clean_price) if clean_price else 100.0
                else:
                    return float(price)
            except:
                return 100.0
        
        return series.apply(fix_price)
    
    def _detect_data_type(self, data: Any) -> str:
        """Detecta automáticamente el tipo de datos"""
        
        if isinstance(data, pd.DataFrame):
            return 'dataframe'
        elif isinstance(data, dict):
            return 'dict'
        elif isinstance(data, list):
            return 'list'
        elif isinstance(data, str):
            # Detectar si es JSON, CSV, SQL, etc.
            data_lower = data.lower().strip()
            if data_lower.startswith('{') or data_lower.startswith('['):
                return 'json_string'
            elif 'create table' in data_lower or 'insert into' in data_lower:
                return 'sql_string'
            elif ',' in data and '\n' in data:
                return 'csv_string'
            else:
                return 'text_string'
        else:
            return 'unknown'
    
    def _convert_to_universal_format(self, data: Any, data_type: str) -> pd.DataFrame:
        """Convierte CUALQUIER formato a DataFrame universal"""
        
        try:
            if data_type == 'dataframe':
                return data.copy()
            
            elif data_type == 'dict':
                return pd.DataFrame([data])
            
            elif data_type == 'list':
                if len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({'data': data})
            
            elif data_type == 'json_string':
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    return pd.DataFrame(json_data)
                else:
                    return pd.DataFrame([json_data])
            
            elif data_type == 'csv_string':
                from io import StringIO
                return pd.read_csv(StringIO(data))
            
            elif data_type == 'sql_string':
                # Extraer datos de SQL
                return self._extract_data_from_sql(data)
            
            elif data_type == 'text_string':
                # Intentar detectar estructura en texto
                return self._parse_text_to_dataframe(data)
            
            else:
                # Formato desconocido - crear DataFrame básico
                return pd.DataFrame({'data': [str(data)]})
                
        except Exception as e:
            # Si falla todo, crear DataFrame con el dato original
            return pd.DataFrame({'original_data': [str(data)], 'error': [str(e)]})
    
    def _apply_global_ai(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica IA GLOBAL a cualquier DataFrame usando optimización inteligente"""
        
        try:
            # Primero intentar optimizador avanzado con ML
            from .advanced_ml_optimizer import AdvancedMLOptimizer
            ml_optimizer = AdvancedMLOptimizer()
            df_ml = ml_optimizer.optimize_with_ml(df)
            
            # Luego aplicar optimizador inteligente
            from .intelligent_data_optimizer import IntelligentDataOptimizer
            intelligent_optimizer = IntelligentDataOptimizer()
            return intelligent_optimizer.optimize_data(df_ml)
            
        except Exception as e:
            print(f"Error en optimización avanzada: {e}")
            try:
                # Fallback a optimizador inteligente solo
                from .intelligent_data_optimizer import IntelligentDataOptimizer
                intelligent_optimizer = IntelligentDataOptimizer()
                return intelligent_optimizer.optimize_data(df)
            except Exception as e2:
                print(f"Error en optimización inteligente: {e2}")
                # Fallback final a método básico
                return self._apply_basic_optimization(df)
    
    def _apply_basic_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización básica como fallback"""
        
        try:
            # 1. LIMPIEZA UNIVERSAL
            df = self._universal_cleaning(df)
            
            # 2. DETECCIÓN AUTOMÁTICA DE COLUMNAS
            df = self._auto_detect_columns(df)
            
            # 3. CORRECCIÓN UNIVERSAL (sin predicción aleatoria)
            df = self._universal_correction_safe(df)
            
            # 4. MANEJO SEGURO DE VALORES FALTANTES
            df = self._safe_missing_value_handling(df)
            
            # 5. OPTIMIZACIÓN UNIVERSAL
            df = self._universal_optimization(df)
            
            return df
            
        except Exception as e:
            print(f"Error en optimización básica: {e}")
            # Fallback mínimo: solo limpieza
            return self._minimal_cleaning(df)
    
    def _minimal_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza mínima como último recurso"""
        try:
            # Solo reemplazar valores obviamente nulos
            null_values = ['', 'nan', 'null', 'none', 'n/a']
            df = df.replace(null_values, pd.NA)
            df = df.dropna(how='all')
            return df.reset_index(drop=True)
        except:
            return df
    
    def _universal_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza UNIVERSAL que funciona con CUALQUIER dato"""
        
        # Valores nulos universales
        null_values = [
            '', ' ', '  ', '   ', 'nan', 'NaN', 'null', 'NULL', 'None', 'NONE',
            'n/a', 'N/A', 'na', 'NA', 'undefined', 'UNDEFINED', '-', '--', '---',
            'missing', 'MISSING', 'empty', 'EMPTY', '?', '??', 'unknown', 'UNKNOWN',
            'sin datos', 'no data', 'no disponible', 'not available', 'vacío', 'vacio',
            '#N/A', '#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!'
        ]
        
        # Reemplazar valores nulos
        df = df.replace(null_values, pd.NA)
        
        # Limpiar espacios en TODAS las columnas
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        
        # Resetear índice
        df = df.reset_index(drop=True)
        
        return df
    
    def _auto_detect_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta automáticamente el tipo de CADA columna"""
        
        column_types = {}
        
        for col in df.columns:
            sample_data = df[col].dropna().astype(str).head(20)
            
            # Detectar por contenido usando patrones universales
            detected_type = 'text'  # Por defecto
            
            for pattern_name, pattern in self.universal_patterns.items():
                matches = sample_data.str.contains(pattern, regex=True, na=False).sum()
                if matches > len(sample_data) * 0.3:  # Si más del 30% coincide
                    detected_type = pattern_name
                    break
            
            column_types[col] = detected_type
        
        # Guardar tipos detectados como atributo
        self.detected_column_types = column_types
        
        return df
    
    def _universal_correction_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrección UNIVERSAL segura sin generar datos aleatorios"""
        
        for col in df.columns:
            col_type = self.detected_column_types.get(col, 'text')
            
            if col_type == 'email':
                df[col] = self._correct_emails_safe(df[col])
            elif col_type == 'phone':
                df[col] = self._correct_phones_safe(df[col])
            elif col_type == 'boolean':
                df[col] = self._correct_booleans_safe(df[col])
            elif col_type == 'number':
                df[col] = self._correct_numbers_safe(df[col])
            elif col_type == 'date':
                df[col] = self._correct_dates_safe(df[col])
            else:
                df[col] = self._correct_text_safe(df[col])
        
        return df
    
    def _safe_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manejo seguro de valores faltantes sin inventar datos"""
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0 or missing_count == len(df):
                continue
            
            col_type = self.detected_column_types.get(col, 'text')
            
            # Solo predecir si hay suficientes datos válidos
            valid_count = df[col].notna().sum()
            if valid_count < 3:
                continue
            
            try:
                if col_type == 'number':
                    # Usar mediana para números
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                
                elif col_type == 'boolean':
                    # Usar moda para booleanos
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                
                # Para otros tipos (email, phone, text, etc.), mantener NA
                
            except Exception:
                continue
        
        return df
    
    def _universal_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización UNIVERSAL de tipos de datos"""
        
        for col in df.columns:
            col_type = self.detected_column_types.get(col, 'text')
            
            if col_type == 'number':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif col_type == 'boolean':
                df[col] = df[col].astype('boolean', errors='ignore')
        
        return df
    
    # Métodos de corrección específicos
    def _correct_emails(self, series: pd.Series) -> pd.Series:
        """Corrige emails universalmente"""
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email = str(email).lower().strip()
            
            # Aplicar correcciones de dominios
            for wrong, correct in self.universal_corrections['email_domains'].items():
                email = email.replace(wrong, correct)
            
            # Completar emails incompletos
            if '@' not in email and email != 'nan':
                email += '@gmail.com'
            elif email.endswith('@'):
                email += 'gmail.com'
            
            return email
        
        return series.apply(fix_email)
    
    def _correct_phones(self, series: pd.Series) -> pd.Series:
        """Corrige teléfonos universalmente"""
        def fix_phone(phone):
            if pd.isna(phone):
                return phone
            
            phone = str(phone).strip()
            # Limpiar caracteres no numéricos excepto +
            phone = re.sub(r'[^\d\+]', '', phone)
            
            # Agregar código de país si falta
            if phone and not phone.startswith('+') and len(phone) >= 9:
                phone = '+34' + phone
            
            return phone
        
        return series.apply(fix_phone)
    
    def _correct_names(self, series: pd.Series) -> pd.Series:
        """Corrige nombres universalmente"""
        def fix_name(name):
            if pd.isna(name):
                return name
            
            name = str(name).strip().title()
            
            # Corregir errores comunes
            for wrong, correct in self.universal_corrections['common_typos'].items():
                name = name.replace(wrong.title(), correct.title())
            
            return name
        
        return series.apply(fix_name)
    
    def _correct_booleans(self, series: pd.Series) -> pd.Series:
        """Corrige booleanos universalmente"""
        def fix_boolean(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).lower().strip()
            return self.universal_corrections['boolean_values'].get(value_str, value)
        
        return series.apply(fix_boolean)
    
    def _correct_numbers(self, series: pd.Series) -> pd.Series:
        """Corrige números universalmente"""
        def fix_number(num):
            if pd.isna(num):
                return num
            
            num_str = str(num).strip()
            
            # Manejar casos específicos
            if num_str.lower() in ['invalid', 'n/a', 'null', 'none', '']:
                return np.nan
            
            # Remover caracteres no numéricos excepto punto y signo
            num_clean = re.sub(r'[^\d\.\-]', '', num_str)
            
            try:
                result = float(num_clean)
                # Validar rangos razonables para diferentes tipos de números
                if abs(result) > 1e10:  # Números extremadamente grandes
                    return np.nan
                return result
            except:
                return np.nan
        
        return series.apply(fix_number)
    
    def _correct_dates(self, series: pd.Series) -> pd.Series:
        """Corrige fechas universalmente"""
        return pd.to_datetime(series, errors='coerce')
    
    def _correct_urls(self, series: pd.Series) -> pd.Series:
        """Corrige URLs universalmente"""
        def fix_url(url):
            if pd.isna(url):
                return url
            
            url = str(url).strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            return url
        
        return series.apply(fix_url)
    
    def _correct_text(self, series: pd.Series) -> pd.Series:
        """Corrige texto general"""
        def fix_text(text):
            if pd.isna(text):
                return text
            
            text = str(text).strip()
            
            # Corregir errores comunes
            for wrong, correct in self.universal_corrections['common_typos'].items():
                text = text.replace(wrong, correct)
            
            return text
        
        return series.apply(fix_text)
    
    # Métodos de predicción
    def _predict_emails(self, series: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Predice emails faltantes"""
        name_cols = [col for col in df.columns if 'name' in col.lower() or 'nombre' in col.lower()]
        
        def predict_email(email, idx):
            if pd.notna(email):
                return email
            
            # Intentar generar desde nombre
            if name_cols:
                name = df.loc[idx, name_cols[0]]
                if pd.notna(name):
                    clean_name = str(name).lower().replace(' ', '.')
                    clean_name = re.sub(r'[^a-z0-9\.]', '', clean_name)
                    return f"{clean_name}@gmail.com"
            
            # Email genérico
            return f"usuario{idx}@gmail.com"
        
        result = series.copy()
        for idx in result.index:
            if pd.isna(result.loc[idx]):
                result.loc[idx] = predict_email(result.loc[idx], idx)
        return result
    
    def _predict_phones(self, series: pd.Series) -> pd.Series:
        """Predice teléfonos faltantes"""
        def predict_phone(phone):
            if pd.notna(phone):
                return phone
            return f"+34{random.randint(600000000, 699999999)}"
        
        return series.apply(predict_phone)
    

    
    def _predict_numbers(self, series: pd.Series) -> pd.Series:
        """Predice números faltantes"""
        median_val = series.median()
        if pd.isna(median_val):
            median_val = 0
        return series.fillna(median_val)
    
    def _predict_booleans(self, series: pd.Series) -> pd.Series:
        """Predice booleanos faltantes"""
        return series.fillna(True)  # Por defecto True
    
    def _predict_dates(self, series: pd.Series) -> pd.Series:
        """Predice fechas faltantes"""
        return series.fillna(pd.Timestamp.now())
    

    
    def _apply_critical_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica correcciones críticas específicas"""
        # Desactivar correcciones críticas que generan datos aleatorios
        return df
    
    # Métodos de corrección seguros
    def _correct_emails_safe(self, series: pd.Series) -> pd.Series:
        """Corrige emails sin inventar datos"""
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email = str(email).lower().strip()
            
            # Aplicar correcciones de dominios
            for wrong, correct in self.universal_corrections['email_domains'].items():
                email = email.replace(wrong, correct)
            
            # Solo corregir si tiene estructura básica de email
            if '@' in email and '.' in email.split('@')[-1]:
                return email
            else:
                return pd.NA  # No inventar emails
        
        return series.apply(fix_email)
    
    def _correct_phones_safe(self, series: pd.Series) -> pd.Series:
        """Corrige teléfonos sin inventar datos"""
        def fix_phone(phone):
            if pd.isna(phone):
                return phone
            
            phone = str(phone).strip()
            # Limpiar caracteres no numéricos excepto +
            phone = re.sub(r'[^\d\+]', '', phone)
            
            # Solo devolver si tiene longitud razonable
            if 7 <= len(phone.replace('+', '')) <= 15:
                return phone
            else:
                return pd.NA
        
        return series.apply(fix_phone)
    
    def _correct_booleans_safe(self, series: pd.Series) -> pd.Series:
        """Corrige booleanos sin inventar datos"""
        def fix_boolean(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).lower().strip()
            return self.universal_corrections['boolean_values'].get(value_str, pd.NA)
        
        return series.apply(fix_boolean)
    
    def _correct_numbers_safe(self, series: pd.Series) -> pd.Series:
        """Corrige números sin inventar datos"""
        def fix_number(num):
            if pd.isna(num):
                return num
            
            num_str = str(num).strip()
            
            # Manejar casos específicos
            if num_str.lower() in ['invalid', 'n/a', 'null', 'none', '']:
                return pd.NA
            
            # Remover caracteres no numéricos excepto punto y signo
            num_clean = re.sub(r'[^\d\.\-]', '', num_str)
            
            try:
                result = float(num_clean)
                # Validar rangos razonables
                if abs(result) > 1e10:  # Números extremadamente grandes
                    return pd.NA
                return result
            except:
                return pd.NA
        
        return series.apply(fix_number)
    
    def _correct_dates_safe(self, series: pd.Series) -> pd.Series:
        """Corrige fechas sin inventar datos"""
        return pd.to_datetime(series, errors='coerce')
    
    def _correct_text_safe(self, series: pd.Series) -> pd.Series:
        """Corrige texto sin inventar datos"""
        def fix_text(text):
            if pd.isna(text):
                return text
            
            text = str(text).strip()
            
            # Solo limpiar espacios múltiples
            text = re.sub(r'\s+', ' ', text)
            
            return text if text else pd.NA
        
        return series.apply(fix_text)
    
    def _extract_data_from_sql(self, sql_content: str) -> pd.DataFrame:
        """Extrae datos de SQL"""
        try:
            # Buscar INSERT statements
            insert_pattern = r'INSERT INTO\s+`?(\w+)`?\s*\(([^)]+)\)\s*VALUES\s*((?:\([^)]+\)(?:\s*,\s*)*)+);'
            matches = re.findall(insert_pattern, sql_content, re.IGNORECASE | re.DOTALL)
            
            all_data = []
            
            for match in matches:
                table = match[0]
                cols_str = match[1]
                values_str = match[2]
                
                columns = [col.strip('` ') for col in cols_str.split(',')]
                
                # Parse values
                val_pattern = r'\(([^)]+)\)'
                val_matches = re.findall(val_pattern, values_str)
                
                for val in val_matches:
                    row = [v.strip("'\" ") for v in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", val)]
                    if len(row) == len(columns):
                        row_dict = dict(zip(columns, row))
                        row_dict['_table'] = table
                        all_data.append(row_dict)
            
            if all_data:
                return pd.DataFrame(all_data)
            else:
                return pd.DataFrame({'sql_content': [sql_content]})
                
        except Exception:
            return pd.DataFrame({'sql_content': [sql_content]})
    
    def _parse_text_to_dataframe(self, text: str) -> pd.DataFrame:
        """Parsea texto a DataFrame"""
        lines = text.strip().split('\n')
        
        # Intentar detectar separadores
        separators = ['\t', ',', ';', '|', ' ']
        best_separator = ','
        max_columns = 0
        
        for sep in separators:
            columns = len(lines[0].split(sep))
            if columns > max_columns:
                max_columns = columns
                best_separator = sep
        
        # Crear DataFrame
        data = []
        for line in lines:
            row = line.split(best_separator)
            data.append(row)
        
        if data:
            return pd.DataFrame(data[1:], columns=data[0] if len(data) > 1 else None)
        else:
            return pd.DataFrame({'text': [text]})
    
    def _convert_back_to_original_format(self, df: pd.DataFrame, original_type: str) -> Any:
        """Convierte de vuelta al formato original"""
        
        if original_type == 'dataframe':
            return df
        elif original_type == 'dict':
            return df.to_dict('records')[0] if len(df) > 0 else {}
        elif original_type == 'list':
            return df.to_dict('records')
        elif original_type == 'json_string':
            return df.to_json(orient='records', indent=2)
        elif original_type == 'csv_string':
            return df.to_csv(index=False)
        elif original_type == 'sql_string':
            return self._convert_dataframe_to_sql(df)
        else:
            return df.to_string()
    
    def _convert_dataframe_to_sql(self, df: pd.DataFrame) -> str:
        """Convierte DataFrame optimizado de vuelta a SQL"""
        
        # Agrupar por tabla si existe columna _table
        if '_table' in df.columns:
            tables = df['_table'].unique()
            sql_parts = []
            
            for table in tables:
                table_data = df[df['_table'] == table].drop('_table', axis=1)
                sql_parts.append(self._generate_sql_for_table(table, table_data))
            
            return '\n\n'.join(sql_parts)
        else:
            return self._generate_sql_for_table('optimized_table', df)
    
    def _generate_sql_for_table(self, table_name: str, df: pd.DataFrame) -> str:
        """Genera SQL para una tabla"""
        
        sql_parts = [
            f"-- Tabla {table_name} optimizada por IA Global",
            f"CREATE TABLE {table_name} ("
        ]
        
        # Generar columnas
        col_defs = []
        for col in df.columns:
            col_type = self._infer_sql_type(df[col])
            col_defs.append(f"    {col} {col_type}")
        
        sql_parts.append(',\n'.join(col_defs))
        sql_parts.append(");")
        sql_parts.append("")
        
        # Generar INSERTs
        if not df.empty:
            cols_str = f"({', '.join(df.columns)})"
            sql_parts.append(f"INSERT INTO {table_name} {cols_str} VALUES")
            
            value_rows = []
            for _, row in df.iterrows():
                values = []
                for val in row:
                    if pd.isna(val):
                        values.append('NULL')
                    elif isinstance(val, str):
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    else:
                        values.append(str(val))
                value_rows.append(f"({', '.join(values)})")
            
            sql_parts.append(',\n'.join(value_rows) + ";")
        
        return '\n'.join(sql_parts)
    
    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infiere tipo SQL"""
        if pd.api.types.is_integer_dtype(series):
            return 'INT'
        elif pd.api.types.is_float_dtype(series):
            return 'DECIMAL(10,2)'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'DATETIME'
        elif pd.api.types.is_bool_dtype(series):
            return 'BOOLEAN'
        else:
            max_len = series.astype(str).str.len().max()
            if max_len <= 255:
                return 'VARCHAR(255)'
            else:
                return 'TEXT'
