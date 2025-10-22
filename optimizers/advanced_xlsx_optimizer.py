#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador XLSX Avanzado para DataSnap IA - Versión Mejorada
Maneja archivos Excel con todos los errores posibles usando las mismas funciones inteligentes del CSV
"""

import pandas as pd
import numpy as np
import re
import html
from datetime import datetime
from io import BytesIO, StringIO
import base64
from difflib import SequenceMatcher
import pycountry
from dateutil import parser
import openpyxl

class AdvancedXLSXOptimizer:
    """Optimizador XLSX que corrige todos los errores posibles"""
    
    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
    
    def optimize_xlsx(self, xlsx_content) -> str:
        """Optimización XLSX completa"""
        
        try:
            # 1. Parse XLSX content
            df = self.parse_xlsx_content(xlsx_content)
            
            # Check if parsing failed
            if 'error' in df.columns:
                error_msg = df['error'].iloc[0] if len(df) > 0 else "Unknown parsing error"
                print(f"[XLSX ERROR] Parsing failed: {error_msg}")
                return f"Error parsing XLSX: {error_msg}"
            
            # 2. Clean and optimize data
            df = self.clean_data(df)
            
            # 3. Fix data types
            df = self.fix_data_types(df)
            
            # 4. Validate and correct values
            df = self.validate_and_correct_values(df)
            
            # 5. Remove duplicates and empty rows
            df = self.remove_duplicates_and_empty(df)
            
            # 6. Normalize column names
            df = self.normalize_column_names(df)
            
            # Return as CSV since XLSX generation is complex
            result = df.to_csv(index=False, encoding='utf-8')
            return result
            
        except Exception as e:
            print(f"[XLSX ERROR] Exception during optimization: {str(e)}")
            import traceback
            print(f"[XLSX ERROR] Traceback: {traceback.format_exc()}")
            self.corrections_applied.append(f"Error during optimization: {e}")
            return f"Error: {e}"
    
    def parse_xlsx_content(self, content) -> pd.DataFrame:
        """Parse XLSX content with multiple strategies"""
        
        try:
            # If content is already bytes, use directly
            if isinstance(content, bytes):
                df = pd.read_excel(BytesIO(content))
                self.corrections_applied.append("XLSX parsed from bytes")
                return df
        except Exception as e:
            print(f"Error parsing bytes: {e}")
        
        try:
            # If content is string, try base64 decode
            if isinstance(content, str):
                if content.startswith('UEsD'):  # ZIP file signature in base64
                    decoded = base64.b64decode(content)
                    df = pd.read_excel(BytesIO(decoded))
                    self.corrections_applied.append("XLSX parsed from base64")
                    return df
        except Exception as e:
            print(f"Error parsing base64: {e}")
        
        try:
            # Try encoding as latin1 and reading
            if isinstance(content, str):
                df = pd.read_excel(BytesIO(content.encode('latin1')))
                self.corrections_applied.append("XLSX parsed with latin1")
                return df
        except Exception as e:
            print(f"Error parsing latin1: {e}")
        
        # Return error DataFrame if all parsing fails
        return pd.DataFrame({'error': ['Failed to parse XLSX content']})
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data values"""
        
        self.original_rows = len(df)
        
        # Replace common null representations
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na', 
                      '#N/A', '#NULL!', 'nil', 'NIL', '-', '--', '---', 'undefined', 
                      'UNDEFINED', 'missing', 'MISSING', '?', '??', '???', 'NaN']
        
        df = df.replace(null_values, np.nan)
        
        # Clean whitespace
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
        
        self.corrections_applied.append("Data cleaned and null values standardized")
        return df
    
    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and optimize data types"""
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Fix email columns
            if any(keyword in col_lower for keyword in ['email', 'mail', 'correo']):
                df[col] = self.fix_emails(df[col])
            
            # Fix phone columns
            elif any(keyword in col_lower for keyword in ['phone', 'telefono', 'tel', 'contacto']):
                df[col] = self.fix_phones(df[col])
            
            # Fix date columns
            elif any(keyword in col_lower for keyword in ['fecha', 'date', 'birth', 'nacimiento', 'contrato', 'registro']):
                df[col] = self.fix_dates(df[col])
            
            # Fix numeric columns
            elif any(keyword in col_lower for keyword in ['edad', 'age']):
                df[col] = self.fix_ages(df[col])
            
            elif any(keyword in col_lower for keyword in ['salario', 'salary', 'precio', 'price', 'anual', 'cost', 'valor']):
                df[col] = self.fix_monetary_values(df[col])
            
            elif any(keyword in col_lower for keyword in ['stock', 'cantidad', 'qty', 'quantity']):
                df[col] = self.fix_quantities(df[col])
            
            # Fix boolean columns
            elif any(keyword in col_lower for keyword in ['activo', 'active', 'enabled', 'esta', 'status']):
                df[col] = self.fix_booleans(df[col])
            
            # Fix name columns
            elif any(keyword in col_lower for keyword in ['nombre', 'name', 'apellido', 'completo', 'student', 'customer']):
                df[col] = self.fix_names(df[col])
            
            # Fix city columns
            elif any(keyword in col_lower for keyword in ['ciudad', 'city', 'residencia']):
                df[col] = self.fix_cities(df[col])
            
            # Fix country columns
            elif any(keyword in col_lower for keyword in ['pais', 'country', 'nation']):
                df[col] = self.fix_countries(df[col])
            
            # Fix gender columns
            elif any(keyword in col_lower for keyword in ['genero', 'gender', 'sexo']):
                df[col] = self.fix_genders(df[col])
        
        self.corrections_applied.append("Data types fixed and optimized")
        return df
    
    def fix_emails(self, series: pd.Series) -> pd.Series:
        """Fix email addresses using smart correction"""
        series = series.astype(str)
        
        # Apply smart email correction
        series = series.apply(self._smart_email_correction)
        
        return series
    
    def _smart_email_correction(self, email):
        """Smart email correction without breaking valid emails"""
        if pd.isna(email) or str(email).strip() == '':
            return email
        
        email_str = str(email).lower().strip()
        
        # Remove mailto: prefix if present
        if email_str.startswith('mailto:'):
            email_str = email_str.replace('mailto:', '')
        
        # If email doesn't have @, add @gmail.com
        if '@' not in email_str:
            return f"{email_str}@gmail.com"
        
        # If email ends with @, add gmail.com
        if email_str.endswith('@'):
            return f"{email_str}gmail.com"
        
        # Split email into user and domain parts
        if '@' in email_str:
            user_part, domain_part = email_str.split('@', 1)
            
            # Fix common domain issues
            if domain_part == 'gmai' or domain_part == 'gmail':
                domain_part = 'gmail.com'
            elif domain_part == 'yahoo' or domain_part == 'yahooo':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmail' or domain_part == 'hotmial':
                domain_part = 'hotmail.com'
            elif domain_part == 'outlook' or domain_part == 'outlok':
                domain_part = 'outlook.com'
            elif domain_part == 'gmailcom':  # Missing dot
                domain_part = 'gmail.com'
            elif domain_part == 'yahoocom':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmailcom':
                domain_part = 'hotmail.com'
            elif domain_part.endswith('.comm'):  # Extra 'm'
                domain_part = domain_part.replace('.comm', '.com')
            elif domain_part == 'gmai.com':
                domain_part = 'gmail.com'
            elif domain_part == 'yahooo.com':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmial.com':
                domain_part = 'hotmail.com'
            elif domain_part == 'outlok.com':
                domain_part = 'outlook.com'
            elif domain_part == 'gmial.com':
                domain_part = 'gmail.com'
            
            # Reconstruct email
            email_str = f"{user_part}@{domain_part}"
        
        # Final validation - if it looks like an email now, return it
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,4}$', email_str):
            return email_str
        
        return email_str
    
    def fix_phones(self, series: pd.Series) -> pd.Series:
        """Fix phone numbers"""
        series = series.astype(str)
        
        # Remove invalid phone indicators
        series = series.replace(['invalid_phone', 'invalid', 'sin_telefono', 'no_phone'], np.nan)
        
        # Clean phone format
        series = series.str.replace(r'[^\\d+\\-\\s()]', '', regex=True)
        
        # Keep phones that have reasonable length
        series = series.where(series.str.len().between(5, 25), series)
        
        return series
    
    def fix_dates(self, series: pd.Series) -> pd.Series:
        """Fix date values"""
        series = series.astype(str)
        
        # Apply smart date conversion
        series = series.apply(self._convert_date_format)
        
        return series
    
    def _convert_date_format(self, date_str):
        """Convert various date formats to YYYY-MM-DD with intelligent validation"""
        if pd.isna(date_str) or str(date_str).strip() == '':
            return np.nan
        
        date_str = str(date_str).strip()
        
        # Handle invalid date indicators
        if date_str.lower() in ['nan', 'invalid_date', 'never', 'null', 'none']:
            return np.nan
        
        try:
            # Convert DD/MM/YYYY to YYYY-MM-DD
            if re.match(r'^\\d{1,2}/\\d{1,2}/\\d{4}$', date_str):
                parts = date_str.split('/')
                if len(parts) == 3:
                    day, month, year = map(int, parts)
                    
                    # Validate and fix invalid dates
                    if month > 12:
                        month = 12
                    if month < 1:
                        month = 1
                    
                    # Fix impossible days
                    if day > 31:
                        day = 31
                    if day < 1:
                        day = 1
                    
                    # Handle February 30/31
                    if month == 2 and day > 29:
                        day = 28
                    
                    # Handle months with 30 days
                    if month in [4, 6, 9, 11] and day > 30:
                        day = 30
                    
                    return f"{year}-{month:02d}-{day:02d}"
            
            # Already in YYYY-MM-DD format - validate
            elif re.match(r'^\\d{4}-\\d{1,2}-\\d{1,2}$', date_str):
                parts = date_str.split('-')
                if len(parts) == 3:
                    year, month, day = map(int, parts)
                    
                    # Validate and fix
                    if month > 12:
                        month = 12
                    if month < 1:
                        month = 1
                    if day > 31:
                        day = 31
                    if day < 1:
                        day = 1
                    
                    # Handle February
                    if month == 2 and day > 29:
                        day = 28
                    
                    # Handle months with 30 days
                    if month in [4, 6, 9, 11] and day > 30:
                        day = 30
                    
                    return f"{year}-{month:02d}-{day:02d}"
            
            # Try to parse with dateutil as fallback
            try:
                parsed_date = parser.parse(date_str, dayfirst=True)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                return np.nan
                
        except (ValueError, TypeError):
            return np.nan
        
        return np.nan
    
    def fix_ages(self, series: pd.Series) -> pd.Series:
        """Fix age values"""
        series = pd.to_numeric(series, errors='coerce')
        
        # Remove invalid ages
        series = series.where((series >= 0) & (series <= 120), np.nan)
        
        return series
    
    def fix_monetary_values(self, series: pd.Series) -> pd.Series:
        """Fix monetary values"""
        series = series.astype(str)
        
        # Remove currency symbols and clean
        series = series.str.replace(r'[$€£¥₹]', '', regex=True)
        series = series.str.replace(',', '', regex=False)
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
        
        # Remove negative values for prices/salaries
        series = series.where(series >= 0, np.nan)
        
        return series
    
    def fix_quantities(self, series: pd.Series) -> pd.Series:
        """Fix quantity values"""
        series = pd.to_numeric(series, errors='coerce')
        
        # Remove negative quantities
        series = series.where(series >= 0, np.nan)
        
        return series
    
    def fix_booleans(self, series: pd.Series) -> pd.Series:
        """Fix boolean values"""
        series = series.astype(str).str.lower()
        
        # Map various boolean representations
        boolean_map = {
            'true': True, '1': True, 'yes': True, 'si': True, 'sí': True, 'y': True,
            'active': True, 'activo': True, 'enabled': True, 'on': True,
            'false': False, '0': False, 'no': False, 'n': False, 'inactive': False,
            'inactivo': False, 'disabled': False, 'off': False, 'invalid': False
        }
        
        series = series.map(boolean_map)
        
        return series
    
    def fix_names(self, series: pd.Series) -> pd.Series:
        """Fix name values properly"""
        series = series.astype(str)
        
        # Remove empty names
        series = series.replace(['', '0', '1', '2', '3', '4', '5'], np.nan)
        
        # Apply smart name correction
        series = series.apply(self._smart_name_correction)
        
        # Fix common name issues
        series = series.str.replace(r'\\s+', ' ', regex=True)  # Multiple spaces
        series = series.str.strip()
        
        return series
    
    def fix_cities(self, series: pd.Series) -> pd.Series:
        """Fix city names"""
        series = series.astype(str)
        
        # Apply smart city correction
        series = series.apply(self._smart_city_correction)
        
        return series
    
    def fix_countries(self, series: pd.Series) -> pd.Series:
        """Fix country names intelligently"""
        series = series.astype(str)
        
        # Apply smart country correction
        series = series.apply(self._smart_country_correction)
        
        return series
    
    def fix_genders(self, series: pd.Series) -> pd.Series:
        """Fix gender values intelligently"""
        series = series.astype(str)
        
        # Apply smart gender correction
        series = series.apply(self._smart_gender_correction)
        
        return series
    
    def _smart_name_correction(self, name):
        """Smart name correction handling mixed cases properly"""
        if pd.isna(name) or str(name).strip() == '':
            return name
        
        name_str = str(name).strip()
        
        # Handle mixed case names like "ROMAN gomez" -> "Roman Gomez"
        words = name_str.split()
        corrected_words = []
        
        for word in words:
            if word.isupper() or word.islower():
                # Convert to proper case
                corrected_words.append(word.capitalize())
            else:
                # Keep mixed case as is (might be intentional like "McDonald")
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _smart_country_correction(self, country):
        """Smart country correction using pycountry and similarity"""
        if pd.isna(country) or str(country).strip() == '':
            return country
        
        country_str = str(country).strip().lower()
        
        # Common Spanish variations
        spanish_variations = {
            'españa': 'España',
            'spain': 'España', 
            'esp': 'España',
            'espana': 'España',
            'spanish': 'España',
            'es': 'España'
        }
        
        if country_str in spanish_variations:
            return spanish_variations[country_str]
        
        # Try to find country using pycountry
        try:
            # Try by name
            try:
                country_obj = pycountry.countries.lookup(country_str)
                return country_obj.name
            except:
                pass
            
            # Try by alpha_2 code
            try:
                country_obj = pycountry.countries.get(alpha_2=country_str.upper())
                if country_obj:
                    return country_obj.name
            except:
                pass
            
            # Try by alpha_3 code
            try:
                country_obj = pycountry.countries.get(alpha_3=country_str.upper())
                if country_obj:
                    return country_obj.name
            except:
                pass
            
            # Fuzzy matching with common countries
            common_countries = ['España', 'France', 'Germany', 'Italy', 'Portugal', 'United Kingdom', 'United States']
            best_match = self._find_best_match(country_str, [c.lower() for c in common_countries])
            if best_match:
                return common_countries[[c.lower() for c in common_countries].index(best_match)]
                
        except Exception:
            pass
        
        # Return capitalized version as fallback
        return str(country).title()
    
    def _smart_gender_correction(self, gender):
        """Smart gender correction"""
        if pd.isna(gender) or str(gender).strip() == '':
            return gender
        
        gender_str = str(gender).strip().lower()
        
        # Gender mappings
        male_variations = ['m', 'male', 'masculino', 'hombre', 'h', 'man', 'boy', 'niño']
        female_variations = ['f', 'female', 'femenino', 'mujer', 'woman', 'girl', 'niña']
        
        if gender_str in male_variations:
            return 'Masculino'
        elif gender_str in female_variations:
            return 'Femenino'
        
        # Return original if not recognized
        return str(gender).title()
    
    def _smart_city_correction(self, city):
        """Smart city name correction using similarity matching"""
        if pd.isna(city) or str(city).strip() == '':
            return city
        
        city_str = str(city).strip().lower()
        
        # Common cities database for correction
        common_cities = [
            'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao', 'zaragoza',
            'málaga', 'murcia', 'palma', 'córdoba', 'valladolid', 'vigo',
            'paris', 'london', 'berlin', 'rome', 'amsterdam', 'vienna',
            'moscow', 'beijing', 'tokyo', 'new york', 'los angeles', 'chicago',
            'houston', 'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas'
        ]
        
        # If city is very short or truncated, try to find best match
        if len(city_str) >= 4 and len(city_str) <= 12:
            best_match = self._find_best_city_match(city_str, common_cities)
            if best_match:
                return best_match.title()
        
        return str(city).title()
    
    def _find_best_city_match(self, city: str, cities: list) -> str:
        """Find best matching city using similarity"""
        best_ratio = 0
        best_match = None
        
        for candidate in cities:
            # Calculate similarity ratio
            ratio = self._similarity_ratio(city, candidate)
            if ratio > best_ratio and ratio >= 0.7:  # 70% similarity threshold
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    def _find_best_match(self, target: str, candidates: list, threshold: float = 0.6) -> str:
        """Find best matching string using similarity"""
        best_ratio = 0
        best_match = None
        
        for candidate in candidates:
            ratio = SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    def validate_and_correct_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional validation and corrections"""
        
        # Remove rows where all values are null
        df = df.dropna(how='all')
        
        self.corrections_applied.append("Values validated and corrected")
        return df
    
    def remove_duplicates_and_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and empty rows"""
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with too many null values (more than 90% null)
        threshold = max(1, len(df.columns) * 0.1)  # Keep rows with at least 10% valid data
        df = df.dropna(thresh=threshold)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            self.corrections_applied.append(f"Removed {removed_rows} duplicate/empty rows")
        
        self.final_rows = final_rows
        return df
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names"""
        
        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_', regex=False)
        df.columns = df.columns.str.replace(r'[^\\w]', '_', regex=True)
        df.columns = df.columns.str.replace(r'_+', '_', regex=True)
        df.columns = df.columns.str.strip('_')
        
        self.corrections_applied.append("Column names normalized")
        return df
    
    def get_optimization_summary(self) -> str:
        """Get summary of optimizations applied"""
        summary = f"XLSX Optimization Summary:\\n"
        summary += f"- Original rows: {self.original_rows}\\n"
        summary += f"- Final rows: {self.final_rows}\\n"
        summary += f"- Rows removed: {self.original_rows - self.final_rows}\\n"
        summary += f"- Corrections applied:\\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\\n"
        
        return summary
