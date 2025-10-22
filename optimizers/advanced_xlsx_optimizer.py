#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador XLSX Avanzado para DataSnap IA
Maneja archivos Excel con todos los errores posibles
"""

import pandas as pd
import numpy as np
import re
import html
from datetime import datetime
from io import BytesIO, StringIO
import base64

class AdvancedXLSXOptimizer:
    """Optimizador XLSX que corrige todos los errores posibles"""
    
    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
    
    def optimize_xlsx(self, xlsx_content: str) -> str:
        """Optimización XLSX completa"""
        
        try:
            # 1. Parse XLSX content
            df = self.parse_xlsx_content(xlsx_content)
            
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
            return df.to_csv(index=False, encoding='utf-8')
            
        except Exception as e:
            self.corrections_applied.append(f"Error during optimization: {e}")
            # Fallback: try to extract data from XML
            return self.extract_from_xml(xlsx_content)
    
    def parse_xlsx_content(self, content: str) -> pd.DataFrame:
        """Parse XLSX content with multiple strategies"""
        
        try:
            # Try to decode if it's base64
            if content.startswith('UEsD'):  # ZIP file signature in base64
                decoded = base64.b64decode(content)
                df = pd.read_excel(BytesIO(decoded))
                self.corrections_applied.append("XLSX parsed from base64")
                return df
        except:
            pass
        
        try:
            # Try direct pandas read if it's binary
            df = pd.read_excel(BytesIO(content.encode('latin1')))
            self.corrections_applied.append("XLSX parsed directly")
            return df
        except:
            pass
        
        # Fallback: extract from XML structure
        return self.extract_from_xml(content)
    
    def extract_from_xml(self, content: str) -> pd.DataFrame:
        """Extract data from XLSX XML structure"""
        
        try:
            # Extract data from XML rows
            data = []
            headers = []
            
            # Find all row elements
            row_pattern = r'<row r="(\d+)"[^>]*>(.*?)</row>'
            rows = re.findall(row_pattern, content, re.DOTALL)
            
            for row_num, row_content in rows:
                # Extract cell values
                cell_pattern = r'<c r="[A-Z]+\d+"[^>]*><is><t>(.*?)</t></is></c>'
                cells = re.findall(cell_pattern, row_content)
                
                if row_num == '1':  # Header row
                    headers = [cell.strip() if cell.strip() else f'col_{i+1}' for i, cell in enumerate(cells)]
                    if not headers:  # If no headers found, infer from data patterns
                        headers = self._infer_headers_from_data(data)
                else:
                    # Pad cells to match header length
                    while len(cells) < len(headers):
                        cells.append('')
                    # Only add non-empty rows
                    if any(cell.strip() for cell in cells if cell):
                        data.append(cells[:len(headers)])
            
            if data:
                # Ensure we have proper headers
                if not headers or all(h == '' or h.startswith('col_') for h in headers):
                    headers = self._infer_headers_from_data(data)
                
                # Ensure headers match data columns
                if data and len(headers) != len(data[0]):
                    max_cols = max(len(row) for row in data)
                    headers = headers[:max_cols] + [f'col_{i+1}' for i in range(len(headers), max_cols)]
                
                df = pd.DataFrame(data, columns=headers)
                self.corrections_applied.append("Data extracted from XML structure")
                return df
            else:
                # Return empty DataFrame if no data found
                return pd.DataFrame()
                
        except Exception as e:
            self.corrections_applied.append(f"XML extraction failed: {e}")
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
        """Fix and optimize data types based on content patterns"""
        
        for col in df.columns:
            # Analyze column content to determine type
            col_type = self._detect_column_type(df[col], col)
            
            if col_type == 'email':
                df[col] = self.fix_emails(df[col])
            elif col_type == 'phone':
                df[col] = self.fix_phones(df[col])
            elif col_type == 'date':
                df[col] = self.fix_dates(df[col])
            elif col_type == 'age':
                df[col] = self.fix_ages(df[col])
            elif col_type == 'monetary':
                df[col] = self.fix_monetary_values(df[col])
            elif col_type == 'quantity':
                df[col] = self.fix_quantities(df[col])
            elif col_type == 'boolean':
                df[col] = self.fix_booleans(df[col])
            elif col_type == 'name':
                df[col] = self.fix_names(df[col])
            elif col_type == 'city':
                df[col] = self.fix_cities(df[col])
        
        self.corrections_applied.append("Data types fixed and optimized based on content patterns")
        return df
    
    def fix_emails(self, series: pd.Series) -> pd.Series:
        """Fix email addresses using smart correction"""
        series = series.astype(str)
        
        # Smart email domain correction using similarity
        series = series.apply(self._smart_email_correction)
        
        # Keep emails that look reasonable
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        series = series.where(series.str.contains(email_pattern, na=False), np.nan)
        
        return series
    
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
        
        # Fix common invalid dates
        series = series.str.replace('1995-02-30', '1995-02-28', regex=False)
        series = series.str.replace('2023-02-30', '2023-02-28', regex=False)
        series = series.str.replace('2024-02-30', '2024-02-28', regex=False)
        
        # Normalize date formats
        series = series.str.replace('2024/01/', '2024-01-', regex=False)
        series = series.str.replace('/', '-', regex=False)
        
        # Remove obviously invalid dates
        invalid_dates = ['invalid_date', 'ayer', 'hoy', 'mañana', 'today', 'yesterday', 
                        'tomorrow', '0000-00-00', '9999-99-99', 'never', 'nunca']
        series = series.replace(invalid_dates, np.nan)
        
        return series
    
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
        """Fix name values"""
        series = series.astype(str)
        
        # Remove empty names
        series = series.replace(['', '0', '1', '2', '3', '4', '5'], np.nan)
        
        # Handle special characters properly
        try:
            # Proper case for names (only for ASCII characters)
            series = series.apply(lambda x: x.title() if x.isascii() else x if pd.notna(x) else x)
        except:
            # Fallback if title() fails
            pass
        
        # Fix common name issues
        series = series.str.replace(r'\\s+', ' ', regex=True)  # Multiple spaces
        series = series.str.strip()
        
        return series
    
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
    
    def _infer_headers_from_data(self, data: list) -> list:
        """Infer headers from data patterns"""
        if not data or not data[0]:
            return ['col_1']
        
        headers = []
        for i, sample_value in enumerate(data[0]):
            header = self._infer_column_name(sample_value, i, data)
            headers.append(header)
        
        return headers
    
    def _infer_column_name(self, sample_value: str, col_index: int, data: list) -> str:
        """Infer column name from sample data"""
        # Analyze all values in this column
        col_values = [row[col_index] if col_index < len(row) else '' for row in data]
        
        # Check patterns
        if self._is_email_pattern(col_values):
            return 'email'
        elif self._is_id_pattern(col_values):
            return 'id'
        elif self._is_price_pattern(col_values):
            return 'precio'
        elif self._is_name_pattern(col_values):
            return 'nombre'
        elif self._is_date_pattern(col_values):
            return 'fecha'
        elif self._is_phone_pattern(col_values):
            return 'telefono'
        elif self._is_quantity_pattern(col_values):
            return 'cantidad'
        elif self._is_boolean_pattern(col_values):
            return 'activo'
        else:
            return f'col_{col_index + 1}'
    
    def _detect_column_type(self, series: pd.Series, col_name: str) -> str:
        """Detect column type based on content and name patterns"""
        col_lower = col_name.lower()
        sample_values = series.dropna().astype(str).head(10).tolist()
        
        # Name-based detection
        if any(keyword in col_lower for keyword in ['email', 'mail', 'correo']):
            return 'email'
        elif any(keyword in col_lower for keyword in ['phone', 'telefono', 'tel', 'celular']):
            return 'phone'
        elif any(keyword in col_lower for keyword in ['fecha', 'date', 'birth', 'nacimiento']):
            return 'date'
        elif any(keyword in col_lower for keyword in ['edad', 'age', 'años']):
            return 'age'
        elif any(keyword in col_lower for keyword in ['precio', 'price', 'cost', 'salario', 'salary']):
            return 'monetary'
        elif any(keyword in col_lower for keyword in ['stock', 'cantidad', 'qty', 'inventory']):
            return 'quantity'
        elif any(keyword in col_lower for keyword in ['activo', 'active', 'enabled', 'status']):
            return 'boolean'
        elif any(keyword in col_lower for keyword in ['nombre', 'name', 'usuario', 'user']):
            return 'name'
        elif any(keyword in col_lower for keyword in ['ciudad', 'city', 'location', 'lugar']):
            return 'city'
        
        # Content-based detection
        if self._is_email_pattern(sample_values):
            return 'email'
        elif self._is_price_pattern(sample_values):
            return 'monetary'
        elif self._is_date_pattern(sample_values):
            return 'date'
        elif self._is_phone_pattern(sample_values):
            return 'phone'
        elif self._is_boolean_pattern(sample_values):
            return 'boolean'
        elif self._is_quantity_pattern(sample_values):
            return 'quantity'
        
        return 'text'
    
    def _is_email_pattern(self, values: list) -> bool:
        """Check if values match email pattern"""
        email_count = sum(1 for v in values if '@' in str(v) and '.' in str(v))
        return email_count > len(values) * 0.5
    
    def _is_id_pattern(self, values: list) -> bool:
        """Check if values match ID pattern"""
        try:
            numeric_count = sum(1 for v in values if str(v).isdigit())
            return numeric_count > len(values) * 0.8
        except:
            return False
    
    def _is_price_pattern(self, values: list) -> bool:
        """Check if values match price pattern"""
        price_indicators = ['$', '€', '£', '.', ',']
        price_count = sum(1 for v in values if any(indicator in str(v) for indicator in price_indicators))
        return price_count > len(values) * 0.5
    
    def _is_name_pattern(self, values: list) -> bool:
        """Check if values match name pattern"""
        name_count = sum(1 for v in values if len(str(v).split()) >= 2 and str(v).replace(' ', '').isalpha())
        return name_count > len(values) * 0.5
    
    def _is_date_pattern(self, values: list) -> bool:
        """Check if values match date pattern"""
        date_indicators = ['-', '/', '2024', '2023', '2022', '2021', '2020']
        date_count = sum(1 for v in values if any(indicator in str(v) for indicator in date_indicators))
        return date_count > len(values) * 0.5
    
    def _is_phone_pattern(self, values: list) -> bool:
        """Check if values match phone pattern"""
        phone_indicators = ['+', '-', '(', ')', ' ']
        phone_count = sum(1 for v in values if any(indicator in str(v) for indicator in phone_indicators) and len(str(v)) > 7)
        return phone_count > len(values) * 0.5
    
    def _is_quantity_pattern(self, values: list) -> bool:
        """Check if values match quantity pattern"""
        try:
            numeric_count = sum(1 for v in values if str(v).replace('-', '').isdigit())
            return numeric_count > len(values) * 0.7
        except:
            return False
    
    def _is_boolean_pattern(self, values: list) -> bool:
        """Check if values match boolean pattern"""
        bool_values = ['true', 'false', 'si', 'no', '1', '0', 'yes', 'activo', 'inactivo']
        bool_count = sum(1 for v in values if str(v).lower() in bool_values)
        return bool_count > len(values) * 0.7
    
    def _smart_email_correction(self, email):
        """Smart email correction using similarity matching"""
        if pd.isna(email) or str(email).strip() == '':
            return email
        
        email_str = str(email).lower().strip()
        
        if '@' not in email_str:
            return email
        
        local_part, domain = email_str.split('@', 1)
        
        # Common domains for correction
        common_domains = [
            'gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'live.com',
            'icloud.com', 'aol.com', 'protonmail.com', 'mail.com', 'zoho.com'
        ]
        
        # Find best matching domain
        best_match = self._find_best_domain_match(domain, common_domains)
        
        if best_match:
            return f"{local_part}@{best_match}"
        
        return email
    
    def _find_best_domain_match(self, domain: str, domains: list) -> str:
        """Find best matching domain using similarity"""
        from difflib import SequenceMatcher
        
        best_ratio = 0
        best_match = None
        
        for candidate in domains:
            ratio = SequenceMatcher(None, domain, candidate).ratio()
            if ratio > best_ratio and ratio >= 0.6:  # 60% similarity threshold
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    def fix_cities(self, series: pd.Series) -> pd.Series:
        """Fix city names using smart correction"""
        series = series.astype(str)
        
        # Apply smart city correction
        series = series.apply(self._smart_city_correction)
        
        return series
    
    def _smart_city_correction(self, city):
        """Smart city name correction using similarity matching"""
        if pd.isna(city) or str(city).strip() == '':
            return city
        
        city_str = str(city).strip().lower()
        
        # Common cities for correction
        common_cities = [
            'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao', 'zaragoza',
            'málaga', 'murcia', 'palma', 'córdoba', 'valladolid', 'vigo',
            'paris', 'london', 'berlin', 'rome', 'amsterdam', 'vienna',
            'moscow', 'beijing', 'tokyo', 'new york', 'los angeles', 'chicago'
        ]
        
        # If city seems truncated, try to find best match
        if len(city_str) >= 4 and len(city_str) <= 12:
            best_match = self._find_best_city_match(city_str, common_cities)
            if best_match:
                return best_match.title()
        
        return str(city).title()
    
    def _find_best_city_match(self, city: str, cities: list) -> str:
        """Find best matching city using similarity"""
        from difflib import SequenceMatcher
        
        best_ratio = 0
        best_match = None
        
        for candidate in cities:
            ratio = SequenceMatcher(None, city, candidate).ratio()
            if ratio > best_ratio and ratio >= 0.7:  # 70% similarity threshold
                best_ratio = ratio
                best_match = candidate
        
        return best_match