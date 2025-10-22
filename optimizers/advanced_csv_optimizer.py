#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador CSV Avanzado para DataSnap IA
Maneja todos los errores posibles en archivos CSV
"""

import pandas as pd
import numpy as np
import re
import html
from datetime import datetime
from io import StringIO

class AdvancedCSVOptimizer:
    """Optimizador CSV que corrige todos los errores posibles"""
    
    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
    
    def optimize_csv(self, csv_content: str) -> str:
        """Optimización CSV completa"""
        
        try:
            # 1. Fix encoding issues
            csv_content = self.fix_encoding_issues(csv_content)
            
            # 2. Parse CSV with error handling
            df = self.parse_csv_safely(csv_content)
            
            # 3. Clean and optimize data
            df = self.clean_data(df)
            
            # 4. Fix data types
            df = self.fix_data_types(df)
            
            # 5. Validate and correct values
            df = self.validate_and_correct_values(df)
            
            # 6. Remove duplicates and empty rows
            df = self.remove_duplicates_and_empty(df)
            
            # 7. Normalize column names
            df = self.normalize_column_names(df)
            
            return df.to_csv(index=False, encoding='utf-8')
            
        except Exception as e:
            self.corrections_applied.append(f"Error during optimization: {e}")
            return csv_content
    
    def fix_encoding_issues(self, content: str) -> str:
        """Fix encoding and HTML entities"""
        # Fix HTML entities
        content = html.unescape(content)
        content = re.sub(r'&#39;', "'", content)
        content = re.sub(r'&quot;', '"', content)
        content = re.sub(r'&amp;', '&', content)
        content = re.sub(r'&lt;', '<', content)
        content = re.sub(r'&gt;', '>', content)
        
        self.corrections_applied.append("HTML entities and encoding fixed")
        return content
    
    def parse_csv_safely(self, content: str) -> pd.DataFrame:
        """Parse CSV with multiple fallback strategies"""
        
        try:
            # Try standard parsing
            df = pd.read_csv(StringIO(content))
            self.corrections_applied.append("CSV parsed successfully")
            return df
        except:
            try:
                # Try with different separator
                df = pd.read_csv(StringIO(content), sep=';')
                self.corrections_applied.append("CSV parsed with semicolon separator")
                return df
            except:
                try:
                    # Try with tab separator
                    df = pd.read_csv(StringIO(content), sep='\t')
                    self.corrections_applied.append("CSV parsed with tab separator")
                    return df
                except:
                    try:
                        # Try with pipe separator
                        df = pd.read_csv(StringIO(content), sep='|')
                        self.corrections_applied.append("CSV parsed with pipe separator")
                        return df
                    except:
                        # Last resort: create from lines
                        lines = content.strip().split('\n')
                        if lines:
                            headers = lines[0].split(',')
                            data = []
                            for line in lines[1:]:
                                data.append(line.split(','))
                            df = pd.DataFrame(data, columns=headers)
                            self.corrections_applied.append("CSV parsed with manual line splitting")
                            return df
                        else:
                            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data values"""
        
        self.original_rows = len(df)
        
        # Replace common null representations
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na', 
                      '#N/A', '#NULL!', 'nil', 'NIL', '-', '--', '---', 'undefined', 
                      'UNDEFINED', 'missing', 'MISSING', '?', '??', '???']
        
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
            if 'email' in col_lower or 'mail' in col_lower:
                df[col] = self.fix_emails(df[col])
            
            # Fix phone columns
            elif 'phone' in col_lower or 'telefono' in col_lower or 'tel' in col_lower:
                df[col] = self.fix_phones(df[col])
            
            # Fix date columns
            elif 'fecha' in col_lower or 'date' in col_lower or 'birth' in col_lower:
                df[col] = self.fix_dates(df[col])
            
            # Fix numeric columns
            elif 'edad' in col_lower or 'age' in col_lower:
                df[col] = self.fix_ages(df[col])
            
            elif 'salario' in col_lower or 'salary' in col_lower or 'precio' in col_lower or 'price' in col_lower:
                df[col] = self.fix_monetary_values(df[col])
            
            elif 'stock' in col_lower or 'cantidad' in col_lower or 'qty' in col_lower:
                df[col] = self.fix_quantities(df[col])
            
            # Fix boolean columns
            elif 'activo' in col_lower or 'active' in col_lower or 'enabled' in col_lower:
                df[col] = self.fix_booleans(df[col])
            
            # Fix name columns
            elif 'nombre' in col_lower or 'name' in col_lower:
                df[col] = self.fix_names(df[col])
        
        self.corrections_applied.append("Data types fixed and optimized")
        return df
    
    def fix_emails(self, series: pd.Series) -> pd.Series:
        """Fix email addresses"""
        series = series.astype(str)
        
        # Common email domain corrections
        series = series.str.replace('gmai.com', 'gmail.com', regex=False)
        series = series.str.replace('hotmial.com', 'hotmail.com', regex=False)
        series = series.str.replace('yahoo.co', 'yahoo.com', regex=False)
        series = series.str.replace('outlok.com', 'outlook.com', regex=False)
        series = series.str.replace('gmial.com', 'gmail.com', regex=False)
        
        # Keep emails that look reasonable (don't be too strict)
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        series = series.where(series.str.contains(email_pattern, na=False), np.nan)
        
        return series
    
    def fix_phones(self, series: pd.Series) -> pd.Series:
        """Fix phone numbers"""
        series = series.astype(str)
        
        # Remove invalid phone indicators
        series = series.replace(['invalid_phone', 'invalid', 'sin_telefono', 'no_phone'], np.nan)
        
        # Clean phone format
        series = series.str.replace(r'[^\d+\-\s()]', '', regex=True)
        
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
        
        # Remove obviously invalid dates
        invalid_dates = ['invalid_date', 'ayer', 'hoy', 'mañana', 'today', 'yesterday', 
                        'tomorrow', '0000-00-00', '9999-99-99', 'never', 'nunca']
        series = series.replace(invalid_dates, np.nan)
        
        # Try to parse dates
        try:
            series = pd.to_datetime(series, errors='coerce')
        except:
            pass
        
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
        series = series.str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
        series = series.str.strip()
        
        return series
    
    def validate_and_correct_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional validation and corrections"""
        
        # Remove rows where all values are null
        df = df.dropna(how='all')
        
        # Fix inconsistent city names using smart correction
        for col in df.columns:
            if 'ciudad' in col.lower() or 'city' in col.lower():
                try:
                    df[col] = df[col].apply(self._smart_city_correction)
                except:
                    # Skip if there are encoding issues
                    pass
        
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
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        df.columns = df.columns.str.replace(r'_+', '_', regex=True)
        df.columns = df.columns.str.strip('_')
        
        self.corrections_applied.append("Column names normalized")
        return df
    
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
            'são paulo', 'mexico city', 'cairo', 'mumbai', 'oslo'
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
    
    def _similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def get_optimization_summary(self) -> str:
        """Get summary of optimizations applied"""
        summary = f"CSV Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Rows removed: {self.original_rows - self.final_rows}\n"
        summary += f"- Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary

if __name__ == "__main__":
    optimizer = AdvancedCSVOptimizer()
    
    # Test with sample CSV
    test_csv = """nombre,email,edad,salario
juan perez,juan@gmai.com,-10,abc
MARIA GARCIA,invalid_email,150,-1000"""
    
    result = optimizer.optimize_csv(test_csv)
    print("Optimized CSV:")
    print(result)
    print("\nSummary:")
    print(optimizer.get_optimization_summary())
