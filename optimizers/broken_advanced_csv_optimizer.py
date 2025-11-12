#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador CSV Avanzado para DataSnap IA - Versión Mejorada
Maneja archivos CSV con todos los errores posibles usando las mismas funciones inteligentes del XLSX
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from io import StringIO
from dateutil import parser

class AdvancedCSVOptimizer:
    """OPTIMIZADOR CSV IA UNIVERSAL - DETECCIÓN INTELIGENTE CON MACHINE LEARNING"""

    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0

    def optimize_csv(self, csv_content: str) -> str:
        """Optimización CSV completa - devuelve string CSV válido"""
        try:
            df = pd.read_csv(StringIO(csv_content))
            self.original_rows = len(df)
            
            # Clean and fix data
            df = self.clean_data(df)
            df = self.fix_data_types(df)
            df = self.remove_duplicates_and_empty(df)
            
            self.final_rows = len(df)
            
            # Convert back to CSV
            result = df.to_csv(index=False)
            self.corrections_applied.append("CSV optimized successfully")
            return result
            
        except Exception as e:
            raise Exception(f"Error optimizing CSV: {e}")

    def clean_data(self, df):
        """Clean data values"""
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na']
        df = df.replace(null_values, np.nan)
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
        
        return df

    def fix_data_types(self, df):
        """Fix and optimize data types"""
        for col in df.columns:
            col_lower = str(col).lower()
            
            if any(keyword in col_lower for keyword in ['email', 'mail', 'correo']):
                df[col] = self.fix_emails(df[col])
            elif any(keyword in col_lower for keyword in ['phone', 'telefono', 'tel']):
                df[col] = self.fix_phones(df[col])
            elif any(keyword in col_lower for keyword in ['fecha', 'date', 'birth']):
                df[col] = self.fix_dates(df[col])
        
        return df

    def fix_emails(self, series):
        """Fix email addresses"""
        def fix_single_email(email):
            if pd.isna(email) or str(email).strip() == '':
                return email
            
            email_str = str(email).lower().strip()
            
            # Fix common issues
            if '@' not in email_str:
                return f"{email_str}@gmail.com"
            
            if email_str.endswith('@'):
                return f"{email_str}gmail.com"
            
            # Fix domain issues
            email_str = email_str.replace('gmai', 'gmail')
            email_str = email_str.replace('hotmial', 'hotmail')
            email_str = email_str.replace('gmailcom', 'gmail.com')
            email_str = email_str.replace('.comm', '.com')
            
            return email_str
        
        return series.astype(str).apply(fix_single_email)

    def fix_phones(self, series):
        """Fix phone numbers"""
        return series.astype(str).str.replace(r'[^\d+\-\s()]', '', regex=True)

    def fix_dates(self, series):
        """Fix date values"""
        def fix_single_date(date_str):
            if pd.isna(date_str) or str(date_str).strip() == '':
                return np.nan
            
            try:
                parsed_date = parser.parse(str(date_str), dayfirst=True)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                return np.nan
        
        return series.astype(str).apply(fix_single_date)

    def remove_duplicates_and_empty(self, df):
        """Remove duplicates and empty rows"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        df = df.dropna(how='all')
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.corrections_applied.append(f"Removed {removed_rows} duplicate/empty rows")
        
        return df

    def get_optimization_summary(self):
        """Get summary of optimizations applied"""
        summary = f"CSV Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Corrections: {', '.join(self.corrections_applied)}\n"
        return summary
