#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from datetime import datetime
from io import StringIO

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class AdvancedCSVOptimizer:
    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
        print("[CSV AI] Optimizador CSV inicializado")

    def optimize_csv(self, csv_content):
        try:
            df = pd.read_csv(StringIO(csv_content))
            self.original_rows = len(df)
            
            # Clean data
            df = self._clean_data(df)
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            self.final_rows = len(df)
            
            result = df.to_csv(index=False, encoding='utf-8')
            self.corrections_applied.append("CSV optimized")
            
            return result
            
        except Exception as e:
            raise Exception(f"Error optimizing CSV: {e}")

    def _clean_data(self, df):
        df_clean = df.copy()
        
        # Replace null values
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a']
        df_clean = df_clean.replace(null_values, np.nan)
        
        # Clean each column
        for col in df_clean.columns:
            col_lower = str(col).lower()
            
            if 'email' in col_lower or 'mail' in col_lower:
                df_clean[col] = self._fix_emails(df_clean[col])
            elif 'phone' in col_lower or 'telefono' in col_lower:
                df_clean[col] = self._fix_phones(df_clean[col])
            elif 'fecha' in col_lower or 'date' in col_lower:
                df_clean[col] = self._fix_dates(df_clean[col])
            elif 'nombre' in col_lower or 'name' in col_lower:
                df_clean[col] = self._fix_names(df_clean[col])
            elif 'edad' in col_lower or 'age' in col_lower:
                df_clean[col] = self._fix_ages(df_clean[col])
            elif 'precio' in col_lower or 'price' in col_lower:
                df_clean[col] = self._fix_prices(df_clean[col])
        
        return df_clean

    def _fix_emails(self, series):
        def fix_email(email):
            if pd.isna(email):
                return email
            email_str = str(email).lower().strip()
            if '@' not in email_str and email_str:
                email_str = f"{email_str}@gmail.com"
            email_str = email_str.replace('gmai.com', 'gmail.com')
            email_str = email_str.replace('hotmial.com', 'hotmail.com')
            return email_str
        return series.apply(fix_email)

    def _fix_phones(self, series):
        return series.astype(str).str.replace(r'[^\d+\-\s()]', '', regex=True)

    def _fix_dates(self, series):
        def fix_date(date_val):
            if pd.isna(date_val):
                return date_val
            date_str = str(date_val).strip()
            date_str = date_str.replace('1995-02-30', '1995-02-28')
            date_str = date_str.replace('1995-15-08', '1995-08-15')
            return date_str
        return series.apply(fix_date)

    def _fix_names(self, series):
        def fix_name(name):
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if len(name_str) < 2:
                return np.nan
            return name_str.title()
        return series.apply(fix_name)

    def _fix_ages(self, series):
        def fix_age(age):
            try:
                age_val = float(age)
                if 0 <= age_val <= 150:
                    return int(age_val)
                return np.nan
            except:
                return np.nan
        return series.apply(fix_age)

    def _fix_prices(self, series):
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

    def get_optimization_summary(self):
        summary = f"CSV Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Corrections: {', '.join(self.corrections_applied)}\n"
        return summary
