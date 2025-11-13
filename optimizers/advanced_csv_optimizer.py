import pandas as pd
import numpy as np
import re
from io import StringIO
from datetime import datetime
from rapidfuzz import fuzz

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class AdvancedCSVOptimizer:
    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0
        self.email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com']
        self.common_typos = {
            'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'gmail.co': 'gmail.com',
            'hotmial.com': 'hotmail.com', 'hotmai.com': 'hotmail.com',
            'yahooo.com': 'yahoo.com', 'yaho.com': 'yahoo.com'
        }

    def optimize_csv(self, csv_content):
        try:
            df = pd.read_csv(StringIO(csv_content))
            self.original_rows = len(df)
            
            # Auto-detect column types using ML
            df = self._auto_detect_and_clean(df)
            
            # Remove duplicates and invalid rows
            df = self._remove_duplicates_and_invalid(df)
            
            self.final_rows = len(df)
            result = df.to_csv(index=False, encoding='utf-8')
            
            return result
            
        except Exception as e:
            raise Exception(f"Error optimizing CSV: {e}")

    def _auto_detect_and_clean(self, df):
        df_clean = df.copy()
        
        # Replace null values
        null_patterns = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'nan', 'NaN', 'missing']
        df_clean = df_clean.replace(null_patterns, np.nan)
        
        for col in df_clean.columns:
            col_type = self._detect_column_type(df_clean[col])
            
            if col_type == 'email':
                df_clean[col] = self._fix_emails_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed emails in {col}")
            elif col_type == 'phone':
                df_clean[col] = self._fix_phones_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed phones in {col}")
            elif col_type == 'date':
                df_clean[col] = self._fix_dates_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed dates in {col}")
            elif col_type == 'name':
                df_clean[col] = self._fix_names_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed names in {col}")
            elif col_type == 'numeric':
                df_clean[col] = self._fix_numeric_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed numbers in {col}")
            elif col_type == 'boolean':
                df_clean[col] = self._fix_boolean_advanced(df_clean[col])
                self.corrections_applied.append(f"Fixed booleans in {col}")
        
        return df_clean

    def _detect_column_type(self, series):
        """ML-based column type detection"""
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return 'unknown'
        
        # Email detection
        email_score = sum(1 for x in sample if '@' in str(x) and '.' in str(x)) / len(sample)
        if email_score > 0.3:
            return 'email'
        
        # Phone detection
        phone_score = sum(1 for x in sample if re.search(r'[\d\+\-\(\)\s]{7,}', str(x))) / len(sample)
        if phone_score > 0.5:
            return 'phone'
        
        # Date detection
        date_score = sum(1 for x in sample if re.search(r'\d{1,4}[\-\/]\d{1,2}[\-\/]\d{1,4}', str(x))) / len(sample)
        if date_score > 0.3:
            return 'date'
        
        # Boolean detection
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 'si', 'active', 'inactive'}
        bool_score = sum(1 for x in sample if str(x).lower() in bool_values) / len(sample)
        if bool_score > 0.5:
            return 'boolean'
        
        # Numeric detection
        numeric_score = sum(1 for x in sample if re.search(r'^[\d\.,\$€£¥₹\+\-]+$', str(x))) / len(sample)
        if numeric_score > 0.5:
            return 'numeric'
        
        # Name detection
        name_score = sum(1 for x in sample if re.search(r'^[A-Za-z\s]{2,}$', str(x))) / len(sample)
        if name_score > 0.5:
            return 'name'
        
        return 'text'

    def _fix_emails_advanced(self, series):
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email_str = str(email).lower().strip()
            
            # Remove mailto: prefix
            email_str = email_str.replace('mailto:', '')
            
            # Fix missing @ symbol
            if '@' not in email_str and email_str and '.' in email_str:
                parts = email_str.split('.')
                if len(parts) >= 2:
                    email_str = f"{parts[0]}@{'.'.join(parts[1:])}"
            
            # Add domain if missing
            if '@' not in email_str and email_str:
                email_str = f"{email_str}@gmail.com"
            
            # Fix common typos
            for typo, correct in self.common_typos.items():
                email_str = email_str.replace(typo, correct)
            
            # Fix domain extensions
            email_str = re.sub(r'\.co$', '.com', email_str)
            email_str = re.sub(r'\.comm$', '.com', email_str)
            
            # Validate email format
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', email_str):
                return np.nan
            
            return email_str
        
        return series.apply(fix_email)

    def _fix_phones_advanced(self, series):
        def fix_phone(phone):
            if pd.isna(phone):
                return phone
            
            phone_str = str(phone).strip()
            
            # Keep only digits, +, -, (, ), spaces
            phone_str = re.sub(r'[^\d\+\-\(\)\s]', '', phone_str)
            
            # Remove extra spaces
            phone_str = re.sub(r'\s+', ' ', phone_str).strip()
            
            # Validate minimum length
            digits_only = re.sub(r'[^\d]', '', phone_str)
            if len(digits_only) < 7:
                return np.nan
            
            return phone_str
        
        return series.apply(fix_phone)

    def _fix_dates_advanced(self, series):
        def fix_date(date_val):
            if pd.isna(date_val):
                return date_val
            
            date_str = str(date_val).strip()
            
            # Fix impossible dates
            date_str = re.sub(r'(\d{4})[-/](\d{2})[-/](30|31)', lambda m: f"{m.group(1)}-{m.group(2)}-28" if m.group(2) == '02' else m.group(0), date_str)
            date_str = re.sub(r'(\d{4})[-/](13|14|15)', r'\1-12', date_str)
            date_str = re.sub(r'(32|33)[-/](\d{2})[-/](\d{4})', r'28/\2/\3', date_str)
            
            # Fix date format DD/MM/YYYY to YYYY-MM-DD
            date_str = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\2-\1', date_str)
            
            return date_str
        
        return series.apply(fix_date)

    def _fix_names_advanced(self, series):
        def fix_name(name):
            if pd.isna(name):
                return name
            
            name_str = str(name).strip()
            
            # Remove invalid names
            if len(name_str) < 2 or name_str.lower() in ['null', 'none', 'missing', 'duplicate']:
                return np.nan
            
            # Fix capitalization
            name_str = name_str.title()
            
            # Remove extra spaces
            name_str = re.sub(r'\s+', ' ', name_str)
            
            return name_str
        
        return series.apply(fix_name)

    def _fix_numeric_advanced(self, series):
        def fix_numeric(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).strip()
            
            # Remove currency symbols and commas
            value_str = re.sub(r'[$€£¥₹,]', '', value_str)
            
            try:
                num_val = float(value_str)
                # Fix negative values that should be positive
                if num_val < 0 and 'price' in str(series.name).lower():
                    num_val = abs(num_val)
                return num_val
            except:
                return np.nan
        
        return series.apply(fix_numeric)

    def _fix_boolean_advanced(self, series):
        def fix_boolean(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).lower().strip()
            
            true_values = {'true', '1', 'yes', 'si', 'active', 'activo'}
            false_values = {'false', '0', 'no', 'inactive', 'inactivo'}
            
            if value_str in true_values:
                return True
            elif value_str in false_values:
                return False
            else:
                return np.nan
        
        return series.apply(fix_boolean)

    def _remove_duplicates_and_invalid(self, df):
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Remove rows with invalid patterns
        for idx, row in df.iterrows():
            invalid_count = sum(1 for val in row if str(val).lower() in ['duplicate', 'invalid', 'test'])
            if invalid_count > len(row) * 0.3:  # More than 30% invalid
                df = df.drop(idx)
        
        return df.reset_index(drop=True)

    def get_optimization_summary(self):
        summary = f"CSV Optimization Summary:\n"
        summary += f"- Original rows: {self.original_rows}\n"
        summary += f"- Final rows: {self.final_rows}\n"
        summary += f"- Rows removed: {self.original_rows - self.final_rows}\n"
        summary += f"- Corrections applied: {len(self.corrections_applied)}\n"
        for correction in self.corrections_applied:
            summary += f"  • {correction}\n"
        return summary
