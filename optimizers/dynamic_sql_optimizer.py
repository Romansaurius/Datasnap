#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador SQL Dinámico para DataSnap IA
Analiza y normaliza cualquier estructura SQL que suba el usuario
"""

import re
import html
import pandas as pd
from io import StringIO
from typing import Dict, List, Tuple, Any

class DynamicSQLOptimizer:
    def __init__(self):
        self.corrections_applied = []
        self.tables_data = {}
        self.normalized_tables = {}
        
    def optimize_sql(self, sql_content: str) -> str:
        """Optimización SQL dinámica para cualquier estructura"""
        
        # 1. Fix HTML entities
        sql_content = self.fix_html_entities(sql_content)
        
        # 2. Parse existing SQL structure
        self.parse_sql_structure(sql_content)
        
        # 3. Clean and validate data
        self.clean_data()
        
        # 4. Apply dynamic normalization
        self.apply_dynamic_normalization()
        
        # 5. Generate optimized SQL
        return self.generate_optimized_sql()
    
    def fix_html_entities(self, content: str) -> str:
        """Fix HTML entities"""
        content = html.unescape(content)
        content = re.sub(r'&#39;', "'", content)
        content = re.sub(r'&quot;', '"', content)
        content = re.sub(r'&amp;', '&', content)
        content = re.sub(r'&lt;', '<', content)
        content = re.sub(r'&gt;', '>', content)
        
        self.corrections_applied.append("HTML entities fixed")
        return content
    
    def parse_sql_structure(self, content: str):
        """Parse SQL to extract table structures and data"""
        
        # Extract CREATE TABLE statements
        create_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        create_matches = re.findall(create_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_name, columns_def in create_matches:
            self.tables_data[table_name] = {
                'columns': self.parse_columns(columns_def),
                'data': []
            }
        
        # Extract INSERT statements
        insert_pattern = r'INSERT\s+INTO\s+(\w+)(?:\s*\([^)]+\))?\s+VALUES\s*(.*?);'
        insert_matches = re.findall(insert_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_name, values_str in insert_matches:
            if table_name in self.tables_data:
                values = self.parse_insert_values(values_str)
                self.tables_data[table_name]['data'].extend(values)
        
        self.corrections_applied.append(f"Parsed {len(self.tables_data)} tables")
    
    def parse_columns(self, columns_def: str) -> List[Dict]:
        """Parse column definitions"""
        columns = []
        
        # Split by comma, but handle nested parentheses
        parts = self.split_columns(columns_def)
        
        for part in parts:
            part = part.strip()
            if not part or 'PRIMARY KEY' in part.upper() or 'FOREIGN KEY' in part.upper():
                continue
                
            # Extract column name and type
            tokens = part.split()
            if len(tokens) >= 2:
                col_name = tokens[0].strip('`"')
                col_type = tokens[1]
                
                # Skip invalid column names
                if col_name and not col_name.startswith('--') and col_name.replace('_', '').replace('-', '').isalnum():
                    columns.append({
                        'name': col_name,
                        'type': col_type,
                        'original_def': part
                    })
        
        return columns
    
    def split_columns(self, text: str) -> List[str]:
        """Split column definitions by comma, respecting parentheses"""
        parts = []
        current = ""
        paren_count = 0
        
        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                parts.append(current.strip())
                current = ""
                continue
            
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def parse_insert_values(self, values_str: str) -> List[List]:
        """Parse INSERT VALUES"""
        values = []
        
        # Find all value tuples
        tuple_pattern = r'\(([^)]+)\)'
        tuples = re.findall(tuple_pattern, values_str)
        
        for tuple_str in tuples:
            row = []
            # Split by comma, handling quoted strings
            parts = self.split_values(tuple_str)
            
            for part in parts:
                part = part.strip()
                # Remove quotes and handle NULL
                if part.upper() == 'NULL':
                    row.append(None)
                elif part.startswith("'") and part.endswith("'"):
                    row.append(part[1:-1])
                elif part.startswith('"') and part.endswith('"'):
                    row.append(part[1:-1])
                else:
                    row.append(part)
            
            values.append(row)
        
        return values
    
    def split_values(self, text: str) -> List[str]:
        """Split values by comma, respecting quotes"""
        parts = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in text:
            if char in ["'", '"'] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
                continue
            
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def clean_data(self):
        """Clean and validate data in all tables"""
        
        for table_name, table_info in self.tables_data.items():
            cleaned_data = []
            
            for row in table_info['data']:
                cleaned_row = []
                
                for i, value in enumerate(row):
                    if i < len(table_info['columns']):
                        col_info = table_info['columns'][i]
                        cleaned_value = self.clean_value(value, col_info)
                        cleaned_row.append(cleaned_value)
                    else:
                        cleaned_row.append(value)
                
                # Only keep rows that aren't completely empty
                if any(v is not None and str(v).strip() for v in cleaned_row):
                    cleaned_data.append(cleaned_row)
            
            # Remove duplicates
            unique_data = []
            seen = set()
            for row in cleaned_data:
                row_tuple = tuple(str(v) if v is not None else 'NULL' for v in row)
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_data.append(row)
            
            table_info['data'] = unique_data
        
        self.corrections_applied.append("Data cleaned and validated")
    
    def clean_value(self, value, col_info):
        """Clean individual value based on column type and name"""
        if value is None or str(value).strip() == '':
            return None
        
        value_str = str(value).strip()
        col_name = col_info['name'].lower()
        col_type = col_info['type'].upper()
        
        # Email cleaning
        if 'email' in col_name or 'mail' in col_name:
            value_str = value_str.replace('gmai.com', 'gmail.com')
            value_str = value_str.replace('hotmial.com', 'hotmail.com')
            value_str = value_str.replace('yahoo.co', 'yahoo.com')
            
            # Validate email format
            if not re.match(r'^[^@]+@[^@]+\.[^@]+$', value_str):
                return None
        
        # Age cleaning
        elif 'edad' in col_name or 'age' in col_name:
            try:
                age = int(float(value_str))
                if age < 0 or age > 120:
                    return None
                return age
            except:
                return None
        
        # Salary/Price cleaning
        elif any(word in col_name for word in ['salario', 'salary', 'precio', 'price']):
            try:
                # Remove currency symbols
                clean_val = re.sub(r'[$€£¥₹,]', '', value_str)
                amount = float(clean_val)
                if amount < 0:
                    return None
                return amount
            except:
                return None
        
        # Date cleaning
        elif 'fecha' in col_name or 'date' in col_name:
            # Fix common invalid dates
            value_str = value_str.replace('1995-02-30', '1995-02-28')
            value_str = value_str.replace('2023-02-30', '2023-02-28')
            
            if value_str in ['invalid_date', 'ayer', 'hoy', 'mañana']:
                return None
            
            return value_str
        
        # Phone cleaning
        elif 'telefono' in col_name or 'phone' in col_name:
            if value_str == 'invalid_phone':
                return None
            
            # Clean phone format
            clean_phone = re.sub(r'[^\d+\-\s()]', '', value_str)
            if len(clean_phone) < 7 or len(clean_phone) > 20:
                return None
            
            return clean_phone
        
        # Boolean cleaning
        elif 'activo' in col_name or 'active' in col_name:
            value_lower = value_str.lower()
            if value_lower in ['true', '1', 'yes', 'si', 'sí', 'y', 'active', 'activo']:
                return True
            elif value_lower in ['false', '0', 'no', 'n', 'inactive', 'inactivo']:
                return False
            else:
                return None
        
        # Numeric types
        elif 'INT' in col_type or 'DECIMAL' in col_type or 'FLOAT' in col_type:
            try:
                if 'INT' in col_type:
                    return int(float(value_str))
                else:
                    return float(value_str)
            except:
                return None
        
        return value_str
    
    def apply_dynamic_normalization(self):
        """Apply normalization based on detected patterns"""
        
        # Identify potential lookup tables
        lookup_candidates = self.identify_lookup_tables()
        
        # Create normalized structure
        for table_name, table_info in self.tables_data.items():
            normalized_table = {
                'columns': [],
                'data': table_info['data'],
                'foreign_keys': []
            }
            
            # Add ID column if not exists
            has_id = any(col['name'].lower() in ['id', f'{table_name}_id'] for col in table_info['columns'])
            if not has_id:
                normalized_table['columns'].append({
                    'name': 'id',
                    'type': 'INT PRIMARY KEY AUTO_INCREMENT'
                })
            
            # Process each column
            for col in table_info['columns']:
                col_name = col['name'].lower()
                
                # Check if this column should reference a lookup table
                if col_name in lookup_candidates:
                    # Create foreign key reference
                    fk_col = f"{col_name}_id"
                    normalized_table['columns'].append({
                        'name': fk_col,
                        'type': 'INT'
                    })
                    normalized_table['foreign_keys'].append({
                        'column': fk_col,
                        'references': f"{col_name}s(id)"
                    })
                else:
                    # Keep original column with optimized type
                    optimized_type = self.optimize_column_type(col, table_info['data'])
                    normalized_table['columns'].append({
                        'name': col['name'],
                        'type': optimized_type
                    })
            
            self.normalized_tables[table_name] = normalized_table
        
        # Create lookup tables
        for lookup_name in lookup_candidates:
            self.create_lookup_table(lookup_name, lookup_candidates[lookup_name])
        
        self.corrections_applied.append("Dynamic normalization applied")
    
    def identify_lookup_tables(self) -> Dict[str, set]:
        """Identify columns that should become lookup tables"""
        lookup_candidates = {}
        
        for table_name, table_info in self.tables_data.items():
            for i, col in enumerate(table_info['columns']):
                col_name = col['name'].lower()
                
                # Skip certain column types
                if any(skip in col_name for skip in ['id', 'email', 'telefono', 'phone', 'fecha', 'date']):
                    continue
                
                # Get unique values for this column
                values = set()
                for row in table_info['data']:
                    if i < len(row) and row[i] is not None:
                        val = str(row[i]).strip()
                        if val and len(val) < 100:  # Reasonable length for lookup
                            values.add(val)
                
                # If column has repeated values and reasonable count, make it a lookup
                if len(values) > 1 and len(values) < len(table_info['data']) * 0.8:
                    if col_name not in lookup_candidates:
                        lookup_candidates[col_name] = set()
                    lookup_candidates[col_name].update(values)
        
        return lookup_candidates
    
    def create_lookup_table(self, lookup_name: str, values: set):
        """Create a lookup table"""
        table_name = f"{lookup_name}s"
        
        self.normalized_tables[table_name] = {
            'columns': [
                {'name': 'id', 'type': 'INT PRIMARY KEY AUTO_INCREMENT'},
                {'name': 'nombre', 'type': 'VARCHAR(100) NOT NULL UNIQUE'}
            ],
            'data': [[i+1, val] for i, val in enumerate(sorted(values))],
            'foreign_keys': []
        }
    
    def optimize_column_type(self, col_info: Dict, table_data: List) -> str:
        """Optimize column type based on data"""
        col_name = col_info['name'].lower()
        original_type = col_info['type'].upper()
        
        # Get column index
        col_index = None
        for i, c in enumerate(self.tables_data[list(self.tables_data.keys())[0]]['columns']):
            if c['name'] == col_info['name']:
                col_index = i
                break
        
        if col_index is None:
            return original_type
        
        # Analyze actual data
        values = [row[col_index] for row in table_data if col_index < len(row) and row[col_index] is not None]
        
        if not values:
            return original_type
        
        # Email columns
        if 'email' in col_name:
            return 'VARCHAR(100) UNIQUE'
        
        # Age columns
        elif 'edad' in col_name or 'age' in col_name:
            return 'INT CHECK (edad >= 0 AND edad <= 120)'
        
        # Money columns
        elif any(word in col_name for word in ['salario', 'salary', 'precio', 'price']):
            return 'DECIMAL(10,2) CHECK (salario >= 0)'
        
        # Date columns
        elif 'fecha' in col_name or 'date' in col_name:
            return 'DATE'
        
        # Boolean columns
        elif 'activo' in col_name or 'active' in col_name:
            return 'BOOLEAN DEFAULT TRUE'
        
        # Phone columns
        elif 'telefono' in col_name or 'phone' in col_name:
            return 'VARCHAR(20)'
        
        # Auto-detect based on data
        else:
            # Check if all values are numeric
            numeric_values = []
            for val in values:
                try:
                    numeric_values.append(float(val))
                except:
                    break
            
            if len(numeric_values) == len(values):
                # All numeric
                if all(isinstance(v, int) or v.is_integer() for v in numeric_values):
                    max_val = max(numeric_values)
                    if max_val < 128:
                        return 'TINYINT'
                    elif max_val < 32768:
                        return 'SMALLINT'
                    else:
                        return 'INT'
                else:
                    return 'DECIMAL(10,2)'
            else:
                # Text data
                max_len = max(len(str(v)) for v in values)
                if max_len <= 50:
                    return f'VARCHAR(50)'
                elif max_len <= 255:
                    return 'VARCHAR(255)'
                else:
                    return 'TEXT'
    
    def generate_optimized_sql(self) -> str:
        """Generate the final optimized SQL"""
        
        sql_parts = [
            "-- Base de datos optimizada por DataSnap IA",
            "-- Aplicando normalización dinámica y corrección de datos",
            ""
        ]
        
        # Generate CREATE TABLE statements
        for table_name, table_info in self.normalized_tables.items():
            sql_parts.append(f"-- Tabla: {table_name}")
            sql_parts.append(f"CREATE TABLE {table_name} (")
            
            # Add columns
            col_defs = []
            for col in table_info['columns']:
                if col['name'] and not col['name'].startswith('--'):
                    col_defs.append(f"    {col['name']} {col['type']}")
            
            # Add foreign keys
            for fk in table_info['foreign_keys']:
                col_defs.append(f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['references']}")
            
            sql_parts.append(",\n".join(col_defs))
            sql_parts.append(");")
            sql_parts.append("")
        
        # Generate INSERT statements
        for table_name, table_info in self.normalized_tables.items():
            if table_info['data']:
                sql_parts.append(f"-- Datos para {table_name}")
                
                col_names = [col['name'] for col in table_info['columns']]
                sql_parts.append(f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES")
                
                value_rows = []
                for row in table_info['data']:
                    formatted_values = []
                    for val in row:
                        if val is None:
                            formatted_values.append('NULL')
                        elif isinstance(val, str):
                            escaped_val = val.replace("'", "''")
                            formatted_values.append(f"'{escaped_val}'")
                        elif isinstance(val, bool):
                            formatted_values.append('TRUE' if val else 'FALSE')
                        else:
                            formatted_values.append(str(val))
                    
                    value_rows.append(f"({', '.join(formatted_values)})")
                
                sql_parts.append(",\n".join(value_rows) + ";")
                sql_parts.append("")
        
        return "\n".join(sql_parts)
    
    def get_optimization_summary(self) -> str:
        """Get summary of optimizations applied"""
        summary = f"SQL Optimization Summary:\n"
        summary += f"- Tables processed: {len(self.tables_data)}\n"
        summary += f"- Normalized tables: {len(self.normalized_tables)}\n"
        summary += f"- Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary

if __name__ == "__main__":
    optimizer = DynamicSQLOptimizer()
    
    # Test with sample SQL
    test_sql = """
    CREATE TABLE usuarios (
        nombre VARCHAR(100),
        email VARCHAR(100),
        edad INT
    );
    
    INSERT INTO usuarios VALUES
    ('Juan', 'juan@gmai.com', -10),
    ('Maria', 'invalid_email', 200);
    """
    
    result = optimizer.optimize_sql(test_sql)
    print("Optimized SQL:")
    print(result)
    print("\nSummary:")
    print(optimizer.get_optimization_summary())