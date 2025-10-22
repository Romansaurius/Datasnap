"""
SIMPLE SQL PARSER
Parser SQL simple que NO mezcla columnas
"""

import pandas as pd
import re
from typing import Dict, List

class SimpleSQLParser:
    """Parser SQL simple que respeta la estructura original"""
    
    def __init__(self):
        self.table_data = {}
        
    def parse_sql_content(self, content: str) -> pd.DataFrame:
        """Parsea contenido SQL respetando columnas originales"""
        
        try:
            # Limpiar contenido
            content = content.replace('&#39;', "'")
            content = content.replace('&quot;', '"')
            
            # Buscar INSERT statements con columnas
            pattern = r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)'
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            all_data = []
            
            for match in matches:
                table_name = match[0].lower()
                columns_str = match[1]
                values_str = match[2]
                
                # Parsear columnas
                columns = [col.strip() for col in columns_str.split(',')]
                
                # Parsear valores
                values = []
                current = ""
                in_quotes = False
                
                for char in values_str:
                    if char == "'" and not in_quotes:
                        in_quotes = True
                    elif char == "'" and in_quotes:
                        in_quotes = False
                    elif char == ',' and not in_quotes:
                        values.append(current.strip().strip("'"))
                        current = ""
                        continue
                    current += char
                
                if current:
                    values.append(current.strip().strip("'"))
                
                # Crear registro
                if len(columns) == len(values):
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col.strip()] = values[i] if values[i] != 'NULL' else None
                    row_dict['_table_type'] = table_name
                    all_data.append(row_dict)
            
            if all_data:
                return pd.DataFrame(all_data)
            else:
                return pd.DataFrame({'error': ['No valid data found']})
                
        except Exception as e:
            return pd.DataFrame({'error': [str(e)]})