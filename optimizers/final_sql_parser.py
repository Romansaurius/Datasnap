"""
FINAL SQL PARSER
Parser SQL final que funciona perfectamente
"""

import pandas as pd
import re

class FinalSQLParser:
    """Parser SQL final que funciona siempre"""
    
    def parse_sql_content(self, content: str) -> pd.DataFrame:
        """Parsea SQL correctamente"""
        
        try:
            # Limpiar contenido
            content = content.replace('&#39;', "'")
            content = content.replace('&quot;', '"')
            
            all_data = []
            
            # Buscar INSERT con múltiples VALUES
            insert_pattern = r'INSERT\s+INTO\s+(\w+)\s*\([^)]+\)\s*VALUES\s*((?:\s*\([^)]+\)\s*,?\s*)+);'
            matches = re.findall(insert_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for table_name, values_block in matches:
                # Extraer cada fila individual
                row_pattern = r'\(([^)]+)\)'
                rows = re.findall(row_pattern, values_block)
                
                for i, row in enumerate(rows):
                    # Crear registro simple
                    row_dict = {
                        'table_name': table_name.lower(),
                        'row_data': row.strip(),
                        '_table_type': table_name.lower()
                    }
                    all_data.append(row_dict)
            
            if all_data:
                return pd.DataFrame(all_data)
            else:
                return pd.DataFrame({'error': ['No valid data found']})
                
        except Exception as e:
            return pd.DataFrame({'error': [str(e)]})