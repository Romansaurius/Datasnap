"""
PERFECT FINAL PARSER
Parser SQL perfecto que mantiene estructura original
"""

import pandas as pd
import re

class PerfectFinalParser:
    """Parser SQL perfecto final"""
    
    def parse_sql_content(self, content: str) -> pd.DataFrame:
        """Parsea SQL complejo manteniendo estructura original"""
        
        try:
            # Limpiar contenido de forma más robusta
            content = self._clean_sql_content(content)
            
            # Extraer CREATE TABLE para obtener columnas
            table_schemas = self._extract_table_schemas(content)
            
            # Extraer INSERT statements con manejo complejo
            all_data = self._extract_insert_data(content, table_schemas)
            
            if all_data:
                return pd.DataFrame(all_data)
            else:
                # Fallback: intentar parseo simple
                return self._simple_fallback_parse(content)
                
        except Exception as e:
            print(f"Error en parser complejo: {e}")
            return self._simple_fallback_parse(content)
    
    def _clean_sql_content(self, content: str) -> str:
        """Limpia contenido SQL de forma robusta"""
        # Reemplazos básicos de HTML entities
        content = content.replace('&#39;', "'")
        content = content.replace('&quot;', '"')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('\r\n', '\n')
        content = content.replace('\r', '\n')
        
        # Remover comentarios
        content = re.sub(r'--.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        return content
    
    def _extract_table_schemas(self, content: str) -> dict:
        """Extrae esquemas de tablas de forma robusta"""
        table_schemas = {}
        
        # Patrón mejorado para CREATE TABLE
        create_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(([^;]+?)\);'
        create_matches = re.findall(create_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_name, columns_def in create_matches:
            columns = []
            
            # Dividir por comas, pero respetando paréntesis
            column_parts = self._split_respecting_parentheses(columns_def, ',')
            
            for part in column_parts:
                part = part.strip()
                if part and not any(keyword in part.upper() for keyword in 
                                  ['PRIMARY KEY', 'FOREIGN KEY', 'REFERENCES', 'CONSTRAINT', 'INDEX']):
                    # Extraer nombre de columna
                    col_match = re.match(r'\s*([`"\[]?\w+[`"\]]?)\s+', part)
                    if col_match:
                        col_name = col_match.group(1).strip('`"[]')
                        if col_name and not col_name.isdigit():
                            columns.append(col_name)
            
            table_schemas[table_name.lower()] = columns
        
        return table_schemas
    
    def _extract_insert_data(self, content: str, table_schemas: dict) -> list:
        """Extrae datos INSERT de forma robusta para SQL complejo"""
        all_data = []
        
        # Dividir contenido en bloques INSERT individuales
        insert_blocks = re.split(r'(?=INSERT\s+INTO)', content, flags=re.IGNORECASE)
        
        for block in insert_blocks:
            if not block.strip() or not block.upper().startswith('INSERT'):
                continue
            
            # Extraer tabla y columnas de cada bloque
            table_match = re.search(r'INSERT\s+INTO\s+(\w+)(?:\s*\(([^)]+)\))?', block, re.IGNORECASE)
            if not table_match:
                continue
            
            table_name = table_match.group(1).lower()
            
            # Si hay columnas especificadas en el INSERT, usarlas
            if table_match.group(2):
                columns = [col.strip() for col in table_match.group(2).split(',')]
            else:
                # Usar esquema de CREATE TABLE si existe
                columns = table_schemas.get(table_name, [])
            
            # Buscar la sección VALUES
            values_match = re.search(r'VALUES\s*(.*?)(?=;|$)', block, re.IGNORECASE | re.DOTALL)
            if not values_match:
                continue
            
            values_section = values_match.group(1)
            
            # Extraer filas de valores
            rows = self._extract_value_rows_complex(values_section)
            
            for row_values in rows:
                if len(row_values) > 0:
                    # Crear registro con mapeo correcto de columnas
                    if len(row_values) > 0:
                        row_dict = {}
                        # Mapear valores a columnas correctamente
                        for i, value in enumerate(row_values):
                            if i < len(columns):
                                row_dict[columns[i]] = value
                            # No crear columnas extra si hay más valores que columnas
                        
                        # Solo agregar si tiene datos válidos
                        if row_dict:
                            row_dict['_table_type'] = table_name
                            all_data.append(row_dict)
        
        return all_data
    
    def _extract_value_rows_complex(self, values_section: str) -> list:
        """Extrae filas de valores para SQL complejo"""
        rows = []
        
        # Limpiar la sección de valores
        values_section = values_section.strip()
        
        # Buscar patrones de filas entre paréntesis, manejando casos complejos
        current_row = ""
        paren_count = 0
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(values_section):
            char = values_section[i]
            
            if char in ["'", '"'] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                # Verificar si es escape
                if i + 1 < len(values_section) and values_section[i + 1] == quote_char:
                    current_row += char + char
                    i += 1
                else:
                    in_quotes = False
                    quote_char = None
            elif char == '(' and not in_quotes:
                paren_count += 1
                if paren_count == 1:
                    current_row = ""
                    i += 1
                    continue
            elif char == ')' and not in_quotes:
                paren_count -= 1
                if paren_count == 0:
                    # Procesar fila completa
                    if current_row.strip():
                        row_values = self._parse_row_values(current_row)
                        if row_values:
                            rows.append(row_values)
                    current_row = ""
                    i += 1
                    continue
            
            if paren_count > 0:
                current_row += char
            
            i += 1
        
        return rows
    
    def _split_respecting_parentheses(self, text: str, delimiter: str) -> list:
        """Divide texto respetando paréntesis"""
        parts = []
        current = ""
        paren_count = 0
        
        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == delimiter and paren_count == 0:
                parts.append(current)
                current = ""
                continue
            current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _extract_value_rows(self, values_block: str) -> list:
        """Extrae filas de valores de forma robusta (método simple)"""
        rows = []
        
        # Buscar patrones de valores entre paréntesis
        row_pattern = r'\(([^)]+)\)'
        row_matches = re.findall(row_pattern, values_block)
        
        for row_str in row_matches:
            values = self._parse_row_values(row_str)
            if values:
                rows.append(values)
        
        return rows
    
    def _simple_fallback_parse(self, content: str) -> pd.DataFrame:
        """Parser simple como fallback para casos complejos"""
        try:
            # Buscar cualquier INSERT simple
            insert_pattern = r'INSERT\s+INTO\s+(\w+).*?VALUES\s*\(([^)]+)\)'
            matches = re.findall(insert_pattern, content, re.IGNORECASE | re.DOTALL)
            
            all_data = []
            for table_name, values_str in matches:
                values = [v.strip().strip("'\"") for v in values_str.split(',')]
                row_dict = {f'col_{i+1}': val for i, val in enumerate(values)}
                row_dict['_table_type'] = table_name.lower()
                all_data.append(row_dict)
            
            if all_data:
                return pd.DataFrame(all_data)
            else:
                return pd.DataFrame({'content': [content[:500]]})
                
        except Exception:
            return pd.DataFrame({'error': ['Complex SQL parsing failed']})
    
    def _parse_row_values(self, row_str: str) -> list:
        """Parsea valores de una fila de forma robusta"""
        values = []
        current = ""
        in_single_quotes = False
        in_double_quotes = False
        escape_next = False
        
        for i, char in enumerate(row_str):
            if escape_next:
                current += char
                escape_next = False
                continue
            
            if char == '\\' and (in_single_quotes or in_double_quotes):
                escape_next = True
                current += char
                continue
            
            if char == "'" and not in_double_quotes:
                in_single_quotes = not in_single_quotes
                current += char
            elif char == '"' and not in_single_quotes:
                in_double_quotes = not in_double_quotes
                current += char
            elif char == ',' and not in_single_quotes and not in_double_quotes:
                val = self._clean_value(current)
                values.append(val)
                current = ""
            else:
                current += char
        
        # Último valor
        if current:
            val = self._clean_value(current)
            values.append(val)
        
        return values
    
    def _clean_value(self, value: str):
        """Limpia y convierte un valor"""
        value = value.strip()
        
        # Remover comillas externas
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        
        # Convertir valores especiales
        if value.upper() == 'TRUE':
            return True
        elif value.upper() == 'FALSE':
            return False
        elif value.upper() == 'NULL':
            return None
        elif value == '':
            return None
        else:
            return value
