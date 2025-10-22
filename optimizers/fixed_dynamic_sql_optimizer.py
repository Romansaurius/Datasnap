#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador SQL Dinámico CORREGIDO para DataSnap IA
Arregla el problema de mezclar columnas entre tablas
"""

import re
import html
import pandas as pd
from io import StringIO
from typing import Dict, List, Tuple, Any

class FixedDynamicSQLOptimizer:
    def __init__(self):
        self.corrections_applied = []
        self.tables_data = {}
        
    def optimize_sql(self, sql_content: str) -> str:
        """Optimización SQL dinámica CORREGIDA"""
        
        # 1. Fix HTML entities
        sql_content = self.fix_html_entities(sql_content)
        
        # 2. Parse existing SQL structure CORRECTAMENTE
        self.parse_sql_structure_correctly(sql_content)
        
        # 3. Clean and validate data
        self.clean_data()
        
        # 4. Generate optimized SQL SIN MEZCLAR TABLAS
        return self.generate_clean_sql()
    
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
    
    def parse_sql_structure_correctly(self, content: str):
        """Parse SQL CORRECTAMENTE - cada tabla por separado"""
        
        # Dividir el contenido por CREATE TABLE
        table_blocks = re.split(r'CREATE\s+TABLE\s+', content, flags=re.IGNORECASE)
        
        for block in table_blocks[1:]:  # Skip first empty block
            # Extract table name and definition
            lines = block.strip().split('\n')
            if not lines:
                continue
                
            # Get table name from first line
            first_line = lines[0]
            table_match = re.match(r'(\w+)\s*\(', first_line)
            if not table_match:
                continue
                
            table_name = table_match.group(1)
            
            # Find the CREATE TABLE block end
            create_block = []
            in_create = True
            paren_count = 0
            
            for line in lines:
                if in_create:
                    create_block.append(line)
                    paren_count += line.count('(') - line.count(')')
                    if paren_count <= 0 and ');' in line:
                        in_create = False
                        break
            
            # Parse columns from CREATE block
            create_text = '\n'.join(create_block)
            columns = self.extract_columns_from_create(create_text)
            
            # Find INSERT statements for this specific table in the entire content
            insert_pattern = rf'INSERT\s+INTO\s+{re.escape(table_name)}(?:\s*\([^)]+\))?\s+VALUES\s*(.*?);'
            insert_matches = re.findall(insert_pattern, content, re.IGNORECASE | re.DOTALL)
            
            data = []
            for values_str in insert_matches:
                values = self.parse_insert_values(values_str)
                data.extend(values)
            
            self.tables_data[table_name] = {
                'columns': columns,
                'data': data
            }
        
        self.corrections_applied.append(f"Parsed {len(self.tables_data)} tables correctly")
    
    def extract_columns_from_create(self, create_text: str) -> List[Dict]:
        """Extract columns from CREATE TABLE statement"""
        columns = []
        
        # Remove CREATE TABLE line and closing parenthesis
        lines = create_text.split('\n')[1:]  # Skip first line with table name
        
        for line in lines:
            line = line.strip().rstrip(',').rstrip(');')
            if not line or line.upper().startswith('PRIMARY KEY') or line.upper().startswith('FOREIGN KEY'):
                continue
            
            # Extract column name and type
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('`"')
                col_type = parts[1]
                
                if col_name and col_name.replace('_', '').replace('-', '').isalnum():
                    columns.append({
                        'name': col_name,
                        'type': col_type,
                        'original_def': line
                    })
        
        return columns
    
    def parse_insert_values(self, values_str: str) -> List[List]:
        """Parse INSERT VALUES correctly"""
        values = []
        
        # Find all value tuples
        tuple_pattern = r'\(([^)]+)\)'
        tuples = re.findall(tuple_pattern, values_str)
        
        for tuple_str in tuples:
            row = []
            # Split by comma, handling quoted strings
            parts = self.split_values_safely(tuple_str)
            
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
    
    def split_values_safely(self, text: str) -> List[str]:
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
                
                # Only keep rows that have at least one meaningful value (not just IDs)
                meaningful_values = []
                for i, v in enumerate(cleaned_row):
                    if v is not None and str(v).strip() not in ['', 'NULL']:
                        # Skip pure ID columns for meaningful count
                        col_name = table_info['columns'][i]['name'].lower() if i < len(table_info['columns']) else ''
                        if not (col_name.endswith('_id') or col_name == 'id'):
                            meaningful_values.append(v)
                
                # Keep row if it has at least 1 meaningful value OR 2+ total values
                total_non_null = sum(1 for v in cleaned_row if v is not None and str(v).strip() not in ['', 'NULL'])
                if len(meaningful_values) >= 1 or total_non_null >= 2:
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
        self.corrections_applied.append("PRIMARY KEYS and FOREIGN KEYS detected automatically")
    
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
            return value_str
        
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
        
        # Stock/Quantity cleaning (incluye numero_camas)
        elif 'stock' in col_name or 'cantidad' in col_name or 'numero' in col_name:
            try:
                stock = int(float(value_str))
                if stock < 0:
                    return abs(stock)  # Convertir negativo a positivo
                return stock
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
        elif 'activo' in col_name or 'active' in col_name or 'vip' in col_name:
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
    
    def generate_clean_sql(self) -> str:
        """Generate clean SQL WITHOUT mixing tables"""
        
        sql_parts = [
            "-- Base de datos optimizada por DataSnap IA",
            "-- Aplicando normalización dinámica y corrección de datos",
            "-- PRIMARY KEYS y FOREIGN KEYS detectadas automáticamente",
            ""
        ]
        
        # Detectar relaciones entre tablas
        foreign_keys = self._detect_foreign_keys()
        
        # Generate CREATE TABLE statements - EACH TABLE SEPARATELY
        for table_name, table_info in self.tables_data.items():
            sql_parts.append(f"-- Tabla: {table_name}")
            sql_parts.append(f"CREATE TABLE {table_name} (")
            
            col_defs = []
            primary_key_added = False
            primary_key_col = None
            
            # Buscar la mejor columna para PRIMARY KEY
            for col in table_info['columns']:
                if self._is_primary_key(col['name'], table_name) and not primary_key_added:
                    primary_key_col = col
                    primary_key_added = True
                    break
            
            # Add ONLY the columns that belong to this table
            for col in table_info['columns']:
                col_name = col['name']
                optimized_type = self.optimize_column_type(col, table_info['data'])
                
                # Solo la primera columna ID será PRIMARY KEY
                if col == primary_key_col:
                    col_defs.append(f"    {col_name} {optimized_type} PRIMARY KEY AUTO_INCREMENT")
                else:
                    col_defs.append(f"    {col_name} {optimized_type}")
            
            # Si no hay PRIMARY KEY, añadir id auto-incremental
            if not primary_key_added:
                col_defs.insert(0, "    id INT PRIMARY KEY AUTO_INCREMENT")
            
            # Añadir FOREIGN KEYS detectadas
            table_fks = foreign_keys.get(table_name, [])
            for fk in table_fks:
                col_defs.append(f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['references_table']}({fk['references_column']})")
            
            sql_parts.append(",\n".join(col_defs))
            sql_parts.append(");")
            sql_parts.append("")
        
        # Generate INSERT statements - ONLY with data that belongs to each table
        for table_name, table_info in self.tables_data.items():
            if table_info['data']:
                sql_parts.append(f"-- Datos para {table_name}")
                
                # Get column names (add id if we added it)
                col_names = []
                has_id = any(col['name'].lower() in ['id', f'{table_name}_id'] for col in table_info['columns'])
                if not has_id:
                    col_names.append('id')
                
                for col in table_info['columns']:
                    col_names.append(col['name'])
                
                sql_parts.append(f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES")
                
                value_rows = []
                for idx, row in enumerate(table_info['data']):
                    # Only skip completely empty rows
                    non_null_values = [v for v in row if v is not None and str(v).strip() not in ['', 'NULL']]
                    if len(non_null_values) == 0:
                        continue  # Skip completely empty rows only
                    
                    formatted_values = []
                    
                    # Add ID if we added it
                    if not has_id:
                        formatted_values.append(str(idx + 1))
                    
                    # Add actual data
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
                
                if value_rows:  # Only add INSERT if there are valid rows
                    sql_parts.append(",\n".join(value_rows) + ";")
                else:
                    sql_parts.append("-- No valid data to insert")
                sql_parts.append("")
        
        return "\n".join(sql_parts)
    
    def optimize_column_type(self, col_info: Dict, table_data: List) -> str:
        """Optimize column type based on data"""
        col_name = col_info['name'].lower()
        
        # Email columns
        if 'email' in col_name or 'mail' in col_name:
            return 'VARCHAR(100) UNIQUE'
        
        # Age columns
        elif 'edad' in col_name or 'age' in col_name:
            return 'INT CHECK (edad >= 0 AND edad <= 120)'
        
        # Money columns - usar el nombre correcto de la columna
        elif any(word in col_name for word in ['salario', 'salary']):
            return f'DECIMAL(10,2) CHECK ({col_info["name"]} >= 0)'
        elif any(word in col_name for word in ['precio', 'price', 'costo', 'cost']):
            return f'DECIMAL(10,2) CHECK ({col_info["name"]} >= 0)'
        
        # Stock columns
        elif 'stock' in col_name or 'cantidad' in col_name:
            return f'INT CHECK ({col_info["name"]} >= 0)'
        
        # Date columns
        elif 'fecha' in col_name or 'date' in col_name:
            return 'DATE'
        
        # Boolean columns
        elif 'activo' in col_name or 'active' in col_name or 'vip' in col_name:
            return 'BOOLEAN DEFAULT TRUE'
        
        # Phone columns
        elif 'telefono' in col_name or 'phone' in col_name:
            return 'VARCHAR(20)'
        
        # ID columns (PRIMARY KEY o FOREIGN KEY)
        elif col_name.endswith('_id') or col_name == 'id':
            return 'INT'
        
        # Text columns with specific sizes
        elif any(word in col_name for word in ['nombre', 'name', 'titulo', 'title']):
            return 'VARCHAR(100)'
        elif any(word in col_name for word in ['descripcion', 'description', 'observaciones', 'notas']):
            return 'TEXT'
        elif any(word in col_name for word in ['codigo', 'code']):
            return 'VARCHAR(50)'
        elif any(word in col_name for word in ['direccion', 'address']):
            return 'VARCHAR(200)'
        
        # Default text columns
        else:
            return 'VARCHAR(100)'
    
    def _is_primary_key(self, col_name: str, table_name: str) -> bool:
        """Detecta si una columna es PRIMARY KEY"""
        col_lower = col_name.lower()
        
        # Patrones comunes de PRIMARY KEY
        pk_patterns = [
            'id',
            f'{table_name}_id',
            f'{table_name[:-1]}_id' if table_name.endswith('s') else f'{table_name}_id',
            'codigo',
            'clave'
        ]
        
        return col_lower in pk_patterns
    
    def _detect_foreign_keys(self) -> Dict[str, List[Dict]]:
        """Detecta FOREIGN KEYS automáticamente"""
        foreign_keys = {}
        
        # Obtener todas las tablas y sus posibles PKs
        table_pks = {}
        for table_name in self.tables_data.keys():
            for col in self.tables_data[table_name]['columns']:
                if self._is_primary_key(col['name'], table_name):
                    table_pks[table_name] = col['name']
                    break
            # Si no hay PK explícita, asumir 'id'
            if table_name not in table_pks:
                table_pks[table_name] = 'id'
        
        # Detectar FKs por patrones de nombres
        for table_name, table_info in self.tables_data.items():
            fks = []
            
            for col in table_info['columns']:
                col_name = col['name'].lower()
                
                # Buscar patrones de FK
                for ref_table, ref_pk in table_pks.items():
                    if ref_table != table_name:
                        # Patrones comunes de FK
                        fk_patterns = [
                            f'{ref_table}_id',
                            f'{ref_table[:-1]}_id' if ref_table.endswith('s') else f'{ref_table}_id',
                            f'id_{ref_table}',
                            f'codigo_{ref_table}'
                        ]
                        
                        # También buscar patrones específicos del dominio médico
                        if ref_table == 'pacientes' and 'paciente' in col_name:
                            fk_patterns.append('paciente_id')
                        elif ref_table == 'medicos' and ('doctor' in col_name or 'medico' in col_name):
                            fk_patterns.extend(['doctor_id', 'medico_id'])
                        elif ref_table == 'hospitales' and 'hospital' in col_name:
                            fk_patterns.append('hospital_id')
                        elif ref_table == 'citas_medicas' and 'cita' in col_name:
                            fk_patterns.append('cita_id')
                        
                        if col_name in fk_patterns:
                            fks.append({
                                'column': col['name'],
                                'references_table': ref_table,
                                'references_column': ref_pk
                            })
            
            if fks:
                foreign_keys[table_name] = fks
        
        return foreign_keys
    
    def get_optimization_summary(self) -> str:
        """Get summary of optimizations applied"""
        foreign_keys = self._detect_foreign_keys()
        total_fks = sum(len(fks) for fks in foreign_keys.values())
        
        summary = f"SQL Optimization Summary:\n"
        summary += f"- Tables processed: {len(self.tables_data)}\n"
        summary += f"- PRIMARY KEYS detected: {len(self.tables_data)}\n"
        summary += f"- FOREIGN KEYS detected: {total_fks}\n"
        summary += f"- Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary

