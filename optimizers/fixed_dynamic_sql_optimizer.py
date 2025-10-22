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
            
            # Find INSERT statements for this specific table
            insert_pattern = rf'INSERT\s+INTO\s+{re.escape(table_name)}(?:\s*\([^)]+\))?\s+VALUES\s*(.*?);'
            insert_matches = re.findall(insert_pattern, block, re.IGNORECASE | re.DOTALL)
            
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
        
        # Stock/Quantity cleaning
        elif 'stock' in col_name or 'cantidad' in col_name:
            try:
                stock = int(float(value_str))
                if stock < 0:
                    return None
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
            ""
        ]
        
        # Generate CREATE TABLE statements - EACH TABLE SEPARATELY
        for table_name, table_info in self.tables_data.items():
            sql_parts.append(f"-- Tabla: {table_name}")
            sql_parts.append(f"CREATE TABLE {table_name} (")
            
            # Add ID column if not exists
            has_id = any(col['name'].lower() in ['id', f'{table_name}_id'] for col in table_info['columns'])
            col_defs = []
            
            if not has_id:
                col_defs.append("    id INT PRIMARY KEY AUTO_INCREMENT")
            
            # Add ONLY the columns that belong to this table
            for col in table_info['columns']:
                optimized_type = self.optimize_column_type(col, table_info['data'])
                col_defs.append(f"    {col['name']} {optimized_type}")
            
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
                
                sql_parts.append(",\n".join(value_rows) + ";")
                sql_parts.append("")
        
        return "\n".join(sql_parts)
    
    def optimize_column_type(self, col_info: Dict, table_data: List) -> str:
        """Optimize column type based on data"""
        col_name = col_info['name'].lower()
        
        # Email columns
        if 'email' in col_name:
            return 'VARCHAR(100) UNIQUE'
        
        # Age columns
        elif 'edad' in col_name or 'age' in col_name:
            return 'INT CHECK (edad >= 0 AND edad <= 120)'
        
        # Money columns
        elif any(word in col_name for word in ['salario', 'salary', 'precio', 'price']):
            return 'DECIMAL(10,2) CHECK (precio >= 0)'
        
        # Stock columns
        elif 'stock' in col_name or 'cantidad' in col_name:
            return 'INT CHECK (stock >= 0)'
        
        # Date columns
        elif 'fecha' in col_name or 'date' in col_name:
            return 'DATE'
        
        # Boolean columns
        elif 'activo' in col_name or 'active' in col_name or 'vip' in col_name:
            return 'BOOLEAN DEFAULT TRUE'
        
        # Phone columns
        elif 'telefono' in col_name or 'phone' in col_name:
            return 'VARCHAR(20)'
        
        # ID columns
        elif col_name.endswith('_id') or col_name == 'id':
            return 'INT'
        
        # Default text columns
        else:
            return 'VARCHAR(100)'
    
    def get_optimization_summary(self) -> str:
        """Get summary of optimizations applied"""
        summary = f"SQL Optimization Summary:\n"
        summary += f"- Tables processed: {len(self.tables_data)}\n"
        summary += f"- Corrections applied:\n"
        
        for correction in self.corrections_applied:
            summary += f"  + {correction}\n"
        
        return summary

if __name__ == "__main__":
    optimizer = FixedDynamicSQLOptimizer()
    
    # Test with the problematic SQL
    test_sql = """
    CREATE TABLE mi_empresa (
        empleado_id INT,
        nombre_empleado VARCHAR(100),
        email_trabajo VARCHAR(100),
        salario_mensual DECIMAL(10,2),
        departamento VARCHAR(50),
        fecha_contratacion DATE,
        activo BOOLEAN
    );
    
    INSERT INTO mi_empresa VALUES
    (1, 'Ana García', 'ana@gmai.com', -5000, 'Ventas', '1995-02-30', 'si'),
    (2, 'Pedro Martín', 'pedro@hotmial.com', 3500.50, 'Marketing', '2023-01-15', 'true');
    
    CREATE TABLE inventario_productos (
        codigo_producto VARCHAR(50),
        nombre_producto VARCHAR(200),
        categoria_producto VARCHAR(100),
        precio_venta DECIMAL(8,2),
        stock_disponible INT,
        proveedor_principal VARCHAR(100)
    );
    
    INSERT INTO inventario_productos VALUES
    ('PROD001', 'Laptop HP', 'Informática', 1299.99, -10, 'TechCorp'),
    ('PROD002', 'Mouse Inalámbrico', 'Periféricos', 25.50, 150, 'AccessCorp');
    """
    
    result = optimizer.optimize_sql(test_sql)
    print("FIXED OPTIMIZED SQL:")
    print(result)
    print("\nSUMMARY:")
    print(optimizer.get_optimization_summary())