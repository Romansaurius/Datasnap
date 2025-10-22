"""
SQL NORMALIZER FIXED
Aplica reglas de normalización 1NF, 2NF, 3NF automáticamente
"""

import pandas as pd
import re
from typing import Dict, List, Tuple

class SQLNormalizer:
    """Normalizador automático de bases de datos SQL"""
    
    def __init__(self):
        self.normalized_tables = {}
        self.relationships = []
    
    def normalize_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Aplica normalización completa 1NF -> 2NF -> 3NF"""
        
        try:
            # 1. Primera Forma Normal (1NF)
            df_1nf = self._apply_1nf(df)
            
            # 2. Crear tablas normalizadas (2NF y 3NF)
            normalized_tables = self._create_normalized_tables(df_1nf)
            
            return normalized_tables
        except Exception as e:
            # Si falla la normalización, devolver tabla original
            return {'main': df}
    
    def _apply_1nf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Primera Forma Normal: Eliminar valores multivaluados"""
        
        df_1nf = df.copy()
        
        # Detectar columnas con valores separados por comas
        for col in df_1nf.columns:
            if df_1nf[col].dtype == 'object':
                # Buscar valores con comas (multivaluados)
                multi_values = df_1nf[col].astype(str).str.contains(',', na=False)
                
                if multi_values.any():
                    # Expandir filas con valores múltiples
                    expanded_rows = []
                    
                    for idx, row in df_1nf.iterrows():
                        if pd.notna(row[col]) and ',' in str(row[col]):
                            # Dividir valores múltiples
                            values = str(row[col]).split(',')
                            for value in values:
                                new_row = row.copy()
                                new_row[col] = value.strip()
                                expanded_rows.append(new_row)
                        else:
                            expanded_rows.append(row)
                    
                    df_1nf = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        return df_1nf
    
    def _create_normalized_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Crea tablas normalizadas (2NF y 3NF)"""
        
        tables = {}
        
        # Detectar clave primaria
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        primary_key = id_cols[0] if id_cols else 'id'
        
        # Si no hay columna ID, crearla
        if primary_key not in df.columns:
            df = df.copy()
            df.insert(0, 'id', range(1, len(df) + 1))
            primary_key = 'id'
        
        # 1. Tabla de usuarios/empleados
        user_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['nombre', 'email', 'edad', 'salario'])]
        if user_cols:
            user_table = df[[primary_key] + user_cols].drop_duplicates()
            if len(user_table) > 0:
                tables['usuarios'] = user_table
        
        # 2. Tabla de ubicaciones (2NF)
        location_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['ciudad', 'estado', 'codigo_postal'])]
        if location_cols:
            location_table = df[location_cols].drop_duplicates().reset_index(drop=True)
            if len(location_table) > 0:
                location_table.insert(0, 'ubicacion_id', range(1, len(location_table) + 1))
                tables['ubicaciones'] = location_table
        
        # 3. Tabla de departamentos (2NF)
        dept_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['departamento', 'jefe'])]
        if dept_cols:
            dept_table = df[dept_cols].drop_duplicates().reset_index(drop=True)
            if len(dept_table) > 0:
                dept_table.insert(0, 'departamento_id', range(1, len(dept_table) + 1))
                tables['departamentos'] = dept_table
        
        # 4. Tabla de contactos
        contact_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['telefono', 'phone', 'movil', 'casa'])]
        if contact_cols:
            contact_table = df[[primary_key] + contact_cols].drop_duplicates()
            if len(contact_table) > 0:
                tables['contactos'] = contact_table
        
        # 5. Aplicar 3NF - Separar estados si hay dependencia transitiva
        if 'ubicaciones' in tables:
            tables = self._apply_3nf_to_locations(tables)
        
        # 6. Tabla principal simplificada
        if len(tables) > 1:
            main_table = df[[primary_key]].drop_duplicates()
            tables['main'] = main_table
        else:
            tables['main'] = df
        
        return tables
    
    def _apply_3nf_to_locations(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Aplica 3NF a la tabla de ubicaciones"""
        
        location_df = tables['ubicaciones']
        
        # Si hay ciudad, estado y código postal, separar estados (3NF)
        if 'ciudad' in location_df.columns and 'estado' in location_df.columns:
            state_cols = ['estado']
            if 'codigo_postal' in location_df.columns:
                state_cols.append('codigo_postal')
            
            # Crear tabla de estados
            state_table = location_df[state_cols].drop_duplicates().reset_index(drop=True)
            if len(state_table) > 1:  # Solo si hay múltiples estados
                state_table.insert(0, 'estado_id', range(1, len(state_table) + 1))
                tables['estados'] = state_table
                
                # Simplificar tabla de ubicaciones
                location_simple = location_df[['ubicacion_id', 'ciudad']].copy()
                tables['ubicaciones'] = location_simple
        
        return tables
    
    def generate_normalized_sql(self, tables: Dict[str, pd.DataFrame], original_table_name: str = 'data') -> str:
        """Genera SQL normalizado con CREATE TABLE e INSERT statements"""
        
        sql_parts = []
        sql_parts.append(f"-- Base de datos normalizada (1NF, 2NF, 3NF) desde {original_table_name}")
        sql_parts.append("-- Generado automaticamente por DataSnap IA")
        sql_parts.append("")
        
        # Generar CREATE TABLE statements
        for table_name, df in tables.items():
            sql_parts.append(f"-- Tabla: {table_name}")
            sql_parts.append(f"CREATE TABLE {table_name} (")
            
            # Generar definiciones de columnas
            col_defs = []
            for col in df.columns:
                col_type = self._infer_sql_type(df[col])
                
                # Detectar clave primaria
                if 'id' in col.lower() and col == df.columns[0]:
                    col_defs.append(f"    {col} {col_type} PRIMARY KEY")
                else:
                    col_defs.append(f"    {col} {col_type}")
            
            sql_parts.append(",\n".join(col_defs))
            sql_parts.append(");")
            sql_parts.append("")
        
        # Generar INSERT statements
        for table_name, df in tables.items():
            if not df.empty:
                sql_parts.append(f"-- Datos para tabla: {table_name}")
                cols_str = f"({', '.join(df.columns)})"
                sql_parts.append(f"INSERT INTO {table_name} {cols_str} VALUES")
                
                # Generar valores
                value_rows = []
                for _, row in df.iterrows():
                    values = []
                    for val in row:
                        if pd.isna(val):
                            values.append('NULL')
                        elif isinstance(val, str):
                            escaped_val = val.replace("'", "''")
                            values.append(f"'{escaped_val}'")
                        else:
                            values.append(str(val))
                    value_rows.append(f"({', '.join(values)})")
                
                sql_parts.append(',\n'.join(value_rows) + ';')
                sql_parts.append("")
        
        return '\n'.join(sql_parts)
    
    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infiere el tipo SQL apropiado"""
        
        if pd.api.types.is_integer_dtype(series):
            return 'INT'
        elif pd.api.types.is_float_dtype(series):
            return 'DECIMAL(10,2)'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'DATE'
        elif pd.api.types.is_bool_dtype(series):
            return 'BOOLEAN'
        else:
            # Para strings, determinar longitud apropiada
            max_len = series.astype(str).str.len().max()
            if pd.isna(max_len) or max_len <= 50:
                return 'VARCHAR(50)'
            elif max_len <= 255:
                return 'VARCHAR(255)'
            else:
                return 'TEXT'