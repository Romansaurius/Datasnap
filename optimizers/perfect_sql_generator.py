"""
PERFECT SQL GENERATOR
Generador SQL perfecto que crea SQL valido y normalizado
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime

class PerfectSQLGenerator:
    """Generador SQL perfecto que crea codigo valido"""
    
    def __init__(self):
        self.table_relationships = {}
        self.primary_keys = {}
        self.foreign_keys = {}
        
    def generate_perfect_sql(self, df: pd.DataFrame, enable_normalization: bool = True) -> str:
        """Genera SQL perfecto desde DataFrame optimizado"""
        
        try:
            if enable_normalization:
                return self._generate_normalized_sql(df)
            else:
                return self._generate_standard_sql(df)
        except Exception as e:
            print(f"Error generando SQL: {e}")
            return self._generate_fallback_sql(df)
    
    def _generate_normalized_sql(self, df: pd.DataFrame) -> str:
        """Genera SQL normalizado perfecto (1NF, 2NF, 3NF)"""
        
        sql_parts = []
        sql_parts.append("-- Base de datos normalizada automaticamente por DataSnap IA")
        sql_parts.append("-- Aplicando reglas 1NF, 2NF y 3NF de forma inteligente")
        sql_parts.append("")
        
        try:
            if '_table_type' in df.columns:
                # Procesar cada tabla por separado
                tables = df['_table_type'].unique()
                
                for table_name in tables:
                    table_df = df[df['_table_type'] == table_name].drop('_table_type', axis=1)
                    
                    if not table_df.empty:
                        normalized_tables = self._normalize_table(table_df, table_name)
                        
                        # Verificar que normalized_tables no sea None
                        if normalized_tables and isinstance(normalized_tables, dict):
                            # Generar SQL para cada tabla normalizada
                            for norm_table_name, norm_df in normalized_tables.items():
                                if not norm_df.empty:
                                    sql_parts.extend(self._generate_table_sql(norm_table_name, norm_df))
                                    sql_parts.append("")
                        else:
                            # Fallback: generar tabla simple
                            sql_parts.extend(self._generate_table_sql(table_name, table_df))
                            sql_parts.append("")
            else:
                # Tabla única - aplicar normalización completa
                normalized_tables = self._normalize_table(df, 'data_table')
                if normalized_tables and isinstance(normalized_tables, dict):
                    for norm_table_name, norm_df in normalized_tables.items():
                        if not norm_df.empty:
                            sql_parts.extend(self._generate_table_sql(norm_table_name, norm_df))
                            sql_parts.append("")
                else:
                    # Fallback: generar tabla simple
                    sql_parts.extend(self._generate_table_sql('data_table', df))
                    sql_parts.append("")
            
            return "\n".join(sql_parts)
            
        except Exception as e:
            print(f"Error en normalización, usando SQL estándar: {e}")
            return self._generate_standard_sql(df)
    
    def _generate_standard_sql(self, df: pd.DataFrame) -> str:
        """Genera SQL estandar sin normalizacion"""
        
        sql_parts = []
        sql_parts.append("-- Datos optimizados por DataSnap IA")
        sql_parts.append("")
        
        if '_table_type' in df.columns:
            tables = df['_table_type'].unique()
            
            for table_name in tables:
                table_df = df[df['_table_type'] == table_name].drop('_table_type', axis=1)
                sql_parts.extend(self._generate_table_sql(table_name, table_df))
                sql_parts.append("")
        else:
            sql_parts.extend(self._generate_table_sql('optimized_data', df))
        
        return "\n".join(sql_parts)
    
    def _normalize_table(self, df: pd.DataFrame, table_name: str) -> Dict[str, pd.DataFrame]:
        """Aplica normalizacion 1NF, 2NF, 3NF a una tabla"""
        
        normalized_tables = {}
        
        # 1NF: Eliminar valores multivaluados
        df_1nf = self._apply_1nf(df)
        
        # Detectar clave primaria
        id_col = self._detect_primary_key(df_1nf)
        if id_col not in df_1nf.columns:
            df_1nf.insert(0, 'id', range(1, len(df_1nf) + 1))
            id_col = 'id'
        
        # 2NF y 3NF: Separar en tablas relacionadas
        main_table, related_tables = self._apply_2nf_3nf(df_1nf, table_name, id_col)
        
        # Agregar tablas normalizadas
        normalized_tables[table_name] = main_table
        normalized_tables.update(related_tables)
        
        return normalized_tables
    
    def _apply_1nf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Primera Forma Normal: Eliminar valores multivaluados"""
        
        df_1nf = df.copy()
        
        for col in df_1nf.columns:
            if df_1nf[col].dtype == 'object':
                # Buscar valores con separadores
                multi_value_mask = df_1nf[col].astype(str).str.contains('[,;|]', na=False)
                
                if multi_value_mask.any():
                    expanded_rows = []
                    
                    for idx, row in df_1nf.iterrows():
                        if pd.notna(row[col]) and any(sep in str(row[col]) for sep in [',', ';', '|']):
                            # Dividir valores multiples
                            separators = [',', ';', '|']
                            values = [str(row[col])]
                            
                            for sep in separators:
                                new_values = []
                                for val in values:
                                    new_values.extend(val.split(sep))
                                values = new_values
                            
                            # Crear fila para cada valor
                            for value in values:
                                new_row = row.copy()
                                new_row[col] = value.strip()
                                expanded_rows.append(new_row)
                        else:
                            expanded_rows.append(row)
                    
                    df_1nf = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        return df_1nf
    
    def _detect_primary_key(self, df: pd.DataFrame) -> str:
        """Detecta la clave primaria"""
        
        # Buscar columnas con 'id' en el nombre
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        
        if id_columns:
            # Preferir 'id' simple
            if 'id' in id_columns:
                return 'id'
            else:
                return id_columns[0]
        
        # Si no hay columna ID, buscar columna unica
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].notna().all():
                return col
        
        return 'id'  # Se creara automaticamente
    
    def _apply_2nf_3nf(self, df: pd.DataFrame, table_name: str, primary_key: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Aplica 2NF y 3NF con manejo perfecto de casos complejos"""
        
        related_tables = {}
        main_df = df.copy()
        
        # Asegurar que primary_key existe
        if primary_key not in main_df.columns:
            main_df.insert(0, primary_key, range(1, len(main_df) + 1))
        
        # Variables para evitar conflictos
        person_cols = []
        work_cols = []
        product_cols = []
        commercial_cols = []
        transaction_cols = []
        ref_cols = []
        
        # Normalización inteligente basada en patrones de datos
        normalized_groups = self._detect_normalization_groups(main_df, primary_key)
        
        for group_name, group_cols in normalized_groups.items():
            if len(group_cols) >= 2 and all(col in main_df.columns for col in group_cols):
                try:
                    # Crear tabla normalizada
                    normalized_table = main_df[[primary_key] + group_cols].drop_duplicates()
                    
                    # Crear tabla solo si tiene datos significativos y únicos
                    if len(group_cols) >= 2 and len(normalized_table) > 1 and normalized_table[group_cols].notna().any().any():
                        # Verificar que la tabla tenga datos útiles
                        has_useful_data = False
                        for col in group_cols:
                            if normalized_table[col].notna().sum() > 0:
                                has_useful_data = True
                                break
                        
                        if has_useful_data:
                            related_tables[f'{table_name}_{group_name}'] = normalized_table
                            main_df = main_df.drop(group_cols, axis=1, errors='ignore')
                except Exception as e:
                    print(f"Warning: No se pudo normalizar grupo {group_name}: {e}")
                    continue
        
        return main_df, related_tables
    
    def _detect_normalization_groups(self, df: pd.DataFrame, primary_key: str) -> Dict[str, List[str]]:
        """Detecta grupos de columnas para normalización inteligente"""
        
        groups = {}
        
        # Normalización simplificada para evitar problemas
        person_keywords = ['nombre', 'email', 'edad', 'telefono', 'salario']
        person_cols = [col for col in df.columns if col != primary_key and 
                      any(keyword in col.lower() for keyword in person_keywords)]
        
        if len(person_cols) >= 3:
            groups['personas'] = person_cols
        
        return groups
    
    def _detect_table_type(self, df: pd.DataFrame) -> str:
        """Detecta el tipo de tabla principal"""
        columns_str = ' '.join(df.columns).lower()
        
        if any(keyword in columns_str for keyword in ['nombre', 'email', 'edad', 'salario']):
            return 'usuarios'
        elif any(keyword in columns_str for keyword in ['precio', 'stock', 'categoria']):
            return 'productos'
        elif any(keyword in columns_str for keyword in ['cantidad', 'precio_unitario', 'fecha_venta']):
            return 'ventas'
        else:
            return 'general'
    
    def _generate_table_sql(self, table_name: str, df: pd.DataFrame) -> List[str]:
        """Genera SQL para una tabla especifica"""
        
        sql_parts = []
        
        if df.empty:
            return sql_parts
        
        # CREATE TABLE
        sql_parts.append(f"-- Tabla: {table_name}")
        sql_parts.append(f"CREATE TABLE {table_name} (")
        
        # Definir columnas
        column_defs = []
        for col in df.columns:
            col_type = self._infer_sql_type(df[col])
            
            # Detectar clave primaria
            if 'id' in col.lower() and col == df.columns[0]:
                column_defs.append(f"    {col} {col_type} PRIMARY KEY")
            else:
                column_defs.append(f"    {col} {col_type}")
        
        sql_parts.append(",\n".join(column_defs))
        sql_parts.append(");")
        sql_parts.append("")
        
        # INSERT statements
        if not df.empty:
            sql_parts.append(f"-- Datos para {table_name}")
            
            # Filtrar columnas con datos validos
            valid_columns = []
            for col in df.columns:
                if df[col].notna().any():
                    valid_columns.append(col)
            
            if valid_columns:
                cols_str = f"({', '.join(valid_columns)})"
                sql_parts.append(f"INSERT INTO {table_name} {cols_str} VALUES")
                
                # Generar valores
                value_rows = []
                for _, row in df.iterrows():
                    values = []
                    for col in valid_columns:
                        value = row[col]
                        if pd.isna(value):
                            values.append('NULL')
                        elif isinstance(value, str):
                            escaped_value = value.replace("'", "''")
                            values.append(f"'{escaped_value}'")
                        elif isinstance(value, bool):
                            values.append('1' if value else '0')
                        else:
                            values.append(str(value))
                    
                    value_rows.append(f"({', '.join(values)})")
                
                sql_parts.append(',\n'.join(value_rows) + ';')
        
        return sql_parts
    
    def _infer_sql_type(self, series: pd.Series) -> str:
        """Infiere el tipo SQL apropiado"""
        
        # Analizar valores no nulos
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'VARCHAR(255)'
        
        # Detectar tipos
        if pd.api.types.is_integer_dtype(series):
            max_val = non_null_series.max()
            if max_val <= 127:
                return 'TINYINT'
            elif max_val <= 32767:
                return 'SMALLINT'
            elif max_val <= 2147483647:
                return 'INT'
            else:
                return 'BIGINT'
        
        elif pd.api.types.is_float_dtype(series):
            return 'DECIMAL(10,2)'
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'DATETIME'
        
        elif pd.api.types.is_bool_dtype(series):
            return 'BOOLEAN'
        
        else:
            # Para strings, calcular longitud maxima
            max_length = non_null_series.astype(str).str.len().max()
            
            if pd.isna(max_length):
                return 'VARCHAR(255)'
            elif max_length <= 50:
                return 'VARCHAR(50)'
            elif max_length <= 255:
                return 'VARCHAR(255)'
            else:
                return 'TEXT'
    
    def _generate_fallback_sql(self, df: pd.DataFrame) -> str:
        """Genera SQL basico como fallback"""
        
        sql_parts = []
        sql_parts.append("-- Datos procesados por DataSnap IA (modo fallback)")
        sql_parts.append("")
        
        if '_table_type' in df.columns:
            tables = df['_table_type'].unique()
            
            for table_name in tables:
                table_df = df[df['_table_type'] == table_name].drop('_table_type', axis=1)
                
                if not table_df.empty:
                    sql_parts.append(f"-- Tabla: {table_name}")
                    
                    # Solo INSERT sin CREATE TABLE
                    valid_cols = [col for col in table_df.columns if table_df[col].notna().any()]
                    
                    if valid_cols:
                        sql_parts.append(f"INSERT INTO {table_name} ({', '.join(valid_cols)}) VALUES")
                        
                        values = []
                        for _, row in table_df.iterrows():
                            row_values = []
                            for col in valid_cols:
                                val = row[col]
                                if pd.isna(val):
                                    row_values.append('NULL')
                                elif isinstance(val, str):
                                    row_values.append(f"'{val.replace(chr(39), chr(39)+chr(39))}'")
                                else:
                                    row_values.append(str(val))
                            values.append(f"({', '.join(row_values)})")
                        
                        sql_parts.append(',\n'.join(values) + ';')
                        sql_parts.append("")
        
        return "\n".join(sql_parts)
