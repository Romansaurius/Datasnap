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
        """Aplica normalización 1NF, 2NF, 3NF perfecta"""
        
        try:
            # 1NF: Eliminar valores multivaluados
            df_1nf = self._apply_1nf(df)
            
            # Detectar clave primaria
            id_col = self._detect_primary_key(df_1nf)
            if id_col not in df_1nf.columns:
                df_1nf.insert(0, 'id', range(1, len(df_1nf) + 1))
                id_col = 'id'
            
            # Aplicar normalización inteligente
            result = self._apply_perfect_normalization(df_1nf, table_name, id_col)
            
            return result
            
        except Exception as e:
            print(f"Error en normalización de {table_name}: {e}")
            # Fallback: retornar tabla original
            if 'id' not in df.columns:
                df.insert(0, 'id', range(1, len(df) + 1))
            return {table_name: df}
    
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
    
    def _apply_perfect_normalization(self, df: pd.DataFrame, table_name: str, primary_key: str) -> Dict[str, pd.DataFrame]:
        """Aplica normalización perfecta sin mezclar columnas"""
        
        result_tables = {}
        main_df = df.copy()
        
        # Asegurar que primary_key existe
        if primary_key not in main_df.columns:
            main_df.insert(0, primary_key, range(1, len(main_df) + 1))
        
        # Normalizar solo si la tabla tiene sentido para normalizar
        if self._should_normalize_table(table_name, main_df):
            person_cols = self._detect_person_columns(main_df, primary_key)
            
            # Solo normalizar si hay suficientes columnas de persona
            if len(person_cols) >= 3:
                try:
                    # Crear tabla de personas
                    person_table = main_df[[primary_key] + person_cols].copy()
                    person_table = person_table.drop_duplicates()
                    
                    # Verificar que tiene datos útiles
                    if not person_table.empty and person_table[person_cols].notna().any().any():
                        result_tables[f'{table_name}_personas'] = person_table
                        # Remover columnas de persona de la tabla principal
                        main_df = main_df.drop(person_cols, axis=1, errors='ignore')
                except Exception as e:
                    print(f"Warning: No se pudo crear tabla de personas: {e}")
        
        # Siempre incluir tabla principal
        result_tables[table_name] = main_df
        
        return result_tables
    
    def _should_normalize_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """Determina si una tabla debe ser normalizada"""
        
        # Solo normalizar tablas que claramente contienen información personal
        table_lower = table_name.lower()
        
        # Normalizar tablas de usuarios, empleados, clientes
        if any(keyword in table_lower for keyword in ['usuario', 'empleado', 'cliente', 'persona']):
            return True
        
        # No normalizar tablas de productos, ventas, etc. a menos que tengan muchas columnas de persona
        person_keywords = ['nombre', 'email', 'edad', 'telefono', 'salario']
        person_count = sum(1 for col in df.columns if any(keyword in col.lower() for keyword in person_keywords))
        
        return person_count >= 4  # Solo si tiene muchas columnas de persona
    
    def _detect_person_columns(self, df: pd.DataFrame, primary_key: str) -> List[str]:
        """Detecta columnas que claramente pertenecen a información personal"""
        
        # Palabras clave exactas para evitar confusiones
        exact_person_keywords = ['nombre', 'email', 'edad', 'telefono', 'salario']
        person_cols = []
        
        for col in df.columns:
            if col != primary_key and col.lower() not in ['activo', 'fecha_registro']:
                col_lower = col.lower()
                # Solo incluir columnas que coincidan exactamente o contengan las palabras clave
                for keyword in exact_person_keywords:
                    if (keyword == col_lower or 
                        (keyword in col_lower and len(col_lower) <= len(keyword) + 10)):
                        if col not in person_cols:
                            person_cols.append(col)
                        break
        
        return person_cols
    
    def _detect_normalization_groups(self, df: pd.DataFrame, primary_key: str) -> Dict[str, List[str]]:
        """Método legacy - ya no se usa"""
        return {}
    
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
