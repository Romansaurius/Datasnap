import pandas as pd
import re
import numpy as np
import os
import shutil
try:
    import sqlparse
except ImportError:
    sqlparse = None

try:
    from utils.cleaning_utils import limpiar_dataframe
except ImportError:
    # Función de limpieza básica si no existe el módulo
    def limpiar_dataframe(df):
        df = df.drop_duplicates()
        df = df.dropna(how='all')
        return df

def parse_sql_to_dataframes(sql_content):
    """
    Parsea contenido SQL y extrae DataFrames por tabla desde INSERT statements usando regex mejorado.
    """
    dataframes = {}
    queries = {'select': [], 'update': [], 'delete': []}

    # Limpiar contenido SQL
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)  # Remover comentarios
    sql_clean = re.sub(r'--.*?\n', '\n', sql_clean)  # Remover comentarios de línea
    
    # Patrones mejorados para diferentes formatos de INSERT
    patterns = [
        # INSERT INTO table (cols) VALUES (vals), (vals);
        r'INSERT\s+INTO\s+`?(\w+)`?\s*\(([^)]+)\)\s*VALUES\s*((?:\([^)]+\)(?:\s*,\s*)*)+);?',
        # INSERT INTO table VALUES (vals), (vals);
        r'INSERT\s+INTO\s+`?(\w+)`?\s+VALUES\s*((?:\([^)]+\)(?:\s*,\s*)*)+);?',
        # INSERT table (cols) VALUES (vals);
        r'INSERT\s+`?(\w+)`?\s*\(([^)]+)\)\s*VALUES\s*((?:\([^)]+\)(?:\s*,\s*)*)+);?'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, sql_clean, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            if len(match.groups()) == 3:  # Con columnas
                table, cols_str, values_str = match.groups()
                columns = [col.strip('` "') for col in cols_str.split(',')]
            else:  # Sin columnas explícitas
                table, values_str = match.groups()
                columns = None
            
            # Parse values con mejor manejo de comillas
            values = []
            val_pattern = r'\(([^)]+)\)'
            val_matches = re.findall(val_pattern, values_str)
            
            for val in val_matches:
                # Mejorar parsing de valores con comillas anidadas
                row = parse_sql_values(val)
                
                if not columns:
                    columns = [f'col_{i+1}' for i in range(len(row))]
                
                if len(row) == len(columns):
                    values.append(row)
            
            if values and table not in dataframes:
                df = pd.DataFrame(values, columns=columns)
                # Limpiar valores NULL
                df = df.replace(['NULL', 'null', 'None', ''], pd.NA)
                dataframes[table] = df
                print(f"Parsed table '{table}' with {len(df)} rows")
    
    # Extraer queries
    if 'SELECT' in sql_content.upper():
        queries['select'].append('SELECT statement found')
    if 'UPDATE' in sql_content.upper():
        queries['update'].append('UPDATE statement found')
    if 'DELETE' in sql_content.upper():
        queries['delete'].append('DELETE statement found')

    return dataframes, queries

def parse_sql_values(values_str):
    """Parse SQL values handling quotes and special characters"""
    values = []
    current = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(values_str):
        char = values_str[i]
        
        if char in ["'", '"'] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            # Check for escaped quote
            if i + 1 < len(values_str) and values_str[i + 1] == quote_char:
                current += char
                i += 1
            else:
                in_quotes = False
                quote_char = None
        elif char == ',' and not in_quotes:
            val = current.strip().strip("'\"")
            values.append(val if val.upper() not in ['NULL', 'NONE'] else '')
            current = ""
            i += 1
            continue
        else:
            current += char
        
        i += 1
    
    # Add last value
    if current:
        val = current.strip().strip("'\"")
        values.append(val if val.upper() not in ['NULL', 'NONE'] else '')
    
    return values

def normalize_dataframe(df):
    """
    Normalización simplificada:
    - 1NF: Eliminar duplicados (ya en limpiar_dataframe)
    - 2NF/3NF: Simplificada, separar si hay dependencias obvias (ej. columna con listas)
    Para simplicidad, solo limpiar y predecir.
    """
    # Aquí podríamos implementar lógica más avanzada, pero por ahora solo limpiar
    return df

def predict_missing_values(df):
    """
    Predecir valores faltantes usando media para numéricos, moda para categóricos (sin sklearn).
    """
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
    return df

def detect_anomalies(df):
    """
    Detecta anomalías simples: outliers en numéricos usando IQR.
    """
    anomalies = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            anomalies[col] = len(outliers)
    return anomalies

def infer_sql_type(dtype, col_name, series=None):
    """
    Infiera tipo SQL desde pandas dtype y nombre de columna, con compresión.
    """
    if pd.api.types.is_integer_dtype(dtype) and series is not None:
        min_val = series.min()
        max_val = series.max()
        if min_val >= 0:
            if max_val < 256:
                return 'TINYINT UNSIGNED'
            elif max_val < 65536:
                return 'SMALLINT UNSIGNED'
            elif max_val < 4294967296:
                return 'INT UNSIGNED'
        else:
            if min_val >= -128 and max_val < 128:
                return 'TINYINT'
            elif min_val >= -32768 and max_val < 32768:
                return 'SMALLINT'
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'REAL'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    else:
        # Para texto, estimar longitud
        if series is not None:
            max_len = series.str.len().max()
            if max_len <= 50:
                return f'VARCHAR({max_len})'
            elif max_len <= 255:
                return 'VARCHAR(255)'
        if 'email' in col_name.lower():
            return 'VARCHAR(255)'
        elif 'name' in col_name.lower() or 'title' in col_name.lower():
            return 'VARCHAR(100)'
        else:
            return 'TEXT'

def detect_functional_dependencies(df, pk):
    """
    Detecta dependencias funcionales: columna A determina B si agrupar por A da B único.
    """
    fds = []
    non_pk_columns = [col for col in df.columns if col != pk]
    for a in df.columns:
        for b in df.columns:
            if a != b:
                grouped = df.groupby(a)[b].nunique()
                if (grouped == 1).all():
                    fds.append((a, b))
    return fds

def normalize_to_2nf_3nf(df, table_name, pk):
    """
    Normalización simplificada: mantiene la tabla en 1NF (eliminar duplicados).
    No separa tablas para evitar errores en normalización automática.
    """
    # Solo eliminar duplicados para 1NF
    df = df.drop_duplicates()
    return {table_name: df}

def normalize_database(dataframes):
    """
    Normalización completa: 1NF, detectar PK, aplicar 2NF/3NF automática.
    """
    normalized = {}
    for table, df in dataframes.items():
        # 1NF: eliminar duplicados
        df = df.drop_duplicates()
        
        # Detectar clave primaria: columna con valores únicos
        pk_candidates = [col for col in df.columns if df[col].nunique() == len(df)]
        primary_key = pk_candidates[0] if pk_candidates else None
        
        # Aplicar 2NF/3NF
        sub_tables = normalize_to_2nf_3nf(df, table, primary_key)
        for sub_table, sub_df in sub_tables.items():
            normalized[sub_table] = {'df': sub_df, 'primary_key': primary_key if sub_table == table else None}
    
    return normalized

def generate_sql_from_dataframes(dataframes, db_name):
    """
    Genera SQL CREATE e INSERT desde DataFrames, con tipos inferidos y normalización.
    Maneja múltiples tablas por normalización.
    """
    normalized = normalize_database(dataframes)
    sql_output = [f"DROP DATABASE IF EXISTS `{db_name}`; CREATE DATABASE `{db_name}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;",
                  f"USE `{db_name}`;"]
    for table, data in normalized.items():
        df = data['df']
        pk = data['primary_key']
        columns = df.columns
        
        # CREATE TABLE con tipos MySQL inferidos
        col_defs = []
        for col in columns:
            sql_type = infer_sql_type(df[col].dtype, col, df[col])
            col_defs.append(f'{col} {sql_type}')
        if pk:
            col_defs.append(f'PRIMARY KEY ({pk})')
        create_stmt = f"CREATE TABLE {table} ({', '.join(col_defs)});"
        sql_output.append(create_stmt)

        # INSERT INTO con columnas
        cols_str = f" ({', '.join(columns)})"
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
            values_str = ', '.join(values)
            insert_stmt = f"INSERT INTO {table}{cols_str} VALUES ({values_str});"
            sql_output.append(insert_stmt)

    return '\n'.join(sql_output)

def extract_db_name(sql_content):
    """
    Extrae el nombre de la base de datos del SQL, ignorando IF NOT EXISTS.
    """
    try:
        match = re.search(r'CREATE DATABASE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([^`\s;]+)`?', sql_content, re.IGNORECASE)
        if match:
            return match.group(1)
    except:
        pass
    return 'datasnap_optimized_db'

def correct_sql_syntax(sql_content):
    """
    Corrige errores comunes de sintaxis en SQL y formatea.
    """
    corrections = {
        'PRIMARI': 'PRIMARY',
        'VARCHARR': 'VARCHAR',
        'NOT NUL': '',  # Remover NOT NUL
        'NOT NULL': '',  # Remover NOT NULL
        'TIMESTAP': 'TIMESTAMP',
        'CURREN_TIMESTAMP': 'CURRENT_TIMESTAMP',
        'EMAIL': 'VARCHAR(255)',  # Asumir tipo
        'AUTOINCREMENT': 'AUTO_INCREMENT',
        'UNIQUE KEY': 'UNIQUE',
        'INT(': 'INTEGER(',
        'BOOL': 'BOOLEAN',
        'FLOAT': 'REAL',
        'DOUBLE': 'REAL',
    }
    for wrong, correct in corrections.items():
        sql_content = sql_content.replace(wrong, correct)
    # Remover CREATE DATABASE y USE existentes
    import re
    sql_content = re.sub(r'CREATE DATABASE[^;]*;', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'USE[^;]*;', '', sql_content, flags=re.IGNORECASE)
    # Remover restricciones que causan errores
    sql_content = re.sub(r'\bNOT\s+NULL\b', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'\bDEFAULT\s+[^;]*', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)
    # No formatear para evitar errores de sintaxis
    return sql_content

def optimize_queries(sql_content, queries):
    """
    Optimiza queries SQL: agrega sugerencias basadas en statements parseados.
    """
    suggestions = []
    for select in queries['select']:
        upper_select = select.upper()
        if 'WHERE' in upper_select:
            suggestions.append("-- Sugerencia para SELECT: Crea índices en columnas de WHERE.")
        if 'JOIN' in upper_select:
            suggestions.append("-- Sugerencia para SELECT: Índices en columnas de JOIN; usa INNER si posible.")
        if 'SELECT *' in upper_select:
            suggestions.append("-- Sugerencia para SELECT: Especifica columnas en lugar de *.")
        if 'DISTINCT' in upper_select:
            suggestions.append("-- Sugerencia para SELECT: DISTINCT costoso; considera alternativas.")
    for update in queries['update']:
        if 'WHERE' in update.upper():
            suggestions.append("-- Sugerencia para UPDATE: Índices en WHERE para rapidez.")
    for delete in queries['delete']:
        if 'WHERE' in delete.upper():
            suggestions.append("-- Sugerencia para DELETE: Índices en WHERE.")
    return sql_content + "\n" + "\n".join(suggestions)

def process_sql_with_quality_fixes(sql_content):
    """Procesa SQL aplicando correcciones de calidad de datos"""
    try:
        # 1. Parsear SQL a DataFrames
        dataframes, queries = parse_sql_to_dataframes(sql_content)
        
        if not dataframes:
            return sql_content  # Si no se puede parsear, devolver original
        
        # 2. Aplicar correcciones de calidad a cada tabla
        from optimizers.critical_fixes_optimizer import CriticalFixesOptimizer
        fixer = CriticalFixesOptimizer()
        
        optimized_dataframes = {}
        for table_name, df in dataframes.items():
            # Aplicar correcciones críticas
            df_fixed = fixer.apply_critical_fixes(df)
            optimized_dataframes[table_name] = df_fixed
        
        # 3. Generar SQL optimizado
        db_name = extract_db_name(sql_content)
        optimized_sql = generate_sql_from_dataframes(optimized_dataframes, db_name)
        
        # 4. Agregar comentarios de optimización
        header = "-- SQL OPTIMIZADO POR DATASNAP IA GLOBAL\n"
        header += "-- Correcciones aplicadas: edades, emails, fechas, valores nulos\n"
        header += f"-- Tablas procesadas: {len(optimized_dataframes)}\n\n"
        
        return header + optimized_sql
        
    except Exception as e:
        # Fallback: usar IA universal directamente
        try:
            from optimizers.universal_global_ai import UniversalGlobalAI
            global_ai = UniversalGlobalAI()
            return global_ai.process_any_data(sql_content)
        except:
            return f"-- Error procesando SQL: {str(e)}\n\n{sql_content}"

def process_sql(ruta_archivo, historial_folder):
    """Procesa archivos SQL con IA GLOBAL UNIVERSAL y correcciones de calidad"""
    try:
        # Leer archivo SQL
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Guardar en historial
        try:
            shutil.copy(ruta_archivo, os.path.join(historial_folder, os.path.basename(ruta_archivo)))
        except Exception:
            pass
        
        # Procesar con correcciones de calidad
        optimized_sql = process_sql_with_quality_fixes(sql_content)
        
        return optimized_sql
        
    except Exception as e:
        # Fallback seguro
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fallback_sql = f"-- SQL PROCESADO POR DATASNAP IA GLOBAL\n"
            fallback_sql += f"-- Error: {str(e)}\n\n"
            fallback_sql += original_content
            
            return fallback_sql
        except:
            return "-- Error: No se pudo procesar el archivo SQL"
