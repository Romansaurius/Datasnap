import pandas as pd
import re
import numpy as np
from utils.cleaning_utils import limpiar_dataframe

def parse_sql_to_dataframes(sql_content):
    """
    Parsea contenido SQL y extrae DataFrames por tabla desde INSERT statements usando regex.
    """
    dataframes = {}
    queries = {'select': [], 'update': [], 'delete': []}

    # Regex para INSERT INTO table (cols) VALUES (vals), (vals);
    insert_pattern = r'INSERT INTO\s+`?(\w+)`?\s*\(([^)]+)\)\s*VALUES\s*((?:\([^)]+\)(?:\s*,\s*)*)+);'
    matches = re.findall(insert_pattern, sql_content, re.IGNORECASE | re.DOTALL)

    for match in matches:
        table = match[0]
        cols_str = match[1]
        values_str = match[2]

        columns = [col.strip('` ') for col in cols_str.split(',')]

        # Parse values
        values = []
        val_pattern = r'\(([^)]+)\)'
        val_matches = re.findall(val_pattern, values_str)
        for val in val_matches:
            # Split by comma but ignore commas inside single quotes
            row = [v.strip("'\" ") for v in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", val)]
            if len(row) == len(columns):
                values.append(row)
            else:
                # Skip invalid rows
                continue

        if values:
            df = pd.DataFrame(values, columns=columns)
            dataframes[table] = df

    # Extraer queries con regex simple
    if 'SELECT' in sql_content.upper():
        queries['select'].append('SELECT statement found')
    if 'UPDATE' in sql_content.upper():
        queries['update'].append('UPDATE statement found')
    if 'DELETE' in sql_content.upper():
        queries['delete'].append('DELETE statement found')

    return dataframes, queries

def process_sql(ruta_archivo, historial_folder):
    """Procesa archivos SQL usando el optimizador perfecto"""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Importar aquí para evitar dependencias circulares
        from optimizers.perfect_sql_optimizer import PerfectSQLOptimizer
        
        sql_optimizer = PerfectSQLOptimizer()
        optimization_result = sql_optimizer.optimize_sql(sql_content)
        
        # Guardar en historial
        import os
        import shutil
        shutil.copy(ruta_archivo, os.path.join(historial_folder, os.path.basename(ruta_archivo)))

        return optimization_result.optimized_sql
    except Exception as e:
        # En caso de error, devolver el contenido original
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        return sql_content