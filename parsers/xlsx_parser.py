import pandas as pd
import shutil
import os
from optimizers.universal_global_ai import UniversalGlobalAI

def process_xlsx(ruta_archivo, historial_folder):
    df = pd.read_excel(ruta_archivo, na_values=["<NA>", "nan", "NaN", ""])
    
    # Usar IA GLOBAL UNIVERSAL
    global_ai = UniversalGlobalAI()
    df_mejorado = global_ai.process_any_data(df)
    
    shutil.copy(ruta_archivo, os.path.join(historial_folder, os.path.basename(ruta_archivo)))
    return df_mejorado
