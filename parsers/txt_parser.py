import pandas as pd
import shutil
import os
from optimizers.universal_global_ai import UniversalGlobalAI

def process_txt(ruta_archivo, historial_folder):
    # Leer como texto para IA Global
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    # Usar IA GLOBAL UNIVERSAL
    global_ai = UniversalGlobalAI()
    optimized_data = global_ai.process_any_data(txt_content)
    
    shutil.copy(ruta_archivo, os.path.join(historial_folder, os.path.basename(ruta_archivo)))
    return optimized_data
