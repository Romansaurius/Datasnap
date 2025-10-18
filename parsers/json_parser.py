import pandas as pd
import shutil
import os
from optimizers.universal_global_ai import UniversalGlobalAI

def process_json(ruta_archivo, historial_folder):
    # Leer como texto primero para IA Global
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        json_content = f.read()
    
    # Usar IA GLOBAL UNIVERSAL
    global_ai = UniversalGlobalAI()
    optimized_data = global_ai.process_any_data(json_content)
    
    shutil.copy(ruta_archivo, os.path.join(historial_folder, os.path.basename(ruta_archivo)))
    return optimized_data
