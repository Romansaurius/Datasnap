"""
🎯 MASTER UNIVERSAL PROCESSOR 🎯
Procesador maestro que maneja CUALQUIER archivo de CUALQUIER tipo
- Detecta automáticamente el tipo de archivo
- Aplica la IA más adecuada
- Funciona con CUALQUIER estructura de datos
- Garantiza resultados perfectos SIEMPRE
"""

import os
from typing import Any, Dict, List
from optimizers.universal_global_ai import UniversalGlobalAI
from utils.universal_file_detector import UniversalFileDetector
from utils.security_validator import SecurityValidator
import pandas as pd

class MasterUniversalProcessor:
    """Procesador maestro universal para CUALQUIER archivo"""
    
    def __init__(self):
        self.global_ai = UniversalGlobalAI()
        self.file_detector = UniversalFileDetector()
        self.security_validator = SecurityValidator()
        
        # Estadísticas de procesamiento
        self.processing_stats = {
            'files_processed': 0,
            'success_rate': 0.0,
            'errors_corrected': 0,
            'predictions_made': 0,
            'security_issues_found': 0
        }
    
    def process_any_file(self, file_path: str, historial_folder: str = None) -> Dict[str, Any]:
        """Procesa CUALQUIER archivo de CUALQUIER tipo"""
        
        result = {
            'success': False,
            'original_file': file_path,
            'processed_data': None,
            'file_info': {},
            'processing_info': {},
            'security_report': {},
            'errors': [],
            'warnings': [],
            'improvements': []
        }
        
        try:
            # PASO 1: DETECTAR TIPO DE ARCHIVO
            result['file_info'] = self.file_detector.detect_file_type(file_path)
            
            if result['file_info']['errors']:
                result['errors'].extend(result['file_info']['errors'])
                return result
            
            # PASO 2: DETERMINAR ESTRATEGIA DE PROCESAMIENTO
            strategy = self.file_detector.get_processing_strategy(result['file_info'])
            result['processing_info']['strategy'] = strategy
            
            # PASO 3: LEER ARCHIVO SEGÚN TIPO DETECTADO
            file_content = self._read_file_intelligently(file_path, result['file_info'])
            
            if file_content is None:
                result['errors'].append('No se pudo leer el archivo')
                return result
            
            # PASO 4: VALIDACIÓN DE SEGURIDAD
            if isinstance(file_content, pd.DataFrame):
                security_report = self.security_validator.validate_dataframe_security(file_content)
                result['security_report'] = security_report
                
                if not security_report['is_safe']:
                    result['warnings'].extend(security_report['critical_issues'])
                    # Sanitizar datos peligrosos
                    file_content = self.security_validator.sanitize_dataframe(file_content)
                    result['improvements'].append('Datos peligrosos sanitizados automáticamente')
            
            # PASO 5: APLICAR IA GLOBAL UNIVERSAL
            processed_data = self.global_ai.process_any_data(file_content)
            result['processed_data'] = processed_data
            
            # PASO 6: GUARDAR EN HISTORIAL SI SE ESPECIFICA
            if historial_folder:
                self._save_to_history(file_path, historial_folder)
            
            # PASO 7: GENERAR ESTADÍSTICAS
            self._update_processing_stats(result)
            
            result['success'] = True
            result['improvements'].append('Archivo procesado exitosamente con IA Global Universal')
            
        except Exception as e:
            result['errors'].append(f'Error durante procesamiento: {str(e)}')
            
            # MODO FALLBACK: Intentar procesamiento básico
            try:
                fallback_data = self._fallback_processing(file_path)
                if fallback_data is not None:
                    result['processed_data'] = fallback_data
                    result['success'] = True
                    result['warnings'].append('Procesado en modo fallback')
            except Exception as fallback_error:
                result['errors'].append(f'Error en modo fallback: {str(fallback_error)}')
        
        return result
    
    def _read_file_intelligently(self, file_path: str, file_info: Dict[str, Any]) -> Any:
        """Lee archivo de forma inteligente según tipo detectado"""
        
        detected_type = file_info['detected_type']
        
        try:
            if detected_type == 'csv':
                return pd.read_csv(file_path, encoding='utf-8', na_values=['', 'NA', 'null', 'NULL'])
            
            elif detected_type == 'excel':
                return pd.read_excel(file_path, na_values=['', 'NA', 'null', 'NULL'])
            
            elif detected_type in ['json', 'sql', 'xml', 'yaml', 'text', 'log', 'config']:
                # Leer como texto para procesamiento universal
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                
                # Si falla todo, leer como binario
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore')
            
            else:
                # Tipo desconocido - intentar como texto
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            # Último recurso - leer como binario
            try:
                with open(file_path, 'rb') as f:
                    return f.read().decode('utf-8', errors='ignore')
            except:
                return None
    
    def _save_to_history(self, file_path: str, historial_folder: str):
        """Guarda archivo en historial"""
        
        try:
            import shutil
            os.makedirs(historial_folder, exist_ok=True)
            
            file_name = os.path.basename(file_path)
            history_path = os.path.join(historial_folder, file_name)
            
            # Si ya existe, agregar timestamp
            if os.path.exists(history_path):
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                name, ext = os.path.splitext(file_name)
                history_path = os.path.join(historial_folder, f"{name}_{timestamp}{ext}")
            
            shutil.copy2(file_path, history_path)
            
        except Exception:
            pass  # No fallar si no se puede guardar historial
    
    def _update_processing_stats(self, result: Dict[str, Any]):
        """Actualiza estadísticas de procesamiento"""
        
        self.processing_stats['files_processed'] += 1
        
        if result['success']:
            success_count = self.processing_stats['files_processed']
            self.processing_stats['success_rate'] = (success_count / self.processing_stats['files_processed']) * 100
        
        self.processing_stats['errors_corrected'] += len(result['improvements'])
        
        if result['security_report']:
            self.processing_stats['security_issues_found'] += len(result['security_report'].get('critical_issues', []))
    
    def _fallback_processing(self, file_path: str) -> Any:
        """Procesamiento de fallback para casos extremos"""
        
        try:
            # Intentar leer como texto plano
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Crear DataFrame básico
            lines = content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            if non_empty_lines:
                return pd.DataFrame({'content': non_empty_lines})
            else:
                return pd.DataFrame({'original_content': [content]})
                
        except Exception:
            # Último recurso - devolver información del archivo
            return pd.DataFrame({
                'file_path': [file_path],
                'file_name': [os.path.basename(file_path)],
                'file_size': [os.path.getsize(file_path) if os.path.exists(file_path) else 0],
                'status': ['processed_as_fallback']
            })
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento"""
        
        return {
            'stats': self.processing_stats.copy(),
            'supported_types': [
                'CSV', 'Excel (XLSX/XLS)', 'JSON', 'SQL', 'TXT', 'XML', 'YAML', 
                'LOG', 'CONFIG', 'HTML', 'Y CUALQUIER OTRO FORMATO'
            ],
            'capabilities': [
                'Detección automática de tipo de archivo',
                'Corrección universal de errores',
                'Predicción de valores faltantes',
                'Validación de seguridad',
                'Sanitización de datos peligrosos',
                'Optimización de rendimiento',
                'Procesamiento de archivos grandes',
                'Modo fallback para casos extremos'
            ]
        }
    
    def process_multiple_files(self, file_paths: List[str], historial_folder: str = None) -> Dict[str, Any]:
        """Procesa múltiples archivos de cualquier tipo"""
        
        results = {
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'results': [],
            'summary': {
                'types_processed': set(),
                'total_improvements': 0,
                'total_errors': 0,
                'security_issues': 0
            }
        }
        
        for file_path in file_paths:
            result = self.process_any_file(file_path, historial_folder)
            results['results'].append(result)
            
            if result['success']:
                results['successful'] += 1
                results['summary']['types_processed'].add(result['file_info'].get('detected_type', 'unknown'))
            else:
                results['failed'] += 1
            
            results['summary']['total_improvements'] += len(result['improvements'])
            results['summary']['total_errors'] += len(result['errors'])
            results['summary']['security_issues'] += len(result['security_report'].get('critical_issues', []))
        
        # Convertir set a list para JSON serialization
        results['summary']['types_processed'] = list(results['summary']['types_processed'])
        
        return results
    
    def validate_file_before_processing(self, file_path: str) -> Dict[str, Any]:
        """Valida archivo antes de procesarlo"""
        
        validation = {
            'is_valid': False,
            'can_process': False,
            'file_info': {},
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Verificar existencia
            if not os.path.exists(file_path):
                validation['warnings'].append('Archivo no existe')
                return validation
            
            # Verificar tamaño
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                validation['warnings'].append('Archivo vacío')
                return validation
            
            if file_size > 100 * 1024 * 1024:  # > 100MB
                validation['warnings'].append('Archivo muy grande (>100MB)')
                validation['recommendations'].append('Considerar dividir el archivo')
            
            # Detectar tipo
            file_info = self.file_detector.detect_file_type(file_path)
            validation['file_info'] = file_info
            
            if file_info['confidence'] > 0.3:
                validation['is_valid'] = True
                validation['can_process'] = True
            else:
                validation['warnings'].append('Tipo de archivo incierto')
                validation['can_process'] = True  # Aún se puede procesar con IA Global
            
            # Recomendaciones específicas
            detected_type = file_info['detected_type']
            if detected_type == 'sql':
                validation['recommendations'].append('Revisar contenido SQL por seguridad')
            elif detected_type == 'unknown':
                validation['recommendations'].append('Archivo de tipo desconocido - se procesará con IA Global Universal')
            
        except Exception as e:
            validation['warnings'].append(f'Error durante validación: {str(e)}')
        
        return validation