"""
🔍 UNIVERSAL FILE DETECTOR 🔍
Detector universal que identifica CUALQUIER tipo de archivo y contenido
"""

import os
import re
import json
from typing import Dict, Any, Tuple

class UniversalFileDetector:
    """Detector universal de archivos y contenido"""
    
    def __init__(self):
        # Patrones de detección de contenido
        self.content_patterns = {
            'sql': [
                r'CREATE\s+TABLE',
                r'INSERT\s+INTO',
                r'SELECT\s+.*\s+FROM',
                r'UPDATE\s+.*\s+SET',
                r'DELETE\s+FROM',
                r'DROP\s+TABLE',
                r'ALTER\s+TABLE'
            ],
            'json': [
                r'^\s*[\{\[]',
                r'\"[^\"]*\"\s*:\s*[^,\}]+',
                r'^\s*\{.*\}\s*$'
            ],
            'csv': [
                r'^[^,\n]*,[^,\n]*',
                r'\n[^,\n]*,[^,\n]*',
                r'^\"[^\"]*\",\"[^\"]*\"'
            ],
            'xml': [
                r'<\?xml',
                r'<[^>]+>.*</[^>]+>',
                r'<[^/>]+/>'
            ],
            'html': [
                r'<!DOCTYPE\s+html',
                r'<html[^>]*>',
                r'<head[^>]*>',
                r'<body[^>]*>'
            ],
            'yaml': [
                r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:',
                r'^\s*-\s+[a-zA-Z_]',
                r'---\s*$'
            ],
            'log': [
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
                r'\[.*\]\s+.*',
                r'ERROR|WARN|INFO|DEBUG'
            ],
            'config': [
                r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=',
                r'^\s*\[[^\]]+\]',
                r'#.*$'
            ]
        }
        
        # Extensiones conocidas
        self.known_extensions = {
            '.csv': 'csv',
            '.json': 'json',
            '.sql': 'sql',
            '.txt': 'text',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.xml': 'xml',
            '.html': 'html',
            '.htm': 'html',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.log': 'log',
            '.conf': 'config',
            '.cfg': 'config',
            '.ini': 'config'
        }
    
    def detect_file_type(self, file_path: str) -> Dict[str, Any]:
        """Detecta el tipo de archivo y su contenido"""
        
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'extension': '',
            'detected_type': 'unknown',
            'content_type': 'unknown',
            'confidence': 0.0,
            'file_size': 0,
            'is_readable': False,
            'encoding': 'utf-8',
            'structure_info': {},
            'errors': []
        }
        
        try:
            # Información básica del archivo
            if os.path.exists(file_path):
                result['file_size'] = os.path.getsize(file_path)
                result['extension'] = os.path.splitext(file_path)[1].lower()
                
                # Detectar por extensión
                if result['extension'] in self.known_extensions:
                    result['detected_type'] = self.known_extensions[result['extension']]
                    result['confidence'] = 0.7
                
                # Leer contenido para análisis
                content = self._read_file_safely(file_path)
                if content:
                    result['is_readable'] = True
                    
                    # Detectar por contenido
                    content_detection = self._detect_by_content(content)
                    if content_detection['type'] != 'unknown':
                        result['content_type'] = content_detection['type']
                        result['confidence'] = max(result['confidence'], content_detection['confidence'])
                        
                        # Si el contenido contradice la extensión, usar contenido
                        if content_detection['confidence'] > 0.8:
                            result['detected_type'] = content_detection['type']
                    
                    # Analizar estructura
                    result['structure_info'] = self._analyze_structure(content, result['detected_type'])
                
            else:
                result['errors'].append('Archivo no encontrado')
                
        except Exception as e:
            result['errors'].append(f'Error al analizar archivo: {str(e)}')
        
        return result
    
    def _read_file_safely(self, file_path: str) -> str:
        """Lee archivo de forma segura probando diferentes encodings"""
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Limitar tamaño para análisis (primeros 50KB)
                    return content[:50000]
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        
        # Si falla todo, intentar leer como binario
        try:
            with open(file_path, 'rb') as f:
                binary_content = f.read(50000)
                return binary_content.decode('utf-8', errors='ignore')
        except:
            return ''
    
    def _detect_by_content(self, content: str) -> Dict[str, Any]:
        """Detecta tipo por contenido"""
        
        content_lower = content.lower()
        best_match = {'type': 'unknown', 'confidence': 0.0, 'matches': 0}
        
        for content_type, patterns in self.content_patterns.items():
            matches = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    matches += 1
            
            confidence = matches / total_patterns
            
            if confidence > best_match['confidence']:
                best_match = {
                    'type': content_type,
                    'confidence': confidence,
                    'matches': matches
                }
        
        return best_match
    
    def _analyze_structure(self, content: str, detected_type: str) -> Dict[str, Any]:
        """Analiza la estructura del contenido"""
        
        structure = {
            'lines': len(content.split('\n')),
            'characters': len(content),
            'words': len(content.split()),
            'empty_lines': content.count('\n\n'),
            'special_info': {}
        }
        
        if detected_type == 'csv':
            structure['special_info'] = self._analyze_csv_structure(content)
        elif detected_type == 'json':
            structure['special_info'] = self._analyze_json_structure(content)
        elif detected_type == 'sql':
            structure['special_info'] = self._analyze_sql_structure(content)
        elif detected_type == 'xml':
            structure['special_info'] = self._analyze_xml_structure(content)
        
        return structure
    
    def _analyze_csv_structure(self, content: str) -> Dict[str, Any]:
        """Analiza estructura CSV"""
        
        lines = content.split('\n')
        if not lines:
            return {}
        
        # Detectar separador
        separators = [',', ';', '\t', '|']
        separator_counts = {}
        
        first_line = lines[0] if lines else ''
        for sep in separators:
            separator_counts[sep] = first_line.count(sep)
        
        likely_separator = max(separator_counts, key=separator_counts.get)
        
        # Estimar columnas
        estimated_columns = separator_counts[likely_separator] + 1 if separator_counts[likely_separator] > 0 else 1
        
        return {
            'estimated_separator': likely_separator,
            'estimated_columns': estimated_columns,
            'estimated_rows': len([line for line in lines if line.strip()]),
            'has_header': self._likely_has_header(lines, likely_separator)
        }
    
    def _analyze_json_structure(self, content: str) -> Dict[str, Any]:
        """Analiza estructura JSON"""
        
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                return {
                    'type': 'array',
                    'length': len(data),
                    'first_item_type': type(data[0]).__name__ if data else 'empty',
                    'is_uniform': self._is_uniform_array(data)
                }
            elif isinstance(data, dict):
                return {
                    'type': 'object',
                    'keys': list(data.keys())[:10],  # Primeras 10 claves
                    'key_count': len(data.keys()),
                    'nested_levels': self._count_nested_levels(data)
                }
            else:
                return {
                    'type': 'primitive',
                    'value_type': type(data).__name__
                }
                
        except json.JSONDecodeError:
            return {
                'type': 'invalid_json',
                'error': 'JSON malformado'
            }
    
    def _analyze_sql_structure(self, content: str) -> Dict[str, Any]:
        """Analiza estructura SQL"""
        
        content_upper = content.upper()
        
        # Contar statements
        statements = {
            'CREATE_TABLE': len(re.findall(r'CREATE\s+TABLE', content_upper)),
            'INSERT_INTO': len(re.findall(r'INSERT\s+INTO', content_upper)),
            'SELECT': len(re.findall(r'SELECT\s+', content_upper)),
            'UPDATE': len(re.findall(r'UPDATE\s+', content_upper)),
            'DELETE': len(re.findall(r'DELETE\s+FROM', content_upper)),
            'DROP': len(re.findall(r'DROP\s+', content_upper))
        }
        
        # Extraer nombres de tablas
        table_matches = re.findall(r'(?:CREATE\s+TABLE|INSERT\s+INTO|FROM|UPDATE|JOIN)\s+`?(\w+)`?', content_upper)
        unique_tables = list(set(table_matches))
        
        return {
            'statements': statements,
            'total_statements': sum(statements.values()),
            'tables_mentioned': unique_tables[:10],  # Primeras 10 tablas
            'table_count': len(unique_tables),
            'has_data': statements['INSERT_INTO'] > 0,
            'has_structure': statements['CREATE_TABLE'] > 0
        }
    
    def _analyze_xml_structure(self, content: str) -> Dict[str, Any]:
        """Analiza estructura XML"""
        
        # Contar tags
        tag_matches = re.findall(r'<(\w+)', content)
        unique_tags = list(set(tag_matches))
        
        return {
            'unique_tags': unique_tags[:10],
            'tag_count': len(unique_tags),
            'total_elements': len(tag_matches),
            'has_declaration': content.strip().startswith('<?xml'),
            'is_well_formed': self._check_xml_well_formed(content)
        }
    
    def _likely_has_header(self, lines: List[str], separator: str) -> bool:
        """Determina si CSV probablemente tiene header"""
        
        if len(lines) < 2:
            return False
        
        first_row = lines[0].split(separator)
        second_row = lines[1].split(separator) if len(lines) > 1 else []
        
        # Si la primera fila tiene texto y la segunda números, probablemente hay header
        first_has_text = any(not self._is_number(cell.strip().strip('"\'')) for cell in first_row)
        second_has_numbers = any(self._is_number(cell.strip().strip('"\'')) for cell in second_row)
        
        return first_has_text and second_has_numbers
    
    def _is_uniform_array(self, arr: list) -> bool:
        """Verifica si array JSON es uniforme"""
        
        if not arr:
            return True
        
        first_type = type(arr[0])
        return all(type(item) == first_type for item in arr)
    
    def _count_nested_levels(self, obj: dict, level: int = 0) -> int:
        """Cuenta niveles de anidación en JSON"""
        
        max_level = level
        
        for value in obj.values():
            if isinstance(value, dict):
                max_level = max(max_level, self._count_nested_levels(value, level + 1))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                max_level = max(max_level, self._count_nested_levels(value[0], level + 1))
        
        return max_level
    
    def _check_xml_well_formed(self, content: str) -> bool:
        """Verifica si XML está bien formado (básico)"""
        
        try:
            # Contar tags de apertura y cierre
            open_tags = re.findall(r'<(\w+)[^>]*>', content)
            close_tags = re.findall(r'</(\w+)>', content)
            self_closing = re.findall(r'<\w+[^>]*/>', content)
            
            # Básicamente verificar que hay balance
            return len(open_tags) >= len(close_tags)
        except:
            return False
    
    def _is_number(self, text: str) -> bool:
        """Verifica si texto es un número"""
        
        try:
            float(text)
            return True
        except ValueError:
            return False
    
    def get_processing_strategy(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determina la estrategia de procesamiento óptima"""
        
        detected_type = detection_result['detected_type']
        confidence = detection_result['confidence']
        
        strategy = {
            'processor': 'universal_global_ai',
            'approach': 'auto_detect',
            'special_handling': [],
            'risk_level': 'low'
        }
        
        # Ajustar estrategia según tipo y confianza
        if confidence < 0.5:
            strategy['approach'] = 'safe_fallback'
            strategy['risk_level'] = 'high'
            strategy['special_handling'].append('low_confidence_detection')
        
        if detected_type == 'sql':
            strategy['special_handling'].append('sql_injection_check')
            strategy['special_handling'].append('data_extraction')
        
        if detected_type in ['json', 'xml']:
            strategy['special_handling'].append('structure_validation')
        
        if detection_result['file_size'] > 10 * 1024 * 1024:  # > 10MB
            strategy['special_handling'].append('large_file_processing')
            strategy['risk_level'] = 'medium'
        
        return strategy