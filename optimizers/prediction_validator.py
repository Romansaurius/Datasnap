"""
PREDICTION VALIDATOR
Validador avanzado de predicciones que asegura calidad y coherencia
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime, timedelta

class PredictionValidator:
    """Validador avanzado de predicciones ML"""
    
    def __init__(self):
        self.validation_rules = {}
        self.confidence_scores = {}
        self.refinement_history = []
        
        # Reglas de validacion por tipo de dato
        self.validation_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[\d\s\-\(\)]{7,20}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'number': r'^-?\d+\.?\d*$',
            'boolean': r'^(true|false|1|0)$',
            'name': r'^[A-Za-z\s]{2,50}$'
        }
        
        # Rangos validos por tipo de campo
        self.valid_ranges = {
            'edad': (0, 120),
            'salario': (0, 1000000),
            'precio': (0, 100000),
            'temperatura': (-50, 100),
            'humedad': (0, 100),
            'stock': (0, 10000)
        }
    
    def validate_and_refine_predictions(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida y refina predicciones usando multiples metodos"""
        
        try:
            refined_df = predicted_df.copy()
            
            # 1. Validacion de formato y patrones
            refined_df = self._validate_formats(original_df, refined_df)
            
            # 2. Validacion de rangos y limites
            refined_df = self._validate_ranges(refined_df)
            
            # 3. Validacion de consistencia entre columnas
            refined_df = self._validate_consistency(refined_df)
            
            # 4. Validacion contextual basada en datos originales
            refined_df = self._validate_context(original_df, refined_df)
            
            # 5. Refinamiento final basado en confianza
            refined_df = self._refine_low_confidence_predictions(original_df, refined_df)
            
            return refined_df
            
        except Exception as e:
            print(f"Error en validacion: {e}")
            return predicted_df  # Devolver sin validar si hay error
    
    def _validate_formats(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida formatos de datos predichos"""
        
        for col in predicted_df.columns:
            if col in ['_table_type']:
                continue
            
            # Detectar tipo de columna basado en datos originales
            col_type = self._detect_column_type(original_df[col] if col in original_df.columns else predicted_df[col])
            
            if col_type in self.validation_patterns:
                pattern = self.validation_patterns[col_type]
                
                # Validar cada valor predicho
                for idx in predicted_df.index:
                    value = predicted_df.loc[idx, col]
                    
                    if pd.notna(value) and not re.match(pattern, str(value)):
                        # Valor no valido, intentar corregir o marcar como baja confianza
                        corrected_value = self._attempt_format_correction(str(value), col_type)
                        if corrected_value:
                            predicted_df.loc[idx, col] = corrected_value
                            self._set_confidence(idx, col, 0.6)  # Confianza media
                        else:
                            # Si no se puede corregir, usar valor original si existe
                            if col in original_df.columns and idx in original_df.index:
                                original_value = original_df.loc[idx, col]
                                if pd.notna(original_value):
                                    predicted_df.loc[idx, col] = original_value
                                    self._set_confidence(idx, col, 1.0)  # Confianza alta (original)
                                else:
                                    predicted_df.loc[idx, col] = pd.NA
                                    self._set_confidence(idx, col, 0.0)  # Sin confianza
                            else:
                                predicted_df.loc[idx, col] = pd.NA
                                self._set_confidence(idx, col, 0.0)
                    else:
                        self._set_confidence(idx, col, 0.9)  # Confianza alta para valores validos
        
        return predicted_df
    
    def _validate_ranges(self, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida rangos numericos"""
        
        for col in predicted_df.columns:
            col_lower = col.lower()
            
            # Buscar rangos aplicables
            applicable_range = None
            for range_key, (min_val, max_val) in self.valid_ranges.items():
                if range_key in col_lower:
                    applicable_range = (min_val, max_val)
                    break
            
            if applicable_range:
                min_val, max_val = applicable_range
                
                for idx in predicted_df.index:
                    value = predicted_df.loc[idx, col]
                    
                    if pd.notna(value):
                        try:
                            numeric_value = float(value)
                            
                            if numeric_value < min_val or numeric_value > max_val:
                                # Valor fuera de rango, corregir
                                corrected_value = max(min_val, min(max_val, numeric_value))
                                predicted_df.loc[idx, col] = corrected_value
                                self._set_confidence(idx, col, 0.5)  # Confianza baja por correccion
                            else:
                                self._set_confidence(idx, col, 0.9)  # Confianza alta
                                
                        except (ValueError, TypeError):
                            # No es numerico, marcar como baja confianza
                            self._set_confidence(idx, col, 0.3)
        
        return predicted_df
    
    def _validate_consistency(self, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia entre columnas relacionadas"""
        
        # Validar edad vs fecha de nacimiento
        age_cols = [col for col in predicted_df.columns if 'edad' in col.lower()]
        birth_cols = [col for col in predicted_df.columns if 'nacimiento' in col.lower() or 'birth' in col.lower()]
        
        if age_cols and birth_cols:
            age_col = age_cols[0]
            birth_col = birth_cols[0]
            
            for idx in predicted_df.index:
                age = predicted_df.loc[idx, age_col]
                birth = predicted_df.loc[idx, birth_col]
                
                if pd.notna(age) and pd.notna(birth):
                    try:
                        birth_date = pd.to_datetime(birth)
                        calculated_age = (datetime.now() - birth_date).days // 365
                        age_value = float(age)
                        
                        # Si la diferencia es significativa, ajustar
                        if abs(calculated_age - age_value) > 2:
                            # Priorizar fecha de nacimiento si parece mas confiable
                            if 1900 <= birth_date.year <= datetime.now().year:
                                predicted_df.loc[idx, age_col] = calculated_age
                                self._set_confidence(idx, age_col, 0.8)
                            else:
                                # Fecha invalida, mantener edad
                                self._set_confidence(idx, age_col, 0.6)
                                
                    except (ValueError, TypeError):
                        continue
        
        return predicted_df
    
    def _validate_context(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida predicciones basandose en contexto de datos originales"""
        
        for col in predicted_df.columns:
            if col not in original_df.columns or col in ['_table_type']:
                continue
            
            # Obtener distribucion de valores originales
            original_values = original_df[col].dropna()
            
            if len(original_values) == 0:
                continue
            
            # Para columnas categoricas, validar contra valores existentes
            if original_values.dtype == 'object':
                unique_original = set(original_values.unique())
                
                for idx in predicted_df.index:
                    predicted_value = predicted_df.loc[idx, col]
                    
                    if pd.notna(predicted_value):
                        if str(predicted_value) in unique_original:
                            self._set_confidence(idx, col, 0.95)  # Alta confianza
                        else:
                            # Buscar valor similar
                            similar_value = self._find_similar_value(str(predicted_value), unique_original)
                            if similar_value:
                                predicted_df.loc[idx, col] = similar_value
                                self._set_confidence(idx, col, 0.7)
                            else:
                                self._set_confidence(idx, col, 0.4)  # Baja confianza
        
        return predicted_df
    
    def _refine_low_confidence_predictions(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Refina predicciones con baja confianza"""
        
        for (idx, col), confidence in self.confidence_scores.items():
            if confidence < 0.5:  # Baja confianza
                
                # Intentar usar valor original si existe
                if col in original_df.columns and idx in original_df.index:
                    original_value = original_df.loc[idx, col]
                    if pd.notna(original_value):
                        predicted_df.loc[idx, col] = original_value
                        self.confidence_scores[(idx, col)] = 1.0
                        continue
                
                # Usar valor mas comun de la columna
                if col in predicted_df.columns:
                    mode_values = predicted_df[col].mode()
                    if len(mode_values) > 0:
                        predicted_df.loc[idx, col] = mode_values[0]
                        self.confidence_scores[(idx, col)] = 0.6
                    else:
                        # Como ultimo recurso, usar NA
                        predicted_df.loc[idx, col] = pd.NA
                        self.confidence_scores[(idx, col)] = 0.0
        
        return predicted_df
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detecta el tipo de una columna"""
        
        sample = series.dropna().astype(str).head(20)
        
        if len(sample) == 0:
            return 'unknown'
        
        # Contar coincidencias con patrones
        pattern_scores = {}
        for pattern_name, pattern in self.validation_patterns.items():
            matches = sample.str.match(pattern, na=False).sum()
            if matches > 0:
                pattern_scores[pattern_name] = matches / len(sample)
        
        # Devolver el patron con mayor puntuacion
        if pattern_scores:
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'text'
    
    def _attempt_format_correction(self, value: str, col_type: str) -> Optional[str]:
        """Intenta corregir formato de un valor"""
        
        if col_type == 'email':
            # Correcciones comunes de email
            value = value.lower().strip()
            
            # Agregar @ si falta
            if '@' not in value and '.' in value:
                parts = value.split('.')
                if len(parts) >= 2:
                    value = f"{parts[0]}@{'.'.join(parts[1:])}"
            
            # Corregir dominios comunes
            domain_fixes = {
                'gmai.com': 'gmail.com',
                'hotmial.com': 'hotmail.com',
                'yahoo.co': 'yahoo.com'
            }
            
            for wrong, correct in domain_fixes.items():
                value = value.replace(wrong, correct)
            
            # Validar resultado
            if re.match(self.validation_patterns['email'], value):
                return value
        
        elif col_type == 'phone':
            # Limpiar telefono
            clean_phone = re.sub(r'[^\d\+]', '', value)
            if 7 <= len(clean_phone.replace('+', '')) <= 15:
                return clean_phone
        
        elif col_type == 'date':
            # Intentar parsear fecha
            try:
                parsed_date = pd.to_datetime(value, errors='coerce')
                if pd.notna(parsed_date):
                    return parsed_date.strftime('%Y-%m-%d')
            except:
                pass
        
        return None
    
    def _find_similar_value(self, target: str, candidates: set) -> Optional[str]:
        """Encuentra valor similar usando distancia de edicion simple"""
        
        target_lower = target.lower()
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = str(candidate).lower()
            
            # Similitud simple basada en caracteres comunes
            common_chars = set(target_lower) & set(candidate_lower)
            total_chars = set(target_lower) | set(candidate_lower)
            
            if total_chars:
                similarity = len(common_chars) / len(total_chars)
                
                if similarity > best_score and similarity > 0.6:
                    best_score = similarity
                    best_match = candidate
        
        return best_match
    
    def _set_confidence(self, idx: int, col: str, confidence: float):
        """Establece puntuacion de confianza"""
        self.confidence_scores[(idx, col)] = confidence
    
    def get_validation_report(self) -> Dict:
        """Genera reporte de validacion"""
        
        if not self.confidence_scores:
            return {
                'total_predictions': 0,
                'average_confidence': 0.0,
                'high_confidence_count': 0,
                'low_confidence_count': 0
            }
        
        confidences = list(self.confidence_scores.values())
        
        return {
            'total_predictions': len(confidences),
            'average_confidence': np.mean(confidences),
            'high_confidence_count': sum(1 for c in confidences if c >= 0.8),
            'medium_confidence_count': sum(1 for c in confidences if 0.5 <= c < 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5),
            'confidence_distribution': {
                'min': np.min(confidences),
                'max': np.max(confidences),
                'std': np.std(confidences)
            }
        }
