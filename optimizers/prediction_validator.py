"""
🔍 PREDICTION VALIDATOR 🔍
Validador y refinador de predicciones para máxima precisión
- Validación cruzada de predicciones
- Detección de predicciones anómalas
- Refinamiento iterativo
- Métricas de confianza
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    from sklearn.model_selection import cross_val_score
    from scipy import stats
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

warnings.filterwarnings('ignore')

class PredictionValidator:
    """Validador avanzado de predicciones"""
    
    def __init__(self):
        self.validation_available = VALIDATION_AVAILABLE
        self.confidence_scores = {}
        self.validation_metrics = {}
        self.anomaly_thresholds = {}
        
    def validate_and_refine_predictions(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida y refina predicciones para máxima precisión"""
        
        if not self.validation_available:
            return self._basic_validation(original_df, predicted_df)
        
        refined_df = predicted_df.copy()
        
        # 1. Validar consistencia interna
        refined_df = self._validate_internal_consistency(refined_df)
        
        # 2. Detectar predicciones anómalas
        refined_df = self._detect_anomalous_predictions(original_df, refined_df)
        
        # 3. Calcular métricas de confianza
        self._calculate_confidence_scores(original_df, refined_df)
        
        # 4. Refinamiento iterativo
        refined_df = self._iterative_refinement(original_df, refined_df)
        
        # 5. Validación final
        refined_df = self._final_validation(refined_df)
        
        return refined_df
    
    def _validate_internal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia interna entre columnas relacionadas"""
        
        # Validar relaciones precio-categoria
        if 'precio' in df.columns and 'categoria' in df.columns:
            df = self._validate_price_category_consistency(df)
        
        # Validar relaciones edad-salario
        if 'edad' in df.columns and 'salario' in df.columns:
            df = self._validate_age_salary_consistency(df)
        
        # Validar relaciones stock-precio
        if 'stock' in df.columns and 'precio' in df.columns:
            df = self._validate_stock_price_consistency(df)
        
        return df
    
    def _validate_price_category_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia precio-categoría"""
        
        # Rangos esperados por categoría
        expected_ranges = {
            'laptop': (400, 3000),
            'mouse': (10, 100),
            'teclado': (20, 200),
            'monitor': (100, 1000),
            'smartphone': (150, 1500),
            'tablet': (80, 800)
        }
        
        for idx, row in df.iterrows():
            categoria = str(row['categoria']).lower()
            precio = row['precio']
            
            if pd.notna(precio) and categoria in expected_ranges:
                min_price, max_price = expected_ranges[categoria]
                
                # Si el precio está muy fuera del rango, ajustar
                if precio < min_price * 0.5:
                    df.loc[idx, 'precio'] = min_price
                elif precio > max_price * 2:
                    df.loc[idx, 'precio'] = max_price
        
        return df
    
    def _validate_age_salary_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia edad-salario"""
        
        for idx, row in df.iterrows():
            edad = row['edad']
            salario = row['salario']
            
            if pd.notna(edad) and pd.notna(salario):
                # Salario mínimo esperado por edad
                expected_min_salary = max(20000, (edad - 18) * 1500 + 25000)
                expected_max_salary = min(150000, edad * 3000)
                
                # Ajustar si está muy fuera del rango
                if salario < expected_min_salary * 0.7:
                    df.loc[idx, 'salario'] = expected_min_salary
                elif salario > expected_max_salary * 1.5:
                    df.loc[idx, 'salario'] = expected_max_salary
        
        return df
    
    def _validate_stock_price_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia stock-precio (relación inversa típica)"""
        
        # Calcular correlación esperada
        valid_data = df[['stock', 'precio']].dropna()
        if len(valid_data) > 5:
            correlation = valid_data.corr().loc['stock', 'precio']
            
            # Si hay correlación negativa fuerte (productos caros = menos stock)
            if correlation < -0.3:
                median_stock = df['stock'].median()
                median_price = df['precio'].median()
                
                for idx, row in df.iterrows():
                    precio = row['precio']
                    stock = row['stock']
                    
                    if pd.notna(precio) and pd.notna(stock):
                        # Ajustar stock si es inconsistente con precio
                        if precio > median_price * 2 and stock > median_stock * 2:
                            df.loc[idx, 'stock'] = median_stock * 0.5
                        elif precio < median_price * 0.5 and stock < median_stock * 0.5:
                            df.loc[idx, 'stock'] = median_stock * 2
        
        return df
    
    def _detect_anomalous_predictions(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y corrige predicciones anómalas"""
        
        refined_df = predicted_df.copy()
        
        for col in predicted_df.columns:
            if col not in original_df.columns:
                continue
            
            # Obtener datos originales válidos
            original_valid = original_df[col].dropna()
            if len(original_valid) < 5:
                continue
            
            # Calcular estadísticas de referencia
            mean_val = original_valid.mean() if original_valid.dtype in ['int64', 'float64'] else None
            std_val = original_valid.std() if original_valid.dtype in ['int64', 'float64'] else None
            
            if mean_val is not None and std_val is not None:
                # Detectar outliers en predicciones
                predicted_values = predicted_df[col]
                
                # Z-score para detectar anomalías
                z_scores = np.abs((predicted_values - mean_val) / std_val)
                anomalous_mask = z_scores > 3  # Más de 3 desviaciones estándar
                
                if anomalous_mask.any():
                    # Reemplazar valores anómalos con valores más conservadores
                    for idx in predicted_df[anomalous_mask].index:
                        if original_df.loc[idx, col] != predicted_df.loc[idx, col]:  # Solo si fue predicho
                            # Usar percentil 75 como valor conservador
                            conservative_value = original_valid.quantile(0.75)
                            refined_df.loc[idx, col] = conservative_value
        
        return refined_df
    
    def _calculate_confidence_scores(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame):
        """Calcula métricas de confianza para las predicciones"""
        
        for col in predicted_df.columns:
            if col not in original_df.columns:
                continue
            
            # Identificar valores predichos vs originales
            original_mask = original_df[col].notna()
            predicted_mask = original_df[col].isna() & predicted_df[col].notna()
            
            if not predicted_mask.any():
                continue
            
            confidence_metrics = {
                'total_predictions': predicted_mask.sum(),
                'prediction_ratio': predicted_mask.sum() / len(predicted_df),
                'data_availability': original_mask.sum() / len(original_df)
            }
            
            # Calcular métricas específicas por tipo de datos
            if predicted_df[col].dtype in ['int64', 'float64']:
                original_values = original_df.loc[original_mask, col]
                
                if len(original_values) > 1:
                    confidence_metrics.update({
                        'original_mean': original_values.mean(),
                        'original_std': original_values.std(),
                        'predicted_mean': predicted_df.loc[predicted_mask, col].mean(),
                        'predicted_std': predicted_df.loc[predicted_mask, col].std()
                    })
                    
                    # Score de confianza basado en similitud estadística
                    mean_diff = abs(confidence_metrics['original_mean'] - confidence_metrics['predicted_mean'])
                    std_diff = abs(confidence_metrics['original_std'] - confidence_metrics['predicted_std'])
                    
                    confidence_score = max(0, 1 - (mean_diff / confidence_metrics['original_mean']) - 
                                         (std_diff / confidence_metrics['original_std']))
                    confidence_metrics['confidence_score'] = confidence_score
            
            else:
                # Para datos categóricos
                original_values = original_df.loc[original_mask, col]
                predicted_values = predicted_df.loc[predicted_mask, col]
                
                original_dist = original_values.value_counts(normalize=True)
                predicted_dist = predicted_values.value_counts(normalize=True)
                
                # Similitud de distribuciones
                common_values = set(original_dist.index) & set(predicted_dist.index)
                if common_values:
                    similarity = sum(min(original_dist.get(val, 0), predicted_dist.get(val, 0)) 
                                   for val in common_values)
                    confidence_metrics['confidence_score'] = similarity
                else:
                    confidence_metrics['confidence_score'] = 0.0
            
            self.confidence_scores[col] = confidence_metrics
    
    def _iterative_refinement(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Refinamiento iterativo de predicciones"""
        
        refined_df = predicted_df.copy()
        max_iterations = 3
        
        for iteration in range(max_iterations):
            previous_df = refined_df.copy()
            
            # Aplicar refinamientos
            refined_df = self._refine_based_on_correlations(original_df, refined_df)
            refined_df = self._refine_based_on_patterns(refined_df)
            
            # Verificar convergencia
            if self._check_convergence(previous_df, refined_df):
                break
        
        return refined_df
    
    def _refine_based_on_correlations(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Refinamiento basado en correlaciones observadas"""
        
        refined_df = predicted_df.copy()
        numeric_cols = predicted_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Calcular correlaciones en datos originales
            original_corr = original_df[numeric_cols].corr()
            
            # Ajustar predicciones para mantener correlaciones
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2 and abs(original_corr.loc[col1, col2]) > 0.5:
                        # Ajustar valores predichos para mantener correlación
                        predicted_mask = original_df[col1].isna() & predicted_df[col1].notna()
                        
                        if predicted_mask.any():
                            for idx in predicted_df[predicted_mask].index:
                                if pd.notna(refined_df.loc[idx, col2]):
                                    # Ajustar col1 basado en col2 y correlación
                                    expected_val = (refined_df.loc[idx, col2] * 
                                                  original_corr.loc[col1, col2] * 
                                                  (original_df[col1].std() / original_df[col2].std()) +
                                                  original_df[col1].mean())
                                    
                                    # Promedio ponderado entre predicción original y esperada
                                    current_val = refined_df.loc[idx, col1]
                                    refined_val = 0.7 * current_val + 0.3 * expected_val
                                    refined_df.loc[idx, col1] = refined_val
        
        return refined_df
    
    def _refine_based_on_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Refinamiento basado en patrones detectados"""
        
        refined_df = df.copy()
        
        # Patrón: productos de la misma categoría deberían tener precios similares
        if 'categoria' in df.columns and 'precio' in df.columns:
            for categoria in df['categoria'].unique():
                if pd.notna(categoria):
                    mask = df['categoria'] == categoria
                    precios = df.loc[mask, 'precio'].dropna()
                    
                    if len(precios) > 1:
                        median_precio = precios.median()
                        std_precio = precios.std()
                        
                        # Ajustar precios extremos dentro de la categoría
                        for idx in df[mask].index:
                            precio = df.loc[idx, 'precio']
                            if pd.notna(precio):
                                if abs(precio - median_precio) > 2 * std_precio:
                                    # Ajustar hacia la mediana
                                    adjusted_precio = 0.8 * precio + 0.2 * median_precio
                                    refined_df.loc[idx, 'precio'] = adjusted_precio
        
        return refined_df
    
    def _check_convergence(self, previous_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """Verifica si el refinamiento ha convergido"""
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            diff = abs(current_df[col] - previous_df[col]).mean()
            if diff > 0.01:  # Umbral de convergencia
                return False
        
        return True
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validación final y limpieza"""
        
        validated_df = df.copy()
        
        # Validar rangos finales
        for col in validated_df.columns:
            if validated_df[col].dtype in ['int64', 'float64']:
                # Eliminar valores infinitos o extremadamente grandes
                validated_df[col] = validated_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Limitar valores extremos
                q99 = validated_df[col].quantile(0.99)
                q01 = validated_df[col].quantile(0.01)
                
                validated_df[col] = validated_df[col].clip(lower=q01, upper=q99)
        
        return validated_df
    
    def _basic_validation(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Validación básica cuando librerías avanzadas no están disponibles"""
        
        validated_df = predicted_df.copy()
        
        # Validación básica de rangos
        for col in validated_df.columns:
            if validated_df[col].dtype in ['int64', 'float64']:
                # Usar estadísticas básicas de datos originales
                original_valid = original_df[col].dropna()
                if len(original_valid) > 0:
                    median_val = original_valid.median()
                    
                    # Reemplazar valores extremos con mediana
                    extreme_mask = (validated_df[col] > median_val * 10) | (validated_df[col] < median_val * 0.1)
                    validated_df.loc[extreme_mask, col] = median_val
        
        return validated_df
    
    def get_validation_report(self) -> Dict:
        """Genera reporte de validación"""
        
        return {
            'validation_available': self.validation_available,
            'confidence_scores': self.confidence_scores,
            'validation_metrics': self.validation_metrics,
            'columns_validated': len(self.confidence_scores),
            'average_confidence': np.mean([
                metrics.get('confidence_score', 0) 
                for metrics in self.confidence_scores.values()
            ]) if self.confidence_scores else 0
        }