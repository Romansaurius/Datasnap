"""
🤖 ADVANCED ML OPTIMIZER 🤖
Optimizador avanzado que usa Machine Learning para predicciones inteligentes
- Usa algoritmos de ML para detectar patrones reales
- Predicciones basadas en correlaciones entre columnas
- Detección de outliers y anomalías
- Corrección inteligente basada en contexto
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from collections import Counter
import re

# Importaciones opcionales de ML
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import KNNImputer
    from sklearn.cluster import DBSCAN
    from scipy import stats
    from fuzzywuzzy import fuzz, process
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedMLOptimizer:
    """Optimizador avanzado con Machine Learning"""
    
    def __init__(self):
        self.ml_available = ML_AVAILABLE
        self.label_encoders = {}
        self.column_relationships = {}
        self.outlier_thresholds = {}
        
        # Patrones de corrección inteligente
        self.smart_corrections = {
            'email_domains': {
                'gmai.com': 'gmail.com', 'gmial.com': 'gmail.com', 'gmaill.com': 'gmail.com',
                'hotmial.com': 'hotmail.com', 'hotmailcom': 'hotmail.com',
                'yahoo.co': 'yahoo.com', 'yahooo.com': 'yahoo.com',
                'outlok.com': 'outlook.com', 'outllok.com': 'outlook.com'
            },
            'common_typos': {
                'recieve': 'receive', 'seperate': 'separate', 'definately': 'definitely',
                'occured': 'occurred', 'begining': 'beginning', 'accomodate': 'accommodate'
            },
            'spanish_names': {
                'jose': 'José', 'maria': 'María', 'jesus': 'Jesús', 'angel': 'Ángel',
                'monica': 'Mónica', 'adrian': 'Adrián', 'ruben': 'Rubén'
            }
        }
    
    def optimize_with_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización principal usando ML con validación avanzada"""
        
        if df.empty:
            return df
        
        try:
            df_original = df.copy()  # Guardar original para validación
            df_optimized = df.copy()
            
            # 1. Análisis de correlaciones entre columnas
            df_optimized = self._analyze_column_relationships(df_optimized)
            
            # 2. Detección y corrección de outliers
            df_optimized = self._detect_and_correct_outliers(df_optimized)
            
            # 3. Corrección inteligente de texto usando fuzzy matching
            df_optimized = self._intelligent_text_correction(df_optimized)
            
            # 4. Predicción ML de valores faltantes
            if self.ml_available:
                df_optimized = self._ml_missing_value_prediction(df_optimized)
            else:
                df_optimized = self._statistical_missing_value_prediction(df_optimized)
            
            # 5. Validación de consistencia usando reglas aprendidas
            df_optimized = self._validate_with_learned_rules(df_optimized)
            
            # 6. NUEVA: Validación y refinamiento avanzado de predicciones
            df_optimized = self._validate_and_refine_predictions(df_original, df_optimized)
            
            return df_optimized
            
        except Exception as e:
            print(f"Error en optimización ML: {e}")
            return self._fallback_optimization(df)
    
    def _validate_and_refine_predictions(self, original_df: pd.DataFrame, predicted_df: pd.DataFrame) -> pd.DataFrame:
        """Valida y refina predicciones usando validador avanzado"""
        
        try:
            from .prediction_validator import PredictionValidator
            validator = PredictionValidator()
            
            # Aplicar validación y refinamiento
            refined_df = validator.validate_and_refine_predictions(original_df, predicted_df)
            
            # Guardar reporte de validación
            validation_report = validator.get_validation_report()
            print(f"Validación completada - Confianza promedio: {validation_report['average_confidence']:.2f}")
            
            return refined_df
            
        except Exception as e:
            print(f"Error en validación de predicciones: {e}")
            return predicted_df  # Devolver sin validar si hay error
    
    def _analyze_column_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analiza relaciones entre columnas para predicciones inteligentes"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Calcular correlaciones
            corr_matrix = df[numeric_cols].corr()
            
            # Encontrar correlaciones fuertes (>0.7)
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.7:
                        if col1 not in self.column_relationships:
                            self.column_relationships[col1] = []
                        self.column_relationships[col1].append({
                            'related_column': col2,
                            'correlation': corr_matrix.loc[col1, col2]
                        })
        
        return df
    
    def _detect_and_correct_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y corrige outliers usando métodos estadísticos"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 10:  # Muy pocos datos
                continue
            
            # Método IQR para detectar outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Guardar umbrales
            self.outlier_thresholds[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'median': df[col].median()
            }
            
            # Corregir outliers extremos (más allá de 3 IQR)
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            
            extreme_outliers = (df[col] < extreme_lower) | (df[col] > extreme_upper)
            
            if extreme_outliers.any():
                # Reemplazar outliers extremos con la mediana
                df.loc[extreme_outliers, col] = df[col].median()
        
        return df
    
    def _intelligent_text_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrección inteligente de texto usando fuzzy matching"""
        
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            col_lower = col.lower()
            
            # Corrección específica por tipo de columna
            if 'email' in col_lower:
                df[col] = self._correct_emails_fuzzy(df[col])
            elif 'nombre' in col_lower or 'name' in col_lower:
                df[col] = self._correct_names_fuzzy(df[col])
            elif 'ciudad' in col_lower or 'city' in col_lower:
                df[col] = self._correct_cities_fuzzy(df[col])
            else:
                df[col] = self._correct_general_text_fuzzy(df[col])
        
        return df
    
    def _correct_emails_fuzzy(self, series: pd.Series) -> pd.Series:
        """Corrección de emails usando fuzzy matching"""
        
        def fix_email(email):
            if pd.isna(email):
                return email
            
            email_str = str(email).lower().strip()
            
            # Aplicar correcciones exactas primero
            for wrong, correct in self.smart_corrections['email_domains'].items():
                email_str = email_str.replace(wrong, correct)
            
            # Fuzzy matching para dominios comunes
            if '@' in email_str:
                local, domain = email_str.split('@', 1)
                common_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com']
                
                if domain not in common_domains:
                    # Buscar dominio más similar
                    if ML_AVAILABLE:
                        best_match = process.extractOne(domain, common_domains, scorer=fuzz.ratio)
                        if best_match and best_match[1] > 80:  # 80% de similitud
                            email_str = f"{local}@{best_match[0]}"
            
            return email_str
        
        return series.apply(fix_email)
    
    def _correct_names_fuzzy(self, series: pd.Series) -> pd.Series:
        """Corrección de nombres usando fuzzy matching"""
        
        def fix_name(name):
            if pd.isna(name):
                return name
            
            name_str = str(name).strip().title()
            
            # Corrección de nombres españoles comunes
            for wrong, correct in self.smart_corrections['spanish_names'].items():
                if wrong.lower() in name_str.lower():
                    name_str = name_str.replace(wrong.title(), correct)
            
            return name_str
        
        return series.apply(fix_name)
    
    def _correct_cities_fuzzy(self, series: pd.Series) -> pd.Series:
        """Corrección de ciudades usando fuzzy matching"""
        
        spanish_cities = [
            'Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza', 'Málaga',
            'Murcia', 'Palma', 'Bilbao', 'Alicante', 'Córdoba', 'Valladolid'
        ]
        
        def fix_city(city):
            if pd.isna(city):
                return city
            
            city_str = str(city).strip().title()
            
            # Fuzzy matching con ciudades españolas
            if ML_AVAILABLE and city_str not in spanish_cities:
                best_match = process.extractOne(city_str, spanish_cities, scorer=fuzz.ratio)
                if best_match and best_match[1] > 85:  # 85% de similitud
                    return best_match[0]
            
            return city_str
        
        return series.apply(fix_city)
    
    def _correct_general_text_fuzzy(self, series: pd.Series) -> pd.Series:
        """Corrección general de texto"""
        
        def fix_text(text):
            if pd.isna(text):
                return text
            
            text_str = str(text).strip()
            
            # Corrección de errores comunes
            for wrong, correct in self.smart_corrections['common_typos'].items():
                text_str = text_str.replace(wrong, correct)
            
            # Limpiar espacios múltiples
            text_str = re.sub(r'\s+', ' ', text_str)
            
            return text_str
        
        return series.apply(fix_text)
    
    def _ml_missing_value_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción avanzada de valores faltantes usando motor de predicción"""
        
        try:
            # Usar motor de predicción avanzado
            from .advanced_prediction_engine import AdvancedPredictionEngine
            prediction_engine = AdvancedPredictionEngine()
            return prediction_engine.predict_missing_values_advanced(df)
            
        except Exception as e:
            print(f"Error en motor de predicción avanzado: {e}")
            # Fallback a predicción básica
            return self._basic_ml_prediction(df)
    
    def _basic_ml_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción ML básica como fallback"""
        
        if not self.ml_available:
            return self._statistical_missing_value_prediction(df)
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0 or missing_count == len(df):
                continue
            
            try:
                # Preparar datos para ML
                feature_cols = [c for c in df.columns if c != col and df[c].notna().sum() > 0]
                if not feature_cols:
                    continue
                
                # Crear dataset para entrenamiento
                train_mask = df[col].notna()
                predict_mask = df[col].isna()
                
                if train_mask.sum() < 5:  # Reducir umbral mínimo
                    continue
                
                X_train = self._prepare_features(df.loc[train_mask, feature_cols])
                y_train = df.loc[train_mask, col]
                X_predict = self._prepare_features(df.loc[predict_mask, feature_cols])
                
                if X_train.empty or X_predict.empty:
                    continue
                
                # Elegir modelo según tipo de datos
                if df[col].dtype in ['object']:
                    # Clasificación para datos categóricos
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train.astype(str))
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train_encoded)
                    
                    predictions = model.predict(X_predict)
                    predictions_decoded = le.inverse_transform(predictions)
                    
                    df.loc[predict_mask, col] = predictions_decoded
                    
                else:
                    # Regresión para datos numéricos
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    
                    predictions = model.predict(X_predict)
                    df.loc[predict_mask, col] = predictions
                    
            except Exception as e:
                print(f"Error en predicción ML básica para columna {col}: {e}")
                continue
        
        return df
    
    def _prepare_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Prepara características para ML"""
        
        df_prep = df_features.copy()
        
        # Codificar variables categóricas
        for col in df_prep.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Ajustar con todos los valores únicos no nulos
                unique_vals = df_prep[col].dropna().unique()
                if len(unique_vals) > 0:
                    self.label_encoders[col].fit(unique_vals.astype(str))
            
            # Transformar valores no nulos
            mask = df_prep[col].notna()
            if mask.any():
                try:
                    df_prep.loc[mask, col] = self.label_encoders[col].transform(
                        df_prep.loc[mask, col].astype(str)
                    )
                except ValueError:
                    # Si hay valores nuevos, usar -1
                    df_prep[col] = -1
        
        # Rellenar NaN con -1 para ML
        df_prep = df_prep.fillna(-1)
        
        return df_prep
    
    def _statistical_missing_value_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción estadística como fallback"""
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # Usar mediana para números
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
            
            elif df[col].dtype == 'object':
                # Usar moda para texto (solo si hay suficientes datos)
                mode_val = df[col].mode()
                if len(mode_val) > 0 and df[col].notna().sum() > 5:
                    df[col] = df[col].fillna(mode_val[0])
        
        return df
    
    def _validate_with_learned_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validación usando reglas aprendidas de los datos"""
        
        # Validar rangos numéricos basados en outliers detectados
        for col, thresholds in self.outlier_thresholds.items():
            if col in df.columns:
                # Corregir valores que están fuera de rangos razonables
                extreme_mask = (df[col] < thresholds['lower'] * 2) | (df[col] > thresholds['upper'] * 2)
                if extreme_mask.any():
                    df.loc[extreme_mask, col] = thresholds['median']
        
        # Validar consistencia entre columnas relacionadas
        for col, relationships in self.column_relationships.items():
            if col in df.columns:
                for rel in relationships:
                    rel_col = rel['related_column']
                    if rel_col in df.columns:
                        # Detectar inconsistencias basadas en correlación
                        correlation = rel['correlation']
                        
                        # Si la correlación es fuerte, validar consistencia
                        if abs(correlation) > 0.8:
                            self._validate_correlated_columns(df, col, rel_col, correlation)
        
        return df
    
    def _validate_correlated_columns(self, df: pd.DataFrame, col1: str, col2: str, correlation: float):
        """Valida consistencia entre columnas correlacionadas"""
        
        # Obtener datos válidos para ambas columnas
        valid_mask = df[col1].notna() & df[col2].notna()
        if valid_mask.sum() < 10:
            return
        
        valid_data = df.loc[valid_mask, [col1, col2]]
        
        # Calcular residuos de la relación lineal
        if correlation > 0:
            expected_col2 = valid_data[col1] * (valid_data[col2].std() / valid_data[col1].std())
        else:
            expected_col2 = -valid_data[col1] * (valid_data[col2].std() / valid_data[col1].std())
        
        residuals = abs(valid_data[col2] - expected_col2)
        threshold = residuals.quantile(0.95)  # Top 5% como outliers
        
        # Marcar valores inconsistentes
        inconsistent_mask = residuals > threshold
        if inconsistent_mask.any():
            # Para valores muy inconsistentes, usar predicción basada en la otra columna
            inconsistent_indices = valid_data.index[inconsistent_mask]
            for idx in inconsistent_indices:
                if abs(correlation) > 0.9:  # Solo para correlaciones muy fuertes
                    predicted_val = df.loc[idx, col1] * correlation
                    df.loc[idx, col2] = predicted_val
    
    def _fallback_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimización básica como fallback"""
        
        try:
            # Limpieza básica
            null_values = ['', 'nan', 'null', 'none', 'n/a', 'invalid']
            df = df.replace(null_values, pd.NA)
            
            # Eliminar filas completamente vacías
            df = df.dropna(how='all')
            
            # Eliminar duplicados
            df = df.drop_duplicates()
            
            return df.reset_index(drop=True)
            
        except Exception:
            return df
    
    def get_optimization_summary(self) -> Dict:
        """Obtiene resumen de la optimización"""
        
        return {
            'ml_available': self.ml_available,
            'column_relationships_found': len(self.column_relationships),
            'outlier_thresholds_set': len(self.outlier_thresholds),
            'label_encoders_created': len(self.label_encoders),
            'corrections_applied': {
                'email_domains': len(self.smart_corrections['email_domains']),
                'common_typos': len(self.smart_corrections['common_typos']),
                'spanish_names': len(self.smart_corrections['spanish_names'])
            }
        }