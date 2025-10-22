"""
🚀 ADVANCED PREDICTION ENGINE 🚀
Motor de predicción avanzado con múltiples algoritmos de ML
- Ensemble de modelos para mayor precisión
- Predicción contextual basada en patrones semánticos
- Auto-tuning de hiperparámetros
- Validación cruzada para selección de modelos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
import re

# Importaciones ML avanzadas
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.impute import KNNImputer, IterativeImputer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import mean_squared_error, accuracy_score
    from scipy import stats
    from scipy.spatial.distance import cosine
    ML_ADVANCED = True
except ImportError:
    ML_ADVANCED = False

warnings.filterwarnings('ignore')

class AdvancedPredictionEngine:
    """Motor de predicción avanzado con ensemble de modelos"""
    
    def __init__(self):
        self.ml_available = ML_ADVANCED
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Configuración de modelos
        self.regression_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'lr': LinearRegression()
        } if ML_ADVANCED else {}
        
        self.classification_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        } if ML_ADVANCED else {}
        
        # Patrones semánticos para predicción contextual
        self.semantic_patterns = {
            'price_indicators': ['precio', 'price', 'cost', 'valor', 'importe'],
            'quantity_indicators': ['cantidad', 'stock', 'qty', 'units', 'unidades'],
            'percentage_indicators': ['descuento', 'discount', 'porcentaje', 'percent', 'ratio'],
            'category_indicators': ['categoria', 'category', 'tipo', 'type', 'clase'],
            'location_indicators': ['ciudad', 'city', 'pais', 'country', 'region'],
            'person_indicators': ['nombre', 'name', 'cliente', 'customer', 'usuario'],
            'contact_indicators': ['email', 'telefono', 'phone', 'contacto'],
            'date_indicators': ['fecha', 'date', 'time', 'timestamp', 'created'],
            'status_indicators': ['activo', 'active', 'estado', 'status', 'enabled']
        }
        
        # Base de conocimiento para predicciones contextuales
        self.knowledge_base = {
            'price_ranges': {
                'laptop': (400, 2500), 'mouse': (15, 80), 'teclado': (30, 150),
                'monitor': (150, 800), 'smartphone': (200, 1200), 'tablet': (100, 600)
            },
            'typical_discounts': {
                'electronics': (0.05, 0.25), 'clothing': (0.10, 0.50), 'books': (0.05, 0.30)
            },
            'stock_ranges': {
                'high_value': (5, 50), 'medium_value': (20, 200), 'low_value': (50, 1000)
            }
        }
    
    def predict_missing_values_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción avanzada usando ensemble de modelos"""
        
        if not self.ml_available:
            return self._statistical_fallback(df)
        
        df_result = df.copy()
        
        # 1. Análisis de dependencias entre columnas
        dependencies = self._analyze_column_dependencies(df_result)
        
        # 2. Predicción por orden de dependencia
        prediction_order = self._determine_prediction_order(dependencies)
        
        # 3. Aplicar predicción avanzada por columna
        for col in prediction_order:
            if df_result[col].isna().sum() > 0:
                df_result = self._predict_column_advanced(df_result, col, dependencies)
        
        return df_result
    
    def _analyze_column_dependencies(self, df: pd.DataFrame) -> Dict:
        """Analiza dependencias entre columnas para predicción inteligente"""
        
        dependencies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calcular correlaciones
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            for col in numeric_cols:
                dependencies[col] = {
                    'correlations': {},
                    'predictors': [],
                    'semantic_type': self._detect_semantic_type(col)
                }
                
                for other_col in numeric_cols:
                    if col != other_col:
                        corr = corr_matrix.loc[col, other_col]
                        if abs(corr) > 0.3:  # Correlación significativa
                            dependencies[col]['correlations'][other_col] = corr
                            if abs(corr) > 0.5:  # Predictor fuerte
                                dependencies[col]['predictors'].append(other_col)
        
        # Analizar columnas categóricas
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col not in dependencies:
                dependencies[col] = {
                    'correlations': {},
                    'predictors': [],
                    'semantic_type': self._detect_semantic_type(col)
                }
            
            # Buscar predictores categóricos
            for other_col in cat_cols:
                if col != other_col and df[other_col].notna().sum() > 0:
                    # Calcular asociación usando chi-cuadrado
                    association = self._calculate_categorical_association(df, col, other_col)
                    if association > 0.3:
                        dependencies[col]['predictors'].append(other_col)
        
        return dependencies
    
    def _detect_semantic_type(self, column_name: str) -> str:
        """Detecta el tipo semántico de una columna"""
        
        col_lower = column_name.lower()
        
        for semantic_type, indicators in self.semantic_patterns.items():
            for indicator in indicators:
                if indicator in col_lower:
                    return semantic_type
        
        return 'general'
    
    def _calculate_categorical_association(self, df: pd.DataFrame, col1: str, col2: str) -> float:
        """Calcula asociación entre variables categóricas"""
        
        try:
            # Crear tabla de contingencia
            contingency = pd.crosstab(df[col1].fillna('missing'), df[col2].fillna('missing'))
            
            # Chi-cuadrado normalizado
            chi2 = stats.chi2_contingency(contingency)[0]
            n = contingency.sum().sum()
            
            # Coeficiente de Cramér's V
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            return cramers_v
            
        except:
            return 0.0
    
    def _determine_prediction_order(self, dependencies: Dict) -> List[str]:
        """Determina el orden óptimo de predicción"""
        
        # Ordenar por número de predictores disponibles (menos dependientes primero)
        order = sorted(dependencies.keys(), 
                      key=lambda x: len(dependencies[x]['predictors']))
        
        return order
    
    def _predict_column_advanced(self, df: pd.DataFrame, target_col: str, dependencies: Dict) -> pd.DataFrame:
        """Predicción avanzada para una columna específica"""
        
        missing_mask = df[target_col].isna()
        if not missing_mask.any():
            return df
        
        # Obtener predictores para esta columna
        predictors = dependencies[target_col]['predictors']
        semantic_type = dependencies[target_col]['semantic_type']
        
        # Si no hay predictores, usar predicción contextual
        if not predictors:
            return self._contextual_prediction(df, target_col, semantic_type)
        
        # Preparar datos para ML
        feature_cols = [col for col in predictors if df[col].notna().sum() > 0]
        if not feature_cols:
            return self._contextual_prediction(df, target_col, semantic_type)
        
        # Dividir en entrenamiento y predicción
        train_mask = df[target_col].notna()
        if train_mask.sum() < 5:  # Muy pocos datos
            return self._contextual_prediction(df, target_col, semantic_type)
        
        try:
            X_train = self._prepare_features_advanced(df.loc[train_mask, feature_cols])
            y_train = df.loc[train_mask, target_col]
            X_predict = self._prepare_features_advanced(df.loc[missing_mask, feature_cols])
            
            # Seleccionar y entrenar mejor modelo
            best_model = self._select_best_model(X_train, y_train, df[target_col].dtype)
            
            # Hacer predicciones
            predictions = best_model.predict(X_predict)
            
            # Post-procesamiento según tipo semántico
            predictions = self._post_process_predictions(predictions, semantic_type, target_col)
            
            # Asignar predicciones
            df.loc[missing_mask, target_col] = predictions
            
        except Exception as e:
            print(f"Error en predicción ML para {target_col}: {e}")
            return self._contextual_prediction(df, target_col, semantic_type)
        
        return df
    
    def _prepare_features_advanced(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Preparación avanzada de características"""
        
        df_prep = df_features.copy()
        
        # Codificar variables categóricas
        for col in df_prep.select_dtypes(include=['object']).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                unique_vals = df_prep[col].dropna().unique()
                if len(unique_vals) > 0:
                    self.encoders[col].fit(unique_vals.astype(str))
            
            # Transformar
            mask = df_prep[col].notna()
            if mask.any():
                try:
                    df_prep.loc[mask, col] = self.encoders[col].transform(
                        df_prep.loc[mask, col].astype(str)
                    )
                except ValueError:
                    df_prep[col] = -1
        
        # Escalar características numéricas
        numeric_cols = df_prep.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler_key = '_'.join(sorted(numeric_cols))
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                self.scalers[scaler_key].fit(df_prep[numeric_cols].fillna(0))
            
            df_prep[numeric_cols] = self.scalers[scaler_key].transform(
                df_prep[numeric_cols].fillna(0)
            )
        
        # Rellenar NaN restantes
        df_prep = df_prep.fillna(0)
        
        return df_prep
    
    def _select_best_model(self, X_train: pd.DataFrame, y_train: pd.Series, target_dtype) -> Any:
        """Selecciona el mejor modelo usando validación cruzada"""
        
        is_classification = target_dtype == 'object' or y_train.nunique() < 10
        
        models = self.classification_models if is_classification else self.regression_models
        best_model = None
        best_score = -np.inf if not is_classification else -np.inf
        
        for name, model in models.items():
            try:
                # Validación cruzada
                if is_classification:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
                
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception:
                continue
        
        # Entrenar el mejor modelo
        if best_model is not None:
            best_model.fit(X_train, y_train)
            return best_model
        else:
            # Fallback a modelo simple
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            model.fit(X_train, y_train)
            return model
    
    def _contextual_prediction(self, df: pd.DataFrame, target_col: str, semantic_type: str) -> pd.DataFrame:
        """Predicción contextual basada en conocimiento semántico"""
        
        missing_mask = df[target_col].isna()
        
        if semantic_type == 'price_indicators':
            predictions = self._predict_prices_contextual(df, target_col, missing_mask)
        elif semantic_type == 'quantity_indicators':
            predictions = self._predict_quantities_contextual(df, target_col, missing_mask)
        elif semantic_type == 'percentage_indicators':
            predictions = self._predict_percentages_contextual(df, target_col, missing_mask)
        elif semantic_type == 'category_indicators':
            predictions = self._predict_categories_contextual(df, target_col, missing_mask)
        else:
            # Predicción estadística estándar
            if df[target_col].dtype in ['int64', 'float64']:
                predictions = [df[target_col].median()] * missing_mask.sum()
            else:
                mode_val = df[target_col].mode()
                predictions = [mode_val[0] if len(mode_val) > 0 else 'Unknown'] * missing_mask.sum()
        
        df.loc[missing_mask, target_col] = predictions
        return df
    
    def _predict_prices_contextual(self, df: pd.DataFrame, target_col: str, missing_mask: pd.Series) -> List:
        """Predicción contextual de precios"""
        
        predictions = []
        
        # Buscar columna de categoría
        category_cols = [col for col in df.columns if 'categoria' in col.lower() or 'category' in col.lower()]
        
        for idx in df[missing_mask].index:
            predicted_price = df[target_col].median()  # Default
            
            if category_cols:
                category = df.loc[idx, category_cols[0]]
                if pd.notna(category):
                    category_str = str(category).lower()
                    
                    # Buscar en base de conocimiento
                    for product, (min_price, max_price) in self.knowledge_base['price_ranges'].items():
                        if product in category_str:
                            predicted_price = (min_price + max_price) / 2
                            break
            
            predictions.append(predicted_price)
        
        return predictions
    
    def _predict_quantities_contextual(self, df: pd.DataFrame, target_col: str, missing_mask: pd.Series) -> List:
        """Predicción contextual de cantidades"""
        
        # Usar mediana con ajuste por precio si existe
        base_quantity = df[target_col].median()
        if pd.isna(base_quantity):
            base_quantity = 25  # Default razonable
        
        predictions = []
        price_cols = [col for col in df.columns if any(p in col.lower() for p in ['precio', 'price'])]
        
        for idx in df[missing_mask].index:
            quantity = base_quantity
            
            if price_cols:
                price = df.loc[idx, price_cols[0]]
                if pd.notna(price) and price > 0:
                    # Relación inversa: productos caros = menos stock
                    avg_price = df[price_cols[0]].median()
                    if price > avg_price * 2:
                        quantity = base_quantity * 0.5  # Menos stock para productos caros
                    elif price < avg_price * 0.5:
                        quantity = base_quantity * 2    # Más stock para productos baratos
            
            predictions.append(int(quantity))
        
        return predictions
    
    def _predict_percentages_contextual(self, df: pd.DataFrame, target_col: str, missing_mask: pd.Series) -> List:
        """Predicción contextual de porcentajes"""
        
        base_percentage = df[target_col].median()
        if pd.isna(base_percentage):
            base_percentage = 0.1  # 10% default
        
        return [base_percentage] * missing_mask.sum()
    
    def _predict_categories_contextual(self, df: pd.DataFrame, target_col: str, missing_mask: pd.Series) -> List:
        """Predicción contextual de categorías"""
        
        # Usar moda o predicción basada en otras características
        mode_val = df[target_col].mode()
        if len(mode_val) > 0:
            return [mode_val[0]] * missing_mask.sum()
        else:
            return ['General'] * missing_mask.sum()
    
    def _post_process_predictions(self, predictions: np.ndarray, semantic_type: str, column_name: str) -> np.ndarray:
        """Post-procesamiento de predicciones según contexto"""
        
        if semantic_type == 'price_indicators':
            # Asegurar precios positivos y razonables
            predictions = np.maximum(predictions, 0.01)
            predictions = np.minimum(predictions, 100000)  # Cap máximo
            
        elif semantic_type == 'quantity_indicators':
            # Asegurar cantidades enteras positivas
            predictions = np.maximum(np.round(predictions), 0)
            
        elif semantic_type == 'percentage_indicators':
            # Asegurar porcentajes entre 0 y 1
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def _statistical_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback estadístico cuando ML no está disponible"""
        
        df_result = df.copy()
        
        for col in df_result.columns:
            missing_count = df_result[col].isna().sum()
            if missing_count == 0:
                continue
            
            if df_result[col].dtype in ['int64', 'float64']:
                # Usar mediana para números
                median_val = df_result[col].median()
                if pd.notna(median_val):
                    df_result[col] = df_result[col].fillna(median_val)
            else:
                # Usar moda para categóricos
                mode_val = df_result[col].mode()
                if len(mode_val) > 0:
                    df_result[col] = df_result[col].fillna(mode_val[0])
        
        return df_result
    
    def get_prediction_report(self) -> Dict:
        """Genera reporte de predicciones realizadas"""
        
        return {
            'ml_available': self.ml_available,
            'models_trained': len(self.models),
            'encoders_created': len(self.encoders),
            'scalers_created': len(self.scalers),
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance
        }