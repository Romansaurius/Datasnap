#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizador CSV Avanzado para DataSnap IA - Versión Mejorada
Maneja archivos CSV con todos los errores posibles usando las mismas funciones inteligentes del XLSX
"""

import pandas as pd
import numpy as np
import re
import html
from datetime import datetime
from io import StringIO, BytesIO
import base64
from difflib import SequenceMatcher
import pycountry
from dateutil import parser

# IA UNIVERSAL - LIBRERÍAS AVANZADAS PARA DETECCIÓN INTELIGENTE
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class AdvancedCSVOptimizer:
    """OPTIMIZADOR CSV IA UNIVERSAL - DETECCIÓN INTELIGENTE CON MACHINE LEARNING"""

    def __init__(self):
        self.corrections_applied = []
        self.original_rows = 0
        self.final_rows = 0

        # CONFIGURACIÓN PARA ENTORNOS DE PRODUCCIÓN - DETECTAR AUTOMÁTICAMENTE
        self.use_heavy_ai = self._should_use_heavy_ai()
        print(f"[CSV OPTIMIZER] Heavy AI enabled: {self.use_heavy_ai}")

        # IA UNIVERSAL - MODELOS Y CONOCIMIENTO (LAZY LOADING)
        self._knowledge_base = None
        self._ml_models = None
        self._nlp_processor = None
        self._embedding_model = None

        # APRENDIZAJE CONTINUO
        self.learning_data = {
            'column_patterns': {},
            'data_type_mappings': {},
            'correction_history': []
        }

    def _should_use_heavy_ai(self) -> bool:
        """Determina si usar IA pesada - SIEMPRE HABILITADA PARA IA PERFECTA"""
        import os

        # FORZAR IA PESADA SIEMPRE - PARA IA PERFECTA
        print("[CSV OPTIMIZER] 🚀 IA PESADA SIEMPRE HABILITADA - MODO PERFECTO")

        # Verificar librerías disponibles
        try:
            import sklearn
            import transformers
            import spacy
            import sentence_transformers
            print("[CSV OPTIMIZER] ✅ All AI libraries available - PERFECT AI ENABLED")
            return True
        except ImportError as e:
            print(f"[CSV OPTIMIZER] ⚠️  Missing AI libraries: {e} - fallback to smart mode")
            # En lugar de deshabilitar completamente, usar modo inteligente
            return True  # Mantener habilitado, pero con fallbacks internos

    @property
    def knowledge_base(self):
        if self._knowledge_base is None:
            self._knowledge_base = self._initialize_knowledge_base()
        return self._knowledge_base

    @property
    def ml_models(self):
        if self._ml_models is None:
            self._ml_models = self._initialize_ml_models()
        return self._ml_models

    @property
    def nlp_processor(self):
        if self._nlp_processor is None:
            self._nlp_processor = self._initialize_nlp_processor()
        return self._nlp_processor

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = self._initialize_embedding_model()
        return self._embedding_model

    def _initialize_knowledge_base(self):
        """BASE DE CONOCIMIENTO UNIVERSAL DE TIPOS DE DATOS"""
        return {
            'data_types': {
                'personal_info': {
                    'patterns': ['name', 'nombre', 'apellido', 'fullname', 'first_name', 'last_name', 'usuario', 'user'],
                    'examples': ['Juan Pérez', 'María García', 'John Smith', 'Ana López'],
                    'validation_rules': ['contains_letters', 'proper_case_possible', 'no_special_chars']
                },
                'contact_info': {
                    'email': {
                        'patterns': ['email', 'mail', 'correo', 'e-mail'],
                        'examples': ['user@gmail.com', 'contact@company.com'],
                        'validation_rules': ['contains_@', 'contains_domain', 'valid_format']
                    },
                    'phone': {
                        'patterns': ['phone', 'telefono', 'tel', 'celular', 'movil', 'contacto'],
                        'examples': ['+54911234567', '(011) 1234-5678', '911234567'],
                        'validation_rules': ['contains_digits', 'length_7_15', 'may_contain_symbols']
                    }
                },
                'temporal': {
                    'date': {
                        'patterns': ['fecha', 'date', 'birth', 'nacimiento', 'created', 'updated'],
                        'examples': ['2024-01-15', '15/01/2024', '2024-01-15 10:30:00'],
                        'validation_rules': ['date_format', 'logical_date_range', 'no_future_dates']
                    },
                    'time': {
                        'patterns': ['hora', 'time', 'timestamp', 'hora_registro'],
                        'examples': ['14:30:00', '2:30 PM', '14:30'],
                        'validation_rules': ['time_format', 'valid_hours_minutes']
                    }
                },
                'quantitative': {
                    'monetary': {
                        'patterns': ['precio', 'price', 'costo', 'cost', 'salario', 'salary', 'monto', 'amount'],
                        'examples': ['$100.50', '€250,00', '1000', '45.99'],
                        'validation_rules': ['numeric', 'positive_values', 'currency_symbols_allowed']
                    },
                    'quantity': {
                        'patterns': ['cantidad', 'qty', 'quantity', 'stock', 'unidades', 'units'],
                        'examples': ['100', '50.5', '25', '0'],
                        'validation_rules': ['numeric', 'non_negative', 'reasonable_range']
                    },
                    'percentage': {
                        'patterns': ['porcentaje', 'percentage', 'rate', 'tasa', 'porciento'],
                        'examples': ['25%', '0.15', '100', '45.5%'],
                        'validation_rules': ['numeric', 'range_0_100', 'percentage_symbol_optional']
                    }
                },
                'categorical': {
                    'boolean': {
                        'patterns': ['activo', 'active', 'enabled', 'status', 'estado', 'flag'],
                        'examples': ['true', 'false', '1', '0', 'si', 'no', 'activo', 'inactivo'],
                        'validation_rules': ['boolean_values', 'case_insensitive']
                    },
                    'category': {
                        'patterns': ['tipo', 'type', 'categoria', 'category', 'clase', 'class', 'grupo'],
                        'examples': ['Premium', 'Básico', 'Estándar', 'VIP'],
                        'validation_rules': ['text_values', 'limited_unique_values']
                    }
                },
                'location': {
                    'country': {
                        'patterns': ['pais', 'country', 'nacion', 'nation', 'nacionalidad'],
                        'examples': ['España', 'Argentina', 'México', 'Colombia'],
                        'validation_rules': ['country_names', 'standard_spelling']
                    },
                    'city': {
                        'patterns': ['ciudad', 'city', 'municipio', 'localidad'],
                        'examples': ['Madrid', 'Buenos Aires', 'México DF', 'Bogotá'],
                        'validation_rules': ['city_names', 'geographic_location']
                    },
                    'postal_code': {
                        'patterns': ['codigo_postal', 'postal_code', 'zip', 'cp'],
                        'examples': ['28001', '10001', 'B2B 1A1', '28001-000'],
                        'validation_rules': ['alphanumeric', 'country_specific_format']
                    }
                },
                'identification': {
                    'id_number': {
                        'patterns': ['id', 'numero_id', 'identificacion', 'dni', 'cedula', 'pasaporte'],
                        'examples': ['12345678A', 'AB123456', 'P1234567'],
                        'validation_rules': ['alphanumeric', 'length_5_20', 'unique_values']
                    },
                    'social_security': {
                        'patterns': ['ssn', 'seguridad_social', 'numero_seguridad'],
                        'examples': ['123-45-6789', '123456789'],
                        'validation_rules': ['numeric_with_dashes', 'length_9_11']
                    }
                }
            },
            'common_patterns': {
                'spanish_keywords': ['nombre', 'apellido', 'telefono', 'email', 'direccion', 'ciudad', 'pais', 'fecha', 'precio', 'cantidad'],
                'english_keywords': ['name', 'phone', 'email', 'address', 'city', 'country', 'date', 'price', 'quantity'],
                'business_terms': ['customer', 'client', 'supplier', 'provider', 'product', 'service', 'order', 'invoice'],
                'technical_terms': ['id', 'code', 'number', 'status', 'type', 'category', 'description']
            }
        }

    def _initialize_ml_models(self):
        """INICIALIZAR MODELOS DE MACHINE LEARNING"""
        models = {}

        if SKLEARN_AVAILABLE:
            try:
                # Modelo de clustering para detectar patrones
                models['text_clusterer'] = {
                    'vectorizer': TfidfVectorizer(max_features=1000, stop_words='english'),
                    'clusterer': KMeans(n_clusters=10, random_state=42, n_init=10)
                }
                print("[IA] Modelo de clustering inicializado")
            except Exception as e:
                print(f"[IA WARNING] Error inicializando clustering: {e}")

        return models

    def _initialize_nlp_processor(self):
        """INICIALIZAR PROCESADOR NLP"""
        if SPACY_AVAILABLE:
            try:
                # Cargar modelo de spaCy para español
                self.nlp_es = spacy.load("es_core_news_sm")
                print("[IA] Procesador NLP español inicializado")
                return True
            except:
                try:
                    # Fallback a modelo inglés
                    self.nlp_en = spacy.load("en_core_web_sm")
                    print("[IA] Procesador NLP inglés inicializado")
                    return True
                except:
                    print("[IA WARNING] No se pudo cargar modelo spaCy")
                    return False
        return False

    def _initialize_embedding_model(self):
        """INICIALIZAR MODELO DE EMBEDDINGS PARA SIMILITUD SEMÁNTICA"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Modelo ligero para embeddings
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                print("[IA] Modelo de embeddings inicializado")
                return True
            except Exception as e:
                print(f"[IA WARNING] Error inicializando embeddings: {e}")
                return False
        return False

    def optimize_csv(self, csv_content: str) -> str:
        """Optimización CSV completa - devuelve string CSV válido"""

        try:
            # 1. Parse CSV content
            df = self.parse_csv_content(csv_content)

            # Check if parsing failed
            if 'error' in df.columns:
                error_msg = df['error'].iloc[0] if len(df) > 0 else "Unknown parsing error"
                print(f"[CSV ERROR] Parsing failed: {error_msg}")
                raise Exception(f"Error parsing CSV: {error_msg}")

            # 2. Clean and optimize data
            df = self.clean_data(df)

            # 3. Fix data types
            df = self.fix_data_types(df)

            # 4. Validate and correct values
            df = self.validate_and_correct_values(df)

            # 5. Remove duplicates and empty rows
            df = self.remove_duplicates_and_empty(df)

            # 6. Skip automatic column normalization to preserve original names
            # df = self.normalize_column_names(df)  # Disabled to keep meaningful names

            # Generate CSV string content
            from io import StringIO

            # Clean and prepare data for CSV
            df_clean = self._prepare_data_for_csv(df)

            # Create CSV string in memory
            csv_buffer = StringIO()
            df_clean.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_string = csv_buffer.getvalue()

            # VALIDACIÓN DE INTEGRIDAD: Verificar CSV generado
            if len(csv_string) > 10:
                # Verificación adicional: intentar leer el CSV generado
                try:
                    validation_df = pd.read_csv(StringIO(csv_string))
                    if len(validation_df) == len(df_clean):
                        self.corrections_applied.append("File integrity validated - read/write successful")
                        return csv_string
                    else:
                        print(f"[WARNING] Row count mismatch: expected {len(df_clean)}, got {len(validation_df)}")
                        return csv_string  # Still return, but log warning
                except Exception as validation_error:
                    print(f"[WARNING] Integrity validation failed: {validation_error}")
                    return csv_string  # Still return the file
            else:
                raise Exception(f"Generated CSV file invalid: size={len(csv_string)}")

        except Exception as e:
            print(f"[CSV ERROR] Exception during optimization: {str(e)}")
            import traceback
            print(f"[CSV ERROR] Traceback: {traceback.format_exc()}")
            self.corrections_applied.append(f"Error during optimization: {e}")
            raise Exception(f"Error: {e}")

    def parse_csv_content(self, content: str) -> pd.DataFrame:
        """Parse CSV content with multiple strategies - ROBUSTO para archivos corruptos"""

        # Strategy 1: Direct string parsing
        try:
            if isinstance(content, str):
                df = pd.read_csv(StringIO(content), engine='python')
                self.corrections_applied.append("CSV parsed from string (python engine)")
                return df
        except Exception as e:
            print(f"Error parsing string with python engine: {e}")

        # Strategy 2: Try with C engine
        try:
            if isinstance(content, str):
                df = pd.read_csv(StringIO(content), engine='c')
                self.corrections_applied.append("CSV parsed from string (c engine)")
                return df
        except Exception as e:
            print(f"Error parsing string with c engine: {e}")

        # Strategy 3: Try different separators
        separators = [',', ';', '\t', '|', ':']
        for sep in separators:
            try:
                df = pd.read_csv(StringIO(content), sep=sep, engine='python')
                self.corrections_applied.append(f"CSV parsed with separator '{sep}'")
                return df
            except Exception as e:
                print(f"Error parsing with separator {sep}: {e}")
                continue

        # Strategy 4: Base64 decode if needed
        try:
            if isinstance(content, str) and not content.startswith(('PK', '<?xml', '<')):
                # Try base64 decode
                try:
                    decoded = base64.b64decode(content)
                    df = pd.read_csv(BytesIO(decoded), engine='python')
                    self.corrections_applied.append("CSV parsed from base64")
                    return df
                except:
                    pass
        except Exception as e:
            print(f"Error parsing base64: {e}")

        # Strategy 5: Attempt recovery for corrupted files
        try:
            if isinstance(content, str) and len(content) > 10:
                # Try to find CSV-like structure
                lines = content.split('\n')
                if len(lines) > 1:
                    # Assume first line is header
                    headers = lines[0].split(',')
                    data = []
                    for line in lines[1:]:
                        if line.strip():
                            data.append(line.split(','))
                    df = pd.DataFrame(data, columns=headers)
                    self.corrections_applied.append("CSV recovered from corrupted file")
                    return df
        except Exception as e:
            print(f"Error recovering corrupted file: {e}")

        # If all strategies fail, return error
        error_msg = "Failed to parse CSV content with all available strategies"
        print(f"[CSV ERROR] {error_msg}")
        return pd.DataFrame({'error': [error_msg]})

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data values"""

        self.original_rows = len(df)

        # Replace common null representations
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', 'NA', 'na',
                      '#N/A', '#NULL!', 'nil', 'NIL', '-', '--', '---', 'undefined',
                      'UNDEFINED', 'missing', 'MISSING', '?', '??', '???']

        df = df.replace(null_values, np.nan)

        # Clean whitespace
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

        self.corrections_applied.append("Data cleaned and null values standardized")
        return df

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and optimize data types - completamente dinámico sin hardcodeo"""

        # Definir patrones inteligentes para detectar tipos de datos - MÁXIMA COBERTURA
        data_type_patterns = {
            'email': {
                'keywords': ['email', 'mail', 'correo', 'e-mail', 'contact', 'email_address', 'mail_address'],
                'validator': lambda x: self._is_email_like(str(x))
            },
            'phone': {
                'keywords': ['phone', 'telefono', 'tel', 'celular', 'movil', 'mobile', 'contacto', 'telefono_fijo', 'telefono_movil', 'numero_telefono'],
                'validator': lambda x: self._is_phone_like(str(x))
            },
            'date': {
                'keywords': ['fecha', 'date', 'birth', 'nacimiento', 'contrato', 'registro', 'created', 'updated', 'time', 'fecha_nacimiento', 'fecha_registro', 'fecha_contrato'],
                'validator': lambda x: self._is_date_like(str(x))
            },
            'age': {
                'keywords': ['edad', 'age', 'years', 'años', 'edad_actual'],
                'validator': lambda x: self._is_age_like(str(x))
            },
            'monetary': {
                'keywords': ['salario', 'salary', 'precio', 'price', 'anual', 'cost', 'valor', 'amount', 'total', 'subtotal', 'pago', 'importe', 'monto', 'costo', 'presupuesto', 'factura', 'recibo'],
                'validator': lambda x: self._is_monetary_like(str(x))
            },
            'quantity': {
                'keywords': ['stock', 'cantidad', 'qty', 'quantity', 'unidades', 'units', 'count', 'inventario', 'existencias', 'disponible'],
                'validator': lambda x: self._is_quantity_like(str(x))
            },
            'boolean': {
                'keywords': ['activo', 'active', 'enabled', 'visible', 'status', 'estado', 'flag', 'is_', 'has_', 'tiene_', 'es_', 'valid', 'valido'],
                'validator': lambda x: self._is_boolean_like(str(x))
            },
            'name': {
                'keywords': ['nombre', 'name', 'apellido', 'completo', 'student', 'customer', 'usuario', 'user', 'cliente', 'client', 'fullname', 'nombre_completo', 'apellido_paterno', 'apellido_materno'],
                'validator': lambda x: self._is_name_like(str(x))
            },
            'city': {
                'keywords': ['ciudad', 'city', 'residencia', 'location', 'address', 'municipio', 'localidad', 'provincia', 'estado'],
                'validator': lambda x: self._is_city_like(str(x))
            },
            'country': {
                'keywords': ['pais', 'country', 'nation', 'nacionalidad', 'nationality', 'pais_origen', 'pais_destino'],
                'validator': lambda x: self._is_country_like(str(x))
            },
            'gender': {
                'keywords': ['genero', 'gender', 'sexo', 'sex', 'sexo_biologico'],
                'validator': lambda x: self._is_gender_like(str(x))
            },
            'url': {
                'keywords': ['url', 'website', 'web', 'sitio', 'pagina', 'link', 'enlace', 'homepage', 'dominio'],
                'validator': lambda x: self._is_url_like(str(x))
            },
            'postal_code': {
                'keywords': ['codigo_postal', 'postal_code', 'zip', 'zip_code', 'cp', 'codigo'],
                'validator': lambda x: self._is_postal_code_like(str(x))
            },
            'id_number': {
                'keywords': ['id', 'numero_id', 'identificacion', 'dni', 'cedula', 'pasaporte', 'licencia', 'documento', 'numero_documento'],
                'validator': lambda x: self._is_id_number_like(str(x))
            },
            'percentage': {
                'keywords': ['porcentaje', 'percentage', 'porciento', 'rate', 'tasa', 'ratio'],
                'validator': lambda x: self._is_percentage_like(str(x))
            },
            'weight': {
                'keywords': ['peso', 'weight', 'kg', 'kilogramos', 'libras', 'lb', 'toneladas'],
                'validator': lambda x: self._is_weight_like(str(x))
            },
            'height': {
                'keywords': ['altura', 'height', 'estatura', 'cm', 'metros', 'pulgadas'],
                'validator': lambda x: self._is_height_like(str(x))
            },
            'numeric': {
                'keywords': ['numero', 'number', 'codigo', 'code', 'identificador', 'numero_cuenta', 'cuenta', 'numero_orden', 'orden', 'secuencia', 'secuence'],
                'validator': lambda x: self._is_numeric_like(str(x))
            }
        }

        for col in df.columns:
            col_lower = str(col).lower()

            # Detectar tipo de dato basado en nombre de columna y contenido
            detected_type = self._detect_column_type(col_lower, df[col], data_type_patterns)

            if detected_type:
                try:
                    if detected_type == 'email':
                        df[col] = self.fix_emails(df[col])
                    elif detected_type == 'phone':
                        df[col] = self.fix_phones(df[col])
                    elif detected_type == 'date':
                        df[col] = self.fix_dates(df[col])
                    elif detected_type == 'age':
                        df[col] = self.fix_ages(df[col])
                    elif detected_type == 'monetary':
                        df[col] = self.fix_monetary_values(df[col])
                    elif detected_type == 'quantity':
                        df[col] = self.fix_quantities(df[col])
                    elif detected_type == 'boolean':
                        df[col] = self.fix_booleans(df[col])
                    elif detected_type == 'name':
                        df[col] = self.fix_names(df[col])
                    elif detected_type == 'city':
                        df[col] = self.fix_cities(df[col])
                    elif detected_type == 'country':
                        df[col] = self.fix_countries(df[col])
                    elif detected_type == 'gender':
                        df[col] = self.fix_genders(df[col])
                    elif detected_type == 'url':
                        df[col] = self.fix_urls(df[col])
                    elif detected_type == 'postal_code':
                        df[col] = self.fix_postal_codes(df[col])
                    elif detected_type == 'id_number':
                        df[col] = self.fix_id_numbers(df[col])
                    elif detected_type == 'percentage':
                        df[col] = self.fix_percentages(df[col])
                    elif detected_type == 'weight':
                        df[col] = self.fix_weights(df[col])
                    elif detected_type == 'height':
                        df[col] = self.fix_heights(df[col])
                    elif detected_type == 'numeric':
                        df[col] = self.fix_numeric_values(df[col])
                except Exception as e:
                    print(f"[WARNING] Error fixing column {col} as {detected_type}: {e}")
                    # Continuar sin modificar la columna si hay error

        self.corrections_applied.append("Data types fixed and optimized (dynamic detection)")
        return df

    def _detect_column_type(self, col_name: str, series: pd.Series, patterns: dict) -> str:
        """IA UNIVERSAL PERFECTA: Detect column type using ALL ML, NLP, and semantic analysis"""

        col_name_clean = str(col_name).lower().strip()

        # 🚀 IA PERFECTA - TODAS LAS ESTRATEGIAS HABILITADAS
        print(f"[IA PERFECTA] Analizando columna: '{col_name}'")

        # ESTRATEGIA 1: BÚSQUEDA POR KEYWORDS EXPANDIDA
        detected_type = self._keyword_based_detection(col_name_clean, patterns)
        if detected_type:
            print(f"[IA PERFECTA] ✅ Detectado por keywords: {detected_type}")
            return detected_type

        # ESTRATEGIA 2: ANÁLISIS SEMÁNTICO CON NLP
        try:
            semantic_type = self._semantic_analysis_detection(col_name_clean, series)
            if semantic_type:
                print(f"[IA PERFECTA] ✅ Detectado por NLP semántico: {semantic_type}")
                return semantic_type
        except Exception as e:
            print(f"[IA PERFECTA] ⚠️  NLP falló: {e}")

        # ESTRATEGIA 3: ANÁLISIS DE CONTENIDO CON ML
        try:
            content_type = self._content_based_detection(series, patterns)
            if content_type:
                print(f"[IA PERFECTA] ✅ Detectado por ML contenido: {content_type}")
                return content_type
        except Exception as e:
            print(f"[IA PERFECTA] ⚠️  ML contenido falló: {e}")

        # ESTRATEGIA 4: CLUSTERING PARA PATRONES OCULTOS
        try:
            cluster_type = self._clustering_based_detection(col_name_clean, series)
            if cluster_type:
                print(f"[IA PERFECTA] ✅ Detectado por clustering: {cluster_type}")
                return cluster_type
        except Exception as e:
            print(f"[IA PERFECTA] ⚠️  Clustering falló: {e}")

        # ESTRATEGIA 5: APRENDIZAJE DE PATRONES ANTERIORES
        learned_type = self._learned_pattern_detection(col_name_clean, series)
        if learned_type:
            print(f"[IA PERFECTA] ✅ Detectado por aprendizaje: {learned_type}")
            return learned_type

        # ESTRATEGIA 6: ANÁLISIS AVANZADO CON EMBEDDINGS
        try:
            embedding_type = self._embedding_based_detection(col_name_clean, series)
            if embedding_type:
                print(f"[IA PERFECTA] ✅ Detectado por embeddings: {embedding_type}")
                return embedding_type
        except Exception as e:
            print(f"[IA PERFECTA] ⚠️  Embeddings fallaron: {e}")

        print(f"[IA PERFECTA] ❌ No se pudo detectar tipo para: '{col_name}'")
        return None

    def _keyword_based_detection(self, col_name: str, patterns: dict) -> str:
        """Detección basada en keywords con expansión inteligente"""
        # Buscar en patrones principales
        for data_type, config in patterns.items():
            if any(keyword in col_name for keyword in config['keywords']):
                return data_type

        # Buscar en base de conocimiento expandida
        for category, types in self.knowledge_base['data_types'].items():
            if isinstance(types, dict):
                for sub_type, config in types.items():
                    if 'patterns' in config and any(pattern in col_name for pattern in config['patterns']):
                        return sub_type

        # Búsqueda difusa con similitud
        best_match = None
        best_score = 0

        all_keywords = []
        for data_type, config in patterns.items():
            all_keywords.extend(config['keywords'])

        for keyword in all_keywords:
            score = self._semantic_similarity(col_name, keyword)
            if score > best_score and score > 0.7:  # Umbral de similitud
                best_match = data_type
                best_score = score

        return best_match

    def _semantic_analysis_detection(self, col_name: str, series: pd.Series) -> str:
        """Análisis semántico usando NLP y embeddings"""
        if not hasattr(self, 'nlp_processor') or not self.nlp_processor:
            return None

        try:
            # Análisis con spaCy si está disponible
            if hasattr(self, 'nlp_es'):
                doc = self.nlp_es(col_name)
                # Buscar entidades nombradas
                if doc.ents:
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                            return self._map_entity_to_data_type(ent.label_)

            # Análisis con embeddings semánticos
            if self.embedding_model:
                return self._embedding_based_classification(col_name, series)

        except Exception as e:
            print(f"[IA WARNING] Error en análisis semántico: {e}")

        return None

    def _content_based_detection(self, series: pd.Series, patterns: dict) -> str:
        """Detección basada en análisis de contenido con ML"""
        sample_values = series.dropna().astype(str).head(100)  # Más muestras para mejor análisis

        if len(sample_values) < 3:  # Necesitamos al menos 3 valores para análisis significativo
            return None

        # Análisis estadístico de los valores
        type_scores = {}
        for data_type, config in patterns.items():
            matches = sum(1 for val in sample_values if config['validator'](val))
            confidence = matches / len(sample_values)

            # Bonus por consistencia (todos los valores válidos o ninguno)
            if confidence == 1.0 or confidence == 0.0:
                confidence *= 1.2

            type_scores[data_type] = min(confidence, 1.0)  # Máximo 100%

        # Retornar tipo con mayor confianza (>60% para ser conservador)
        if type_scores:
            best_type, best_score = max(type_scores.items(), key=lambda x: x[1])
            if best_score >= 0.6:
                return best_type

        return None

    def _clustering_based_detection(self, col_name: str, series: pd.Series) -> str:
        """Detección usando clustering para encontrar patrones ocultos"""
        if not SKLEARN_AVAILABLE or 'text_clusterer' not in self.ml_models:
            return None

        try:
            sample_values = series.dropna().astype(str).head(50)
            if len(sample_values) < 10:  # Necesitamos suficientes datos para clustering
                return None

            # Vectorizar texto
            vectors = self.ml_models['text_clusterer']['vectorizer'].fit_transform(sample_values)

            # Clustering
            clusters = self.ml_models['text_clusterer']['clusterer'].fit_predict(vectors.toarray())

            # Analizar clusters para inferir tipo
            cluster_patterns = {}
            for i, cluster in enumerate(clusters):
                if cluster not in cluster_patterns:
                    cluster_patterns[cluster] = []
                cluster_patterns[cluster].append(sample_values.iloc[i])

            # Inferir tipo basado en patrones de cluster
            return self._analyze_cluster_patterns(cluster_patterns, col_name)

        except Exception as e:
            print(f"[IA WARNING] Error en clustering: {e}")
            return None

    def _learned_pattern_detection(self, col_name: str, series: pd.Series) -> str:
        """Detección basada en aprendizaje de patrones anteriores"""
        # Buscar en historial de aprendizaje
        if col_name in self.learning_data['column_patterns']:
            return self.learning_data['column_patterns'][col_name]

        # Buscar patrones similares aprendidos
        for learned_col, learned_type in self.learning_data['column_patterns'].items():
            if self._semantic_similarity(col_name, learned_col) > 0.8:
                return learned_type

        return None

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud semántica entre textos"""
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode([text1, text2])
                from sklearn.metrics.pairwise import cosine_similarity
                return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            except:
                pass

        # Fallback a SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _embedding_based_detection(self, col_name: str, series: pd.Series) -> str:
        """Clasificación usando embeddings semánticos - IA PERFECTA"""
        if not self.embedding_model:
            print("[IA PERFECTA] ❌ Embeddings no disponibles")
            return None

        try:
            print("[IA PERFECTA] 🔍 Analizando con embeddings semánticos...")

            # Crear embeddings para el nombre de columna
            col_embedding = self.embedding_model.encode([col_name])[0]

            # Comparar con embeddings conocidos de tipos de datos
            type_embeddings = {}
            for data_type, config in self.knowledge_base['data_types'].items():
                if isinstance(config, dict) and 'patterns' in config:
                    for pattern in config['patterns'][:5]:  # Usar primeros 5 patrones para mejor precisión
                        pattern_emb = self.embedding_model.encode([pattern])[0]
                        similarity = np.dot(col_embedding, pattern_emb) / (np.linalg.norm(col_embedding) * np.linalg.norm(pattern_emb))
                        type_embeddings[data_type] = max(type_embeddings.get(data_type, 0), similarity)

            if type_embeddings:
                best_type = max(type_embeddings.items(), key=lambda x: x[1])
                if best_type[1] > 0.6:  # Umbral de similitud más flexible para IA perfecta
                    print(f"[IA PERFECTA] 🎯 Embeddings detectaron: {best_type[0]} (similitud: {best_type[1]:.3f})")
                    return best_type[0]

            print("[IA PERFECTA] 📊 Embeddings no encontraron coincidencia suficiente")

        except Exception as e:
            print(f"[IA PERFECTA] ❌ Error en embeddings: {e}")

        return None

    def _map_entity_to_data_type(self, entity_label: str) -> str:
        """Mapear etiquetas de entidades de spaCy a tipos de datos"""
        mapping = {
            'PERSON': 'name',
            'ORG': 'name',  # Organizaciones pueden ser tratadas como nombres
            'GPE': 'city',  # Entidades geopolíticas
            'LOC': 'city'   # Localizaciones
        }
        return mapping.get(entity_label)

    def _analyze_cluster_patterns(self, cluster_patterns: dict, col_name: str) -> str:
        """Analizar patrones de clusters para inferir tipo de dato"""
        # Esta es una implementación simplificada
        # En producción, se usaría análisis más sofisticado

        for cluster, values in cluster_patterns.items():
            if len(values) < 3:
                continue

            # Analizar patrones en el cluster
            sample_value = values[0]

            # Patrones simples de inferencia
            if '@' in sample_value and '.' in sample_value:
                return 'email'
            elif re.match(r'^\+?\d', sample_value) and len([c for c in sample_value if c.isdigit()]) >= 7:
                return 'phone'
            elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', sample_value):
                return 'date'
            elif sample_value.replace('.', '').replace(',', '').replace('$', '').replace('€', '').isdigit():
                return 'monetary'

        return None

    def learn_from_correction(self, col_name: str, detected_type: str, series: pd.Series):
        """APRENDIZAJE CONTINUO: Aprender de correcciones aplicadas"""
        self.learning_data['column_patterns'][col_name] = detected_type

        # Registrar en historial
        self.learning_data['correction_history'].append({
            'column': col_name,
            'type': detected_type,
            'timestamp': datetime.now(),
            'sample_values': series.dropna().astype(str).head(3).tolist()
        })

        # Limitar historial para evitar crecimiento excesivo
        if len(self.learning_data['correction_history']) > 1000:
            self.learning_data['correction_history'] = self.learning_data['correction_history'][-500:]

    def fix_emails(self, series: pd.Series) -> pd.Series:
        """Fix email addresses using smart correction"""
        series = series.astype(str)

        # Apply smart email correction
        series = series.apply(self._smart_email_correction)

        return series

    def _smart_email_correction(self, email):
        """Smart email correction without breaking valid emails"""
        if pd.isna(email) or str(email).strip() == '':
            return email

        email_str = str(email).lower().strip()

        # Remove mailto: prefix if present
        if email_str.startswith('mailto:'):
            email_str = email_str.replace('mailto:', '')

        # If email doesn't have @, add @gmail.com
        if '@' not in email_str:
            return f"{email_str}@gmail.com"

        # If email ends with @, add gmail.com
        if email_str.endswith('@'):
            return f"{email_str}gmail.com"

        # Split email into user and domain parts
        if '@' in email_str:
            user_part, domain_part = email_str.split('@', 1)

            # Fix common domain issues
            if domain_part == 'gmai' or domain_part == 'gmail':
                domain_part = 'gmail.com'
            elif domain_part == 'yahoo' or domain_part == 'yahooo':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmail' or domain_part == 'hotmial':
                domain_part = 'hotmail.com'
            elif domain_part == 'outlook' or domain_part == 'outlok':
                domain_part = 'outlook.com'
            elif domain_part == 'gmailcom':  # Missing dot
                domain_part = 'gmail.com'
            elif domain_part == 'yahoocom':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmailcom':
                domain_part = 'hotmail.com'
            elif domain_part.endswith('.comm'):  # Extra 'm'
                domain_part = domain_part.replace('.comm', '.com')
            elif domain_part == 'gmai.com':
                domain_part = 'gmail.com'
            elif domain_part == 'yahooo.com':
                domain_part = 'yahoo.com'
            elif domain_part == 'hotmial.com':
                domain_part = 'hotmail.com'
            elif domain_part == 'outlok.com':
                domain_part = 'outlook.com'
            elif domain_part == 'gmial.com':
                domain_part = 'gmail.com'

            # Reconstruct email
            email_str = f"{user_part}@{domain_part}"

        # Final validation - if it looks like an email now, return it
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$', email_str):
            return email_str

        return email_str

    def fix_phones(self, series: pd.Series) -> pd.Series:
        """Fix phone numbers"""
        series = series.astype(str)

        # Remove invalid phone indicators
        series = series.replace(['invalid_phone', 'invalid', 'sin_telefono', 'no_phone'], np.nan)

        # Clean phone format
        series = series.str.replace(r'[^\\d+\\-\\s()]', '', regex=True)

        # Keep phones that have reasonable length
        series = series.where(series.str.len().between(5, 25), series)

        return series

    def fix_dates(self, series: pd.Series) -> pd.Series:
        """Fix date values"""
        series = series.astype(str)

        # Apply smart date conversion
        series = series.apply(self._convert_date_format)

        return series

    def _convert_date_format(self, date_str):
        """Convert various date formats to YYYY-MM-DD with intelligent validation"""
        if pd.isna(date_str) or str(date_str).strip() == '':
            return np.nan

        date_str = str(date_str).strip()

        # Handle invalid date indicators
        if date_str.lower() in ['nan', 'invalid_date', 'never', 'null', 'none']:
            return np.nan

        try:
            # Convert DD/MM/YYYY to YYYY-MM-DD
            if re.match(r'^\\d{1,2}/\\d{1,2}/\\d{4}$', date_str):
                parts = date_str.split('/')
                if len(parts) == 3:
                    day, month, year = map(int, parts)

                    # Validate and fix invalid dates
                    if month > 12:
                        month = 12
                    if month < 1:
                        month = 1

                    # Fix impossible days
                    if day > 31:
                        day = 31
                    if day < 1:
                        day = 1

                    # Handle February 30/31
                    if month == 2 and day > 29:
                        day = 28

                    # Handle months with 30 days
                    if month in [4, 6, 9, 11] and day > 30:
                        day = 30

                    return f"{year}-{month:02d}-{day:02d}"

            # Already in YYYY-MM-DD format - validate
            elif re.match(r'^\\d{4}-\\d{1,2}-\\d{1,2}$', date_str):
                parts = date_str.split('-')
                if len(parts) == 3:
                    year, month, day = map(int, parts)

                    # Validate and fix
                    if month > 12:
                        month = 12
                    if month < 1:
                        month = 1
                    if day > 31:
                        day = 31
                    if day < 1:
                        day = 1

                    # Handle February
                    if month == 2 and day > 29:
                        day = 28

                    # Handle months with 30 days
                    if month in [4, 6, 9, 11] and day > 30:
                        day = 30

                    return f"{year}-{month:02d}-{day:02d}"

            # Try to parse with dateutil as fallback
            try:
                parsed_date = parser.parse(date_str, dayfirst=True)
                return parsed_date.strftime('%Y-%m-%d')
            except:
                return np.nan

        except (ValueError, TypeError):
            return np.nan

        return np.nan

    def fix_ages(self, series: pd.Series) -> pd.Series:
        """Fix age values"""
        series = pd.to_numeric(series, errors='coerce')

        # Remove invalid ages
        series = series.where((series >= 0) & (series <= 120), np.nan)

        return series

    def fix_monetary_values(self, series: pd.Series) -> pd.Series:
        """Fix monetary values"""
        series = series.astype(str)

        # Remove currency symbols and clean
        series = series.str.replace(r'[$€£¥₹]', '', regex=True)
        series = series.str.replace(',', '', regex=False)

        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')

        # Remove negative values for prices/salaries
        series = series.where(series >= 0, np.nan)

        return series

    def fix_quantities(self, series: pd.Series) -> pd.Series:
        """Fix quantity values"""
        series = pd.to_numeric(series, errors='coerce')

        # Remove negative quantities
        series = series.where(series >= 0, np.nan)

        return series

    def fix_numeric_values(self, series: pd.Series) -> pd.Series:
        """Fix generic numeric values"""
        series = pd.to_numeric(series, errors='coerce')

        # Remove extreme values that are likely errors
        series = series.where((series >= -999999) & (series <= 999999), np.nan)

        return series

    def fix_urls(self, series: pd.Series) -> pd.Series:
        """Fix URL values"""
        series = series.astype(str)

        def _fix_single_url(url):
            if pd.isna(url) or str(url).strip() == '':
                return url

            url_str = str(url).strip()

            # Add protocol if missing
            if url_str.startswith('www.'):
                url_str = 'https://' + url_str
            elif not url_str.startswith(('http://', 'https://', 'ftp://')):
                url_str = 'https://' + url_str

            # Basic URL validation
            if '://' in url_str and '.' in url_str.split('://')[-1]:
                return url_str

            return url_str

        return series.apply(_fix_single_url)
def fix_postal_codes(self, series: pd.Series) -> pd.Series:
    """Fix postal code values"""
    series = series.astype(str)

    def _fix_single_postal(postal):
        if pd.isna(postal) or str(postal).strip() == '':
            return postal

        postal_str = str(postal).strip().upper()

        # Remove spaces and special chars except hyphen
        postal_str = re.sub(r'[^A-Z0-9-]', '', postal_str)

        # Basic formatting for common patterns
        if len(postal_str) == 5 and postal_str.isdigit():  # US 5-digit
            return postal_str
        elif len(postal_str) == 9 and postal_str.isdigit():  # US 9-digit
            return f"{postal_str[:5]}-{postal_str[5:]}"
        elif len(postal_str) == 6 and postal_str.isalnum():  # Canadian
            return f"{postal_str[:3]} {postal_str[3:]}"

        return postal_str

    return series.apply(_fix_single_postal)

def fix_id_numbers(self, series: pd.Series) -> pd.Series:
    """Fix ID number values"""
    series = series.astype(str)

    def _fix_single_id(id_num):
        if pd.isna(id_num) or str(id_num).strip() == '':
            return id_num

        id_str = str(id_num).strip().upper()

        # Remove invalid characters but keep letters, numbers, and some symbols
        id_str = re.sub(r'[^A-Z0-9\-_\.]', '', id_str)

        # Ensure reasonable length
        if 3 <= len(id_str) <= 20:
            return id_str

        return id_str

    return series.apply(_fix_single_id)

def fix_percentages(self, series: pd.Series) -> pd.Series:
    """Fix percentage values"""
    series = series.astype(str)

    def _fix_single_percentage(perc):
        if pd.isna(perc) or str(perc).strip() == '':
            return perc

        perc_str = str(perc).strip()

        # Remove % symbol and convert to numeric
        perc_str = perc_str.replace('%', '').strip()

        try:
            val = float(perc_str)
            # Ensure reasonable percentage range
            if 0 <= val <= 100:
                return f"{val}%"
            else:
                return f"{min(100, max(0, val))}%"
        except:
            return perc_str

    return series.apply(_fix_single_percentage)

def fix_weights(self, series: pd.Series) -> pd.Series:
    """Fix weight values"""
    series = series.astype(str)

    def _fix_single_weight(weight):
        if pd.isna(weight) or str(weight).strip() == '':
            return weight

        weight_str = str(weight).strip().lower()

        # Extract numeric value
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', weight_str)
        if numeric_match:
            val = float(numeric_match.group(1))

            # Determine unit
            if 'kg' in weight_str or 'kilogramos' in weight_str:
                return f"{val} kg"
            elif 'lb' in weight_str or 'libras' in weight_str:
                return f"{val} lb"
            elif 'ton' in weight_str or 'toneladas' in weight_str:
                return f"{val} ton"
            elif 'g' in weight_str or 'gramos' in weight_str:
                return f"{val} g"
            else:
                return f"{val} kg"  # Default to kg

        return weight_str

    return series.apply(_fix_single_weight)

def fix_heights(self, series: pd.Series) -> pd.Series:
    """Fix height values"""
    series = series.astype(str)

    def _fix_single_height(height):
        if pd.isna(height) or str(height).strip() == '':
            return height

        height_str = str(height).strip().lower()

        # Extract numeric value
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', height_str)
        if numeric_match:
            val = float(numeric_match.group(1))

            # Determine unit and convert to cm if needed
            if 'cm' in height_str:
                return f"{val} cm"
            elif 'm' in height_str or 'metros' in height_str:
                return f"{val * 100} cm"
            elif 'inch' in height_str or 'pulgadas' in height_str:
                return f"{val * 2.54} cm"
            elif 'ft' in height_str or 'pies' in height_str:
                return f"{val * 30.48} cm"
            else:
                # Assume cm if no unit specified
                return f"{val} cm"

        return height_str

    return series.apply(_fix_single_height)

def fix_booleans(self, series: pd.Series) -> pd.Series:
    """Fix boolean values"""
    series = series.astype(str).str.lower()

    # Map various boolean representations
    boolean_map = {
        'true': True, '1': True, 'yes': True, 'si': True, 'sí': True, 'y': True,
        'active': True, 'activo': True, 'enabled': True, 'on': True,
        'false': False, '0': False, 'no': False, 'n': False, 'inactive': False,
        'inactivo': False, 'disabled': False, 'off': False, 'invalid': False
    }

    series = series.map(boolean_map)

    return series

def fix_names(self, series: pd.Series) -> pd.Series:
    """Fix name values properly"""
    series = series.astype(str)

    # Remove empty names
    series = series.replace(['', '0', '1', '2', '3', '4', '5'], np.nan)

    # Apply smart name correction
    series = series.apply(self._smart_name_correction)

    # Fix common name issues
    series = series.str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
    series = series.str.strip()

    return series

def fix_cities(self, series: pd.Series) -> pd.Series:
    """Fix city names"""
    series = series.astype(str)

    # Apply smart city correction
    series = series.apply(self._smart_city_correction)

    return series

def fix_countries(self, series: pd.Series) -> pd.Series:
    """Fix country names intelligently"""
    series = series.astype(str)

    # Apply smart country correction
    series = series.apply(self._smart_country_correction)

    return series

def fix_genders(self, series: pd.Series) -> pd.Series:
    """Fix gender values intelligently"""
    series = series.astype(str)

    # Apply smart gender correction
    series = series.apply(self._smart_gender_correction)

    return series

def validate_and_correct_values(self, df: pd.DataFrame) -> pd.DataFrame:
    """Additional validation and corrections"""

    # Remove rows where all values are null
    df = df.dropna(how='all')

    self.corrections_applied.append("Values validated and corrected")
    return df

def remove_duplicates_and_empty(self, df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and empty rows"""

    initial_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with too many null values (more than 90% null)
    threshold = max(1, len(df.columns) * 0.1)  # Keep rows with at least 10% valid data
    df = df.dropna(thresh=threshold)

    final_rows = len(df)
    removed_rows = initial_rows - final_rows

    if removed_rows > 0:
        self.corrections_applied.append(f"Removed {removed_rows} duplicate/empty rows")

    self.final_rows = final_rows
    return df

def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names"""

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    df.columns = df.columns.str.replace(r'_+', '_', regex=True)
    df.columns = df.columns.str.strip('_')

    self.corrections_applied.append("Column names normalized")
    return df

def _prepare_data_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for CSV export - MÁXIMA OPTIMIZACIÓN Y LIMPIEZA PROFUNDA"""

    # Create a clean copy
    df_clean = df.copy()

    # OPTIMIZACIÓN 1: Eliminar columnas completamente vacías
    df_clean = df_clean.dropna(axis=1, how='all')
    if len(df_clean.columns) == 0:
        # Si todas las columnas estaban vacías, crear una columna dummy
        df_clean = pd.DataFrame({'datos': ['Archivo vacío procesado']})

    # OPTIMIZACIÓN 2: Limpiar nombres de columnas con máxima compatibilidad
    new_columns = []
    for i, col in enumerate(df_clean.columns):
        if pd.isna(col) or str(col).strip() == '' or 'Unnamed' in str(col):
            # Usar nombres más descriptivos basados en contenido
            sample_values = df_clean[col].dropna().head(3).astype(str)
            if len(sample_values) > 0:
                # Intentar inferir tipo de columna del contenido
                first_val = sample_values.iloc[0]
                if '@' in first_val and '.' in first_val:
                    new_columns.append(f'email_{i+1}')
                elif any(char.isdigit() for char in first_val) and len([c for c in first_val if c.isdigit()]) >= 7:
                    new_columns.append(f'telefono_{i+1}')
                elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', first_val):
                    new_columns.append(f'fecha_{i+1}')
                else:
                    new_columns.append(f'columna_{i+1}')
            else:
                new_columns.append(f'columna_{i+1}')
        else:
            # Clean existing column name while preserving meaning
            clean_name = str(col).strip()[:50]  # Limit length for CSV
            # Replace problematic characters with underscores but keep structure
            clean_name = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ]', '_', clean_name)  # Allow Spanish chars
            clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
            clean_name = re.sub(r'_+', '_', clean_name)   # Remove multiple underscores
            clean_name = clean_name.strip('_')            # Remove leading/trailing underscores

            # If cleaning removed everything, use original with safe chars only
            if not clean_name:
                clean_name = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]', '_', str(col))[:30]
                if not clean_name:
                    clean_name = f'columna_{i+1}'

            new_columns.append(clean_name)

    # Ensure unique column names
    final_columns = []
    for col in new_columns:
        if col in final_columns:
            counter = 1
            while f"{col}_{counter}" in final_columns:
                counter += 1
            final_columns.append(f"{col}_{counter}")
        else:
            final_columns.append(col)

    df_clean.columns = final_columns

    # OPTIMIZACIÓN 3: Limpieza profunda de datos
    for col in df_clean.columns:
        # Convert all values to strings first
        df_clean[col] = df_clean[col].astype(str)

        # Replace various null representations
        null_patterns = ['nan', 'NaN', 'None', 'none', 'NULL', 'null', 'N/A', 'n/a', '']
        df_clean[col] = df_clean[col].replace(null_patterns, '')

        # Remove control characters and problematic chars
        df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', str(x)))

        # Normalize whitespace
        df_clean[col] = df_clean[col].str.strip()
        df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)

        # Limit string length to prevent CSV issues
        df_clean[col] = df_clean[col].str[:32767]

    # OPTIMIZACIÓN 4: Eliminar filas duplicadas inteligentes
    # Primero intentar deduplicar por todas las columnas
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()

    # Si hay muchas filas, hacer deduplicación más agresiva
    if len(df_clean) > 10000:
        # Mantener solo filas con al menos 50% de datos no vacíos
        min_valid_data = max(1, len(df_clean.columns) // 2)
        df_clean = df_clean.dropna(thresh=min_valid_data)

    # OPTIMIZACIÓN 5: Comprimir datos numéricos donde sea posible
    for col in df_clean.columns:
        # Intentar convertir a numérico si parece numérico
        if df_clean[col].astype(str).str.match(r'^[\d\s\.,-]+$').any():
            try:
                # Intentar conversión numérica
                numeric_series = pd.to_numeric(df_clean[col].astype(str).str.replace(',', '.'), errors='coerce')
                if numeric_series.notna().sum() > len(numeric_series) * 0.8:  # 80% válido
                    df_clean[col] = numeric_series
            except:
                pass  # Mantener como string si falla

    # OPTIMIZACIÓN 6: Eliminar filas completamente vacías al final
    df_clean = df_clean.replace('', np.nan)
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.fillna('')  # Fill remaining NaN with empty strings

    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows

    self.corrections_applied.append(f"Data optimized: {removed_rows} rows removed, {len(df_clean.columns)} columns cleaned")

    return df_clean

def _smart_city_correction(self, city):
    """Smart city name correction using similarity matching"""
    if pd.isna(city) or str(city).strip() == '':
        return city

    city_str = str(city).strip().lower()

    # Common cities database for correction
    common_cities = [
        'madrid', 'barcelona', 'valencia', 'sevilla', 'bilbao', 'zaragoza',
        'málaga', 'murcia', 'palma', 'córdoba', 'valladolid', 'vigo',
        'paris', 'london', 'berlin', 'rome', 'amsterdam', 'vienna',
        'moscow', 'beijing', 'tokyo', 'new york', 'los angeles', 'chicago',
        'houston', 'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas'
    ]

    # If city is very short or truncated, try to find best match
    if len(city_str) >= 4 and len(city_str) <= 12:
        best_match = self._find_best_city_match(city_str, common_cities)
        if best_match:
            return best_match.title()

    return str(city).title()

def _find_best_city_match(self, city: str, cities: list) -> str:
    """Find best matching city using similarity"""
    best_ratio = 0
    best_match = None

    for candidate in cities:
        # Calculate similarity ratio
        ratio = self._similarity_ratio(city, candidate)
        if ratio > best_ratio and ratio >= 0.7:  # 70% similarity threshold
            best_ratio = ratio
            best_match = candidate

    return best_match

def _similarity_ratio(self, str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()

def _smart_email_correction(self, email):
    """Smart email correction without breaking valid emails"""
    if pd.isna(email) or str(email).strip() == '':
        return email

    email_str = str(email).lower().strip()

    # Remove mailto: prefix if present
    if email_str.startswith('mailto:'):
        email_str = email_str.replace('mailto:', '')

    # If email doesn't have @, add @gmail.com
    if '@' not in email_str:
        return f"{email_str}@gmail.com"

    # If email ends with @, add gmail.com
    if email_str.endswith('@'):
        return f"{email_str}gmail.com"

    # Split email into user and domain parts
    if '@' in email_str:
        user_part, domain_part = email_str.split('@', 1)

        # Fix common domain issues
        if domain_part == 'gmai' or domain_part == 'gmail':
            domain_part = 'gmail.com'
        elif domain_part == 'yahoo' or domain_part == 'yahooo':
            domain_part = 'yahoo.com'
        elif domain_part == 'hotmail' or domain_part == 'hotmial':
            domain_part = 'hotmail.com'
        elif domain_part == 'outlook' or domain_part == 'outlok':
            domain_part = 'outlook.com'
        elif domain_part == 'gmailcom':  # Missing dot
            domain_part = 'gmail.com'
        elif domain_part == 'yahoocom':
            domain_part = 'yahoo.com'
        elif domain_part == 'hotmailcom':
            domain_part = 'hotmail.com'
        elif domain_part.endswith('.comm'):  # Extra 'm'
            domain_part = domain_part.replace('.comm', '.com')
        elif domain_part == 'gmai.com':
            domain_part = 'gmail.com'
        elif domain_part == 'yahooo.com':
            domain_part = 'yahoo.com'
        elif domain_part == 'hotmial.com':
            domain_part = 'hotmail.com'
        elif domain_part == 'outlok.com':
            domain_part = 'outlook.com'
        elif domain_part == 'gmial.com':
            domain_part = 'gmail.com'

        # Reconstruct email
        email_str = f"{user_part}@{domain_part}"

    # Final validation - if it looks like an email now, return it
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$', email_str):
        return email_str

    return email_str

def _smart_name_correction(self, name):
    """Smart name correction handling mixed cases properly"""
    if pd.isna(name) or str(name).strip() == '':
        return name

    name_str = str(name).strip()

    # Handle mixed case names like "ROMAN gomez" -> "Roman Gomez"
    words = name_str.split()
    corrected_words = []

    for word in words:
        if word.isupper() or word.islower():
            # Convert to proper case
            corrected_words.append(word.capitalize())
        else:
            # Keep mixed case as is (might be intentional like "McDonald")
            corrected_words.append(word)

    return ' '.join(corrected_words)

def _smart_country_correction(self, country):
    """Smart country correction using pycountry and similarity"""
    if pd.isna(country) or str(country).strip() == '':
        return country

    country_str = str(country).strip().lower()

    # Common Spanish variations
    spanish_variations = {
        'españa': 'España',
        'spain': 'España',
        'esp': 'España',
        'espana': 'España',
        'spanish': 'España',
        'es': 'España'
    }

    if country_str in spanish_variations:
        return spanish_variations[country_str]

    # Try to find country using pycountry
    try:
        # Try by name
        try:
            country_obj = pycountry.countries.lookup(country_str)
            return country_obj.name
        except:
            pass

        # Try by alpha_2 code
        try:
            country_obj = pycountry.countries.get(alpha_2=country_str.upper())
            if country_obj:
                return country_obj.name
        except:
            pass

        # Try by alpha_3 code
        try:
            country_obj = pycountry.countries.get(alpha_3=country_str.upper())
            if country_obj:
                return country_obj.name
        except:
            pass

        # Fuzzy matching with common countries
        common_countries = ['España', 'France', 'Germany', 'Italy', 'Portugal', 'United Kingdom', 'United States']
        best_match = self._find_best_match(country_str, [c.lower() for c in common_countries])
        if best_match:
            return common_countries[[c.lower() for c in common_countries].index(best_match)]

    except Exception:
        pass

    # Return capitalized version as fallback
    return str(country).title()

def _smart_gender_correction(self, gender):
    """Smart gender correction"""
    if pd.isna(gender) or str(gender).strip() == '':
        return gender

    gender_str = str(gender).strip().lower()

    # Gender mappings
    male_variations = ['m', 'male', 'masculino', 'hombre', 'h', 'man', 'boy', 'niño']
    female_variations = ['f', 'female', 'femenino', 'mujer', 'woman', 'girl', 'niña']

    if gender_str in male_variations:
        return 'Masculino'
    elif gender_str in female_variations:
        return 'Femenino'

    # Return original if not recognized
    return str(gender).title()

def _find_best_match(self, target: str, candidates: list, threshold: float = 0.6) -> str:
    """Find best matching string using similarity"""
    best_ratio = 0
    best_match = None

    for candidate in candidates:
        ratio = SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = candidate

    return best_match

def _is_date_like(self, value: str) -> bool:
    """Check if value looks like a date"""
    if not value or value.lower() in ['nan', 'none', 'null', '']:
        return False

    # Check for date patterns
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
        r'\d{1,2}\s+(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}',
        r'\d{4}-\d{2}-\d{2}',  # ISO format
    ]

    return any(re.search(pattern, value, re.IGNORECASE) for pattern in date_patterns)

def _is_age_like(self, value: str) -> bool:
    """Check if value looks like an age"""
    try:
        num = float(value)
        return 0 <= num <= 150
    except:
        return False

def _is_monetary_like(self, value: str) -> bool:
    """Check if value looks like monetary amount"""
    if not value:
        return False

    # Check for currency symbols or decimal patterns
    if any(char in value for char in ['$', '€', '£', '¥', '₹']):
        return True

    try:
        num = float(value.replace(',', '').replace(' ', ''))
        return num >= 0  # Monetary values are usually positive
    except:
        return False

def _is_quantity_like(self, value: str) -> bool:
    """Check if value looks like a quantity"""
    try:
        num = float(value)
        return num >= 0  # Quantities are usually non-negative
    except:
        return False

def _is_boolean_like(self, value: str) -> bool:
    """Check if value looks like a boolean"""
    value_lower = value.lower().strip()
    boolean_values = [
        'true', 'false', '1', '0', 'yes', 'no', 'si', 'sí', 'on', 'off',
        'active', 'inactive', 'enabled', 'disabled', 'activo', 'inactivo'
    ]
    return value_lower in boolean_values

def _is_name_like(self, value: str) -> bool:
    """Check if value looks like a name"""
    if not value or len(value) < 2:
        return False

    # Names typically have letters and spaces, no special chars except apostrophes
    if not re.match(r"^[a-zA-Z\s'áéíóúñÁÉÍÓÚÑ-]+$", value):
        return False

    # Names are usually 2-50 characters
    return 2 <= len(value) <= 50

def _is_city_like(self, value: str) -> bool:
    """Check if value looks like a city name"""
    if not value or len(value) < 2:
        return False

    # Cities can have letters, spaces, hyphens
    if not re.match(r"^[a-zA-Z\s\-'áéíóúñÁÉÍÓÚÑ]+$", value):
        return False

    # Cities are usually 2-100 characters
    return 2 <= len(value) <= 100

def _is_country_like(self, value: str) -> bool:
    """Check if value looks like a country name"""
    if not value or len(value) < 2:
        return False

    # Countries can have letters, spaces
    if not re.match(r"^[a-zA-Z\s\-'áéíóúñÁÉÍÓÚÑ]+$", value):
        return False

    # Countries are usually 2-50 characters
    return 2 <= len(value) <= 50

def _is_gender_like(self, value: str) -> bool:
    """Check if value looks like a gender"""
    value_lower = value.lower().strip()
    gender_values = [
        'm', 'f', 'male', 'female', 'masculino', 'femenino',
        'hombre', 'mujer', 'man', 'woman', 'boy', 'girl', 'niño', 'niña'
    ]
    return value_lower in gender_values

def _is_email_like(self, value: str) -> bool:
    """Check if value looks like an email"""
    if not value or len(value) < 5:
        return False
    value = str(value).strip()
    # Check for @ and domain structure
    if '@' not in value:
        return False
    local, domain = value.split('@', 1)
    return len(local) > 0 and '.' in domain and len(domain) > 3

def _is_phone_like(self, value: str) -> bool:
    """Check if value looks like a phone number"""
    if not value:
        return False
    value = str(value).strip()
    # Count digits
    digits = [c for c in value if c.isdigit()]
    return len(digits) >= 7 and len(digits) <= 15

def _is_url_like(self, value: str) -> bool:
    """Check if value looks like a URL"""
    if not value or len(value) < 4:
        return False
    value = str(value).lower().strip()
    return value.startswith(('http://', 'https://', 'www.', 'ftp://'))

def _is_postal_code_like(self, value: str) -> bool:
    """Check if value looks like a postal code"""
    if not value:
        return False
    value = str(value).strip()
    # Common postal code patterns (5 digits, 5+4, alphanumeric, etc.)
    patterns = [
        r'^\d{5}$',  # US 5-digit
        r'^\d{5}-\d{4}$',  # US ZIP+4
        r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$',  # Canadian
        r'^\d{4,6}$',  # European 4-6 digits
        r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$',  # UK
    ]
    return any(re.match(pattern, value.upper()) for pattern in patterns)

def _is_id_number_like(self, value: str) -> bool:
    """Check if value looks like an ID number"""
    if not value:
        return False
    value = str(value).strip()
    # ID numbers are typically alphanumeric with specific patterns
    if len(value) < 4 or len(value) > 20:
        return False
    # Should contain mix of letters and numbers
    has_digit = any(c.isdigit() for c in value)
    has_letter = any(c.isalpha() for c in value)
    return has_digit and has_letter

def _is_percentage_like(self, value: str) -> bool:
    """Check if value looks like a percentage"""
    if not value:
        return False
    value = str(value).strip()
    # Check for % symbol or decimal between 0-100
    if '%' in value:
        return True
    try:
        num = float(value)
        return 0 <= num <= 100
    except:
        return False

def _is_weight_like(self, value: str) -> bool:
    """Check if value looks like a weight"""
    if not value:
        return False
    value = str(value).lower().strip()
    # Check for weight units
    weight_units = ['kg', 'kilogramos', 'libras', 'lb', 'toneladas', 'ton', 'gramos', 'g']
    has_unit = any(unit in value for unit in weight_units)
    if has_unit:
        return True
    # Check for reasonable weight numbers
    try:
        num = float(re.sub(r'[^\d.]', '', value))
        return 0.1 <= num <= 10000  # Reasonable weight range
    except:
        return False

def _is_height_like(self, value: str) -> bool:
    """Check if value looks like a height"""
    if not value:
        return False
    value = str(value).lower().strip()
    # Check for height units
    height_units = ['cm', 'metros', 'm', 'pulgadas', 'inch', 'pies', 'ft']
    has_unit = any(unit in value for unit in height_units)
    if has_unit:
        return True
    # Check for reasonable height numbers
    try:
        num = float(re.sub(r'[^\d.]', '', value))
        return 20 <= num <= 300  # Reasonable height range in cm
    except:
        return False

def _is_numeric_like(self, value: str) -> bool:
    """Check if value looks like a generic number"""
    try:
        float(value.replace(',', '').replace(' ', ''))
        return True
    except:
        return False

def get_optimization_summary(self) -> str:
    """Get summary of optimizations applied"""
    summary = f"CSV Optimization Summary:\n"
    summary += f"- Original rows: {self.original_rows}\n"
    summary += f"- Final rows: {self.final_rows}\n"
    summary += f"- Rows removed: {self.original_rows - self.final_rows}\n"
    summary += f"- Corrections applied:\n"

    for correction in self.corrections_applied:
        summary += f"  + {correction}\n"

    return summary
               
