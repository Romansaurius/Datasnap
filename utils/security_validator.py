"""
🔒 SECURITY VALIDATOR 🔒
Validador de seguridad avanzado con IA
"""

import pandas as pd
import re
from typing import List, Dict, Tuple

class SecurityValidator:
    """Validador de seguridad con IA"""
    
    def __init__(self):
        # Patrones de datos sensibles
        self.sensitive_patterns = {
            'credit_card': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            'dni': r'\b\d{8}[A-Z]\b',
            'social_security': r'\b\d{3}-\d{2}-\d{4}\b',
            'password': r'password|contraseña|clave',
            'api_key': r'[A-Za-z0-9]{32,}',
            'token': r'token|bearer|jwt'
        }
        
        # Patrones de inyección SQL
        self.sql_injection_patterns = [
            r"';.*--",
            r"union\s+select",
            r"drop\s+table",
            r"delete\s+from.*where.*1=1",
            r"insert\s+into.*values",
            r"update.*set.*where"
        ]
        
        # Patrones de XSS
        self.xss_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe.*?>",
            r"<object.*?>"
        ]
    
    def validate_dataframe_security(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida seguridad del DataFrame"""
        
        security_report = {
            'is_safe': True,
            'warnings': [],
            'critical_issues': [],
            'recommendations': []
        }
        
        # 1. Detectar datos sensibles
        sensitive_data = self._detect_sensitive_data(df)
        if sensitive_data:
            security_report['critical_issues'].extend(sensitive_data)
            security_report['is_safe'] = False
        
        # 2. Detectar inyección SQL
        sql_injection = self._detect_sql_injection(df)
        if sql_injection:
            security_report['critical_issues'].extend(sql_injection)
            security_report['is_safe'] = False
        
        # 3. Detectar XSS
        xss_attempts = self._detect_xss(df)
        if xss_attempts:
            security_report['warnings'].extend(xss_attempts)
        
        # 4. Validar estructura de datos
        structure_issues = self._validate_data_structure(df)
        if structure_issues:
            security_report['warnings'].extend(structure_issues)
        
        # 5. Generar recomendaciones
        security_report['recommendations'] = self._generate_security_recommendations(df)
        
        return security_report
    
    def _detect_sensitive_data(self, df: pd.DataFrame) -> List[str]:
        """Detecta datos sensibles"""
        issues = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Verificar nombres de columnas sensibles
            if any(sensitive in col_lower for sensitive in ['password', 'contraseña', 'tarjeta', 'dni']):
                issues.append(f"🚨 CRÍTICO: Columna sensible detectada: '{col}'")
            
            # Verificar contenido
            for _, value in df[col].items():
                if pd.isna(value):
                    continue
                    
                value_str = str(value)
                
                # Buscar patrones sensibles
                for pattern_name, pattern in self.sensitive_patterns.items():
                    if re.search(pattern, value_str, re.IGNORECASE):
                        issues.append(f"🚨 CRÍTICO: {pattern_name} detectado en columna '{col}'")
                        break
        
        return issues
    
    def _detect_sql_injection(self, df: pd.DataFrame) -> List[str]:
        """Detecta intentos de inyección SQL"""
        issues = []
        
        for col in df.columns:
            for _, value in df[col].items():
                if pd.isna(value):
                    continue
                    
                value_str = str(value).lower()
                
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        issues.append(f"🚨 CRÍTICO: Posible SQL injection en columna '{col}': {value_str[:50]}...")
                        break
        
        return issues
    
    def _detect_xss(self, df: pd.DataFrame) -> List[str]:
        """Detecta intentos de XSS"""
        issues = []
        
        for col in df.columns:
            for _, value in df[col].items():
                if pd.isna(value):
                    continue
                    
                value_str = str(value)
                
                for pattern in self.xss_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        issues.append(f"⚠️ Posible XSS en columna '{col}': {value_str[:50]}...")
                        break
        
        return issues
    
    def _validate_data_structure(self, df: pd.DataFrame) -> List[str]:
        """Valida estructura de datos"""
        issues = []
        
        # Verificar tamaño del DataFrame
        if len(df) > 100000:
            issues.append("⚠️ Dataset muy grande (>100k filas) - considerar paginación")
        
        # Verificar número de columnas
        if len(df.columns) > 50:
            issues.append("⚠️ Muchas columnas (>50) - considerar normalización")
        
        # Verificar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > len(df) * 0.1:
            issues.append(f"⚠️ Alto porcentaje de duplicados: {duplicates} filas")
        
        # Verificar valores nulos
        null_percentage = (df.isnull().sum().sum() / df.size) * 100
        if null_percentage > 30:
            issues.append(f"⚠️ Alto porcentaje de valores nulos: {null_percentage:.1f}%")
        
        return issues
    
    def _generate_security_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Genera recomendaciones de seguridad"""
        recommendations = []
        
        # Recomendaciones generales
        recommendations.append("🔒 Hashear contraseñas antes de almacenar")
        recommendations.append("🔒 Validar y sanitizar todos los inputs")
        recommendations.append("🔒 Implementar rate limiting")
        recommendations.append("🔒 Usar HTTPS para transmisión de datos")
        
        # Recomendaciones específicas por columnas
        for col in df.columns:
            col_lower = col.lower()
            
            if 'email' in col_lower:
                recommendations.append(f"📧 Validar formato de email en columna '{col}'")
            
            if 'phone' in col_lower or 'telefono' in col_lower:
                recommendations.append(f"📱 Validar formato de teléfono en columna '{col}'")
            
            if 'precio' in col_lower or 'price' in col_lower:
                recommendations.append(f"💰 Validar rangos de precio en columna '{col}'")
        
        return recommendations
    
    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitiza DataFrame eliminando contenido peligroso"""
        
        df_clean = df.copy()
        
        for col in df_clean.columns:
            for idx, value in df_clean[col].items():
                if pd.isna(value):
                    continue
                
                value_str = str(value)
                
                # Remover scripts maliciosos
                for pattern in self.xss_patterns:
                    value_str = re.sub(pattern, '', value_str, flags=re.IGNORECASE)
                
                # Remover SQL peligroso
                for pattern in self.sql_injection_patterns:
                    value_str = re.sub(pattern, '', value_str, flags=re.IGNORECASE)
                
                df_clean.loc[idx, col] = value_str
        
        return df_clean