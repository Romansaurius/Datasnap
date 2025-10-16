# DataSnap - Optimizador Universal de Datos con IA

## 🚀 Funcionalidades Principales

### ✨ Nuevas Características Integradas
- **Optimización Universal**: Funciona con CSV, TXT, XLSX, JSON y SQL
- **Inteligencia Artificial**: Corrección automática de datos con ML
- **Optimizador SQL Perfecto**: Normalización BCNF + detección de fraude
- **Estadísticas Completas**: Análisis detallado de calidad de datos
- **API REST**: Mantiene compatibilidad con Google Drive y MySQL

### 📊 Tipos de Archivos Soportados
- **CSV**: Limpieza, corrección de emails, normalización de nombres
- **TXT**: Procesamiento inteligente de texto estructurado
- **XLSX**: Optimización de hojas de cálculo Excel
- **JSON**: Normalización y validación de estructuras JSON
- **SQL**: Normalización completa, detección de fraude, optimización de rendimiento

### 🤖 Optimizaciones con IA
- **Corrección de Emails**: Detecta y corrige dominios malformados
- **Normalización de Nombres**: Capitalización inteligente
- **Predicción de Precios**: ML para valores faltantes por categoría
- **Detección de Fraude**: Análisis de patrones sospechosos
- **Generación de Datos**: Completa información faltante automáticamente

### 🔐 Características de Seguridad
- **Análisis de Vulnerabilidades**: Detecta SQL injection y exposición de datos
- **Encriptación de Passwords**: Hash automático con SHA2
- **Sistema de Auditoría**: Registro completo de cambios
- **Detección de Fraude**: Scoring automático de transacciones

## 🛠️ Instalación y Configuración

### Requisitos Previos
- Python 3.8+
- MySQL Database
- Google Drive API credentials

### Instalación Local
```bash
git clone <repository>
cd Datasnap-main
pip install -r requirements.txt
```

### Variables de Entorno
Copia `.env.example` a `.env` y configura:
```env
DB_HOST=tu_host_mysql
DB_USER=tu_usuario_mysql
DB_PASS=tu_password_mysql
DB_NAME=tu_base_datos
GCP_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
GDRIVE_FOLDER_ID=tu_folder_id_drive
```

### Despliegue en Render
1. Conecta tu repositorio a Render
2. Configura las variables de entorno en el dashboard
3. El `Procfile` está configurado para despliegue automático

## 📡 API Endpoints

### POST /procesar
Procesa archivos por ID desde la base de datos
```json
{
  "id": "123"
}
```

**Respuesta:**
```json
{
  "success": true,
  "archivo_id": "123",
  "archivo_optimizado": "contenido_optimizado",
  "nombre_archivo": "optimizado_archivo.csv",
  "estadisticas": {
    "archivo": {
      "tipo": "csv",
      "filas_originales": 1000,
      "columnas": 15,
      "tamaño_mb": 2.5
    },
    "calidad_datos": {
      "valores_nulos": 45,
      "duplicados": 12,
      "completitud": 95.2
    },
    "optimizaciones": {
      "mejoras_aplicadas": 8,
      "problemas_corregidos": 3,
      "confianza": 97.5
    }
  },
  "mejoras_aplicadas": [
    "🤖 IA: 15 datos corregidos automáticamente",
    "📧 Emails malformados corregidos",
    "💰 Precios anómalos detectados y corregidos"
  ]
}
```

### POST /procesar_drive
Procesa archivos directamente desde Google Drive
```json
{
  "drive_file_id": "google_drive_file_id"
}
```

### POST /upload_original
Sube archivos a Google Drive
```json
{
  "success": true,
  "drive_id": "file_id",
  "drive_link": "https://drive.google.com/..."
}
```

## 🎯 Ejemplos de Optimización

### CSV - Antes y Después
**Antes:**
```csv
nombre,email,precio
juan,juan@gmai.com,
maria,maria@hotmial.com,abc
```

**Después:**
```csv
nombre,email,precio
Juan,juan@gmail.com,99.99
Maria,maria@hotmail.com,99.99
```

### SQL - Optimización Completa
**Antes:** Base de datos desnormalizada con problemas de seguridad
**Después:** 
- Normalización BCNF completa
- Índices optimizados
- Sistema de auditoría
- Detección de fraude
- Procedimientos almacenados seguros

## 📈 Estadísticas Generadas

### Para todos los tipos de archivo:
- **Calidad de datos**: Completitud, valores nulos, duplicados
- **Análisis por columna**: Tipos, rangos, valores únicos
- **Optimizaciones aplicadas**: Lista detallada de mejoras

### Específico para SQL:
- **Tablas creadas**: Número de tablas normalizadas
- **Índices**: Cantidad de índices de rendimiento
- **Seguridad**: Score de vulnerabilidades
- **Normalización**: Nivel alcanzado (1NF, 2NF, 3NF, BCNF)

## 🔧 Arquitectura Técnica

### Optimizadores Especializados
- `UniversalDataOptimizer`: Para CSV, TXT, XLSX, JSON
- `PerfectSQLOptimizer`: Para bases de datos SQL

### Parsers Inteligentes
- Detección automática de tipos de columna
- Corrección de formatos
- Predicción de valores faltantes

### Sistema de Estadísticas
- Análisis en tiempo real
- Métricas de calidad
- Reportes detallados

## 🚀 Rendimiento

### Optimizaciones Implementadas
- **Procesamiento en memoria**: Pandas optimizado
- **Índices inteligentes**: Para consultas SQL rápidas
- **Caché de patrones**: Reutilización de correcciones
- **Procesamiento por lotes**: Para archivos grandes

### Escalabilidad
- Compatible con Render, Heroku, AWS
- Manejo eficiente de memoria
- Procesamiento asíncrono preparado

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Revisa la documentación de API
- Consulta los ejemplos de uso

---

**DataSnap** - Transformando datos caóticos en información perfecta con Inteligencia Artificial 🤖✨