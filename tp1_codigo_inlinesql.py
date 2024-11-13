"""
Nombre del grupo: inlineSQL
Integrantes: Wencelblat Agustin (878/23)
En este archivo se encuentra el códgo para los puntos G, H e I del TP. 
- En el ejercicio G se importan los datos a las tablas segun nuestro DER y MER.
- En el ejercicio H se realizan distintos reportes sobre la informacion de nuestras tablas.
- En el ejercicio I se implementan visualizaciones segun lo pedido en las consignas.
"""

import pandas as pd
from inline_sql import sql, sql_val
import seaborn as sns
import matplotlib.pyplot as plt

carpeta = "D:/Facultad/Laboratorio de Datos/TP 1 - inlineSQL/Tablas Limpias/"
sedes = pd.read_csv(carpeta+'lista-sedes-limpia.csv')
secciones = pd.read_csv(carpeta+'lista-secciones-limpia.csv')
migraciones = pd.read_csv(carpeta+'datos_migraciones_limpia.csv')
datos = pd.read_csv(carpeta+'lista-sedes-datos-limpia.csv')

#%%
###############
# Ejercicio G #
###############

seccion = sql^"""
SELECT sede_id, sede_desc_castellano
FROM secciones
"""

sede = sql^"""
SELECT sede_id, pais_iso_3
FROM datos
"""

pais = sql^"""
SELECT DISTINCT pais_iso_3, region_geografica, pais_castellano
FROM datos
"""

redes = sql^"""
SELECT sede_id,
       CASE 
           WHEN red_social LIKE '%twitter%' THEN 'Twitter'
           WHEN red_social LIKE '%facebook%' THEN 'Facebook'
           WHEN red_social LIKE '%instagram%' THEN 'Instagram'
           ELSE 'Otra'
       END AS red_social_nombre,
       TRIM(red_social) AS red_social_link
FROM (
    -- Divide la columna redes_sociales en múltiples filas por sede_id
    SELECT sede_id, UNNEST(STRING_TO_ARRAY(redes_sociales, ' // ')) AS red_social
    FROM lista_sedes_datos
) AS social_links
WHERE red_social IS NOT NULL AND red_social <> '';
"""

migraciones=sql^"""
SELECT DISTINCT "Country Dest Code" AS codigo_pais_destino,
       "Country Origin Code" AS codigo_pais_origen,
       "1960 [1960]" AS migracion_1960,
       "1970 [1970]" AS migracion_1970,
       "1980 [1980]" AS migracion_1980,
       "1990 [1990]" AS migracion_1990,
       "2000 [2000]" AS migracion_2000
FROM datos_migraciones;
"""
#%%
###############
# Ejercicio H #
###############

# Consigna I: 
    
consigna1 = sql^"""
SELECT 
    p.pais_castellano AS pais,
    COUNT(DISTINCT s.sede_id) AS cantidad_sedes,
    AVG(seccion_count.num_secciones) AS promedio_secciones,
    SUM(CASE 
            WHEN m.codigo_pais_destino = p.pais_iso_3 THEN m.migracion_2000
            WHEN m.codigo_pais_origen = p.pais_iso_3 THEN -m.migracion_2000
            ELSE 0 
        END) AS flujo_migratorio_neto
FROM pais AS p
LEFT JOIN sede AS s ON p.pais_iso_3 = s.pais_iso_3
LEFT JOIN (
    SELECT sede_id, COUNT(*) AS num_secciones
    FROM seccion
    GROUP BY sede_id
) AS seccion_count ON s.sede_id = seccion_count.sede_id
LEFT JOIN migraciones AS m 
    ON m.codigo_pais_destino = p.pais_iso_3 OR m.codigo_pais_origen = p.pais_iso_3
GROUP BY p.pais_castellano
ORDER BY cantidad_sedes DESC, pais ASC;
"""

#%% Consigna II: 
    
consigna2 = sql^"""
SELECT 
    p.region_geografica AS region,
    COUNT(DISTINCT p.pais_iso_3) AS cantidad_paises_con_sede,
    AVG(CASE 
            WHEN m.codigo_pais_destino = p.pais_iso_3 THEN m.migracion_2000
            ELSE NULL 
        END) AS promedio_flujo_migratorio_2000
FROM pais AS p
JOIN sede AS s ON p.pais_iso_3 = s.pais_iso_3
LEFT JOIN migraciones AS m ON m.codigo_pais_origen = 'ARG' AND m.codigo_pais_destino = p.pais_iso_3
GROUP BY p.region_geografica
ORDER BY promedio_flujo_migratorio_2000 DESC;
"""

#%% Consigna III:
    
consigna3 = sql^"""
SELECT 
    p.pais_castellano AS pais,
    COUNT(DISTINCT r.red_social_nombre) AS cantidad_tipos_redes
FROM sede AS s
JOIN pais AS p ON s.pais_iso_3 = p.pais_iso_3
JOIN redes AS r ON s.sede_id = r.sede_id
GROUP BY p.pais_castellano
ORDER BY cantidad_tipos_redes DESC;
"""

#%% Consigna IV: 
consigna3 = sql^"""
SELECT 
    p.pais_castellano AS pais,
    s.sede_desc_castellano AS sede,
    r.red_social_nombre AS tipo_red_social,
    r.red_social_link AS url
FROM redes AS r
JOIN sede AS s ON r.sede_id = s.sede_id
JOIN pais AS p ON s.pais_iso_3 = p.pais_iso_3
ORDER BY pais ASC, sede ASC, tipo_red_social ASC, url ASC;
"""
#%%
###############
# Ejercicio I #
###############

# Consigna I: 
    
sedes_por_region = sql^"""
SELECT 
    p.region_geografica AS region,
    COUNT(s.sede_id) AS cantidad_sedes
FROM sede AS s
JOIN pais AS p ON s.pais_iso_3 = p.pais_iso_3
GROUP BY p.region_geografica
ORDER BY cantidad_sedes DESC;
"""

plt.figure(figsize=(10, 6))
sns.barplot(data=sedes_por_region, x='region', y='cantidad_sedes', palette='Blues_d')
plt.title('Cantidad de Sedes por Región Geográfica')
plt.xlabel('Región Geográfica')
plt.ylabel('Cantidad de Sedes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Consigna II: 
    
promedios_flujo_region = sql^"""
SELECT 
    p.region_geografica AS region,
    AVG((m.migracion_1960 + m.migracion_1970 + m.migracion_1980 + m.migracion_1990 + m.migracion_2000) / 5) AS promedio_flujo_migratorio
FROM migraciones AS m
JOIN sede AS s ON m.codigo_pais_destino = s.pais_iso_3
JOIN pais AS p ON s.pais_iso_3 = p.pais_iso_3
WHERE m.codigo_pais_origen = 'ARG'
GROUP BY p.region_geografica, m.codigo_pais_destino
ORDER BY p.region_geografica;
"""

plt.figure(figsize=(12, 6))
sns.boxplot(data=promedios_flujo_region, x='region', y='promedio_flujo_migratorio', order=promedios_flujo_region.groupby('region')['promedio_flujo_migratorio'].median().sort_values().index)
plt.title('Distribución del Promedio de Flujo Migratorio hacia Argentina por Región Geográfica')
plt.xlabel('Región Geográfica')
plt.ylabel('Promedio de Flujo Migratorio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Consigna III:
    
flujo_sedes = sql^"""
SELECT 
    p.pais_castellano AS pais,
    m.migracion_2000 AS flujo_migratorio_2000,
    COUNT(s.sede_id) AS cantidad_sedes
FROM migraciones AS m
LEFT JOIN sede AS s ON m.codigo_pais_destino = s.pais_iso_3
LEFT JOIN pais AS p ON s.pais_iso_3 = p.pais_iso_3
WHERE m.codigo_pais_origen = 'ARG'
GROUP BY p.pais_castellano, m.migracion_2000
HAVING flujo_migratorio_2000 IS NOT NULL
ORDER BY flujo_migratorio_2000 DESC;
"""

plt.figure(figsize=(10, 6))
sns.boxplot(data=flujo_sedes, x='cantidad_sedes', y='flujo_migratorio_2000', palette='Set3')
plt.title('Distribución del Flujo Migratorio hacia Argentina (Año 2000) por Cantidad de Sedes en el Exterior')
plt.xlabel('Cantidad de Sedes de Argentina')
plt.ylabel('Flujo Migratorio hacia Argentina (Año 2000)')
plt.tight_layout()
plt.show()