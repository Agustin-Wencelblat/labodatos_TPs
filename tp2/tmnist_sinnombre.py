# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:48:33 2024

@author: agusw
"""
#%%
# Nombre de Grupo: Sin Nombre
# Integrantes:
# Wencelblat, Agustin - LU: 878/23
# Bejarano, Kevin - LU: 657/23
# El codigo esta en orden, segun el ejercicio al que corresponde cada consigna.
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

file_path = 'D://TMNIST_Data.csv'  
tmnist_data = pd.read_csv(file_path)

#%% Definicion de funciones:
def promedios(lista_digitos, agrupadas):
    fig, axes = plt.subplots(1, len(lista_digitos), figsize=(5 * len(lista_digitos), 5))
    for i, digito in enumerate(lista_digitos):
        digito_medio = agrupadas.loc[digito].values.reshape(28, 28)
        axes[i].imshow(digito_medio, cmap='gray')
        axes[i].set_title(f'Dígito promedio: {digito}')
        axes[i].axis('off')
    plt.show()

def diferencias(lista_digitos, agrupadas, filas=1):
    pares_digitos = [(lista_digitos[i], lista_digitos[j]) for i in range(len(lista_digitos)) for j in range(i + 1, len(lista_digitos))]
    num_pares = len(pares_digitos)
    
    columnas = (num_pares + filas - 1) // filas  
    
    fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 5 * filas))
    axes = axes.flatten() if num_pares > 1 else [axes]  
    
    for i, (digito1, digito2) in enumerate(pares_digitos):
        diferencia_absoluta = abs(agrupadas.loc[digito1] - agrupadas.loc[digito2]).values.reshape(28, 28)
        axes[i].imshow(diferencia_absoluta, cmap='hot')
        axes[i].set_title(f'Diferencia: {digito1} y {digito2}')
        axes[i].axis('off')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.show()

def visualizar_ejemplos_varianza(data, digitos, muestra=9):
    for digito in digitos:
        digito_data = data[data['labels'] == digito].iloc[:, 1:].astype(float)
        
        digito_muestra = digito_data.sample(muestra)
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Ejemplos de dígito {digito}", fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            img_data = digito_muestra.iloc[i].values.reshape(28, 28)
            ax.imshow(img_data, cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        varianza_pixeles = digito_data.var(axis=0).values.reshape(28, 28)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(varianza_pixeles, cmap='hot')
        plt.title(f"Varianza por píxel para el dígito {digito}", fontsize=14)
        plt.colorbar(label="Varianza")
        plt.axis('off')
        plt.show()

def evaluar_knn(X_train, X_test, y_train, y_test, atributos_seleccionados, k):

    X_train_sel = X_train[atributos_seleccionados]
    X_test_sel = X_test[atributos_seleccionados]
    
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_sel, y_train)

    y_train_pred = knn_model.predict(X_train_sel)
    y_test_pred = knn_model.predict(X_test_sel)
    

    train_exactitud = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=1)
    train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=1)
    
    test_exactitud = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=1)
    
    return {
        "train": (train_exactitud, train_precision, train_recall, train_f1),
        "test": (test_exactitud, test_precision, test_recall, test_f1)
    }

def analizar_por_atributos(X_train, X_test, y_train, y_test, varianzas, rangos_atributos):
    resultados = []
    k = 9
    varianzas_ordenadas = varianzas_bin.sort_values(ascending=False)

    for n_atributos in rangos_atributos:
        atributos_seleccionados = varianzas_ordenadas.iloc[:n_atributos].index

        X_train_sel = X_train[atributos_seleccionados]
        X_test_sel = X_test[atributos_seleccionados]

        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train_sel, y_train)

        y_train_pred = knn_model.predict(X_train_sel)
        y_test_pred = knn_model.predict(X_test_sel)

        train_exactitud = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=1)
        train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=1)
        train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=1)

        test_exactitud = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=1)
        test_recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=1)
        test_precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=1)

        resultados.append({
            "n_atributos": n_atributos,
            "train_exactitud": train_exactitud,
            "train_f1": train_f1,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "test_exactitud": test_exactitud,
            "test_f1": test_f1,
            "test_recall": test_recall,
            "test_precision": test_precision,
        })

    return pd.DataFrame(resultados)

#%%
###########################################
# PARTE 1: Analisis exploratorio de datos #
###########################################

#%%

print("Informacion del Dataframe:")
print(tmnist_data.info())

print("\nPrimeras filas del Dataframe:")
print(tmnist_data.head())

print("\nClases en el dataset (digitos):", tmnist_data['labels'].unique())
print("Cantidad de ejemplos por digito:")
print(tmnist_data['labels'].value_counts())


#%%
muestra_imagenes = tmnist_data.sample(9) 

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    img_data = muestra_imagenes.iloc[i, 2:].astype(float).values.reshape(28, 28)
    label = muestra_imagenes.iloc[i, 1]  
    ax.imshow(img_data, cmap='gray')
    ax.set_title(f'Digito: {label}')
    ax.axis('off')

plt.tight_layout()
plt.show()

#%%

data = tmnist_data.iloc[:, 2:]

varianzas = data.var().values.reshape(28, 28)

plt.figure(figsize=(8, 8))

sns.heatmap(varianzas, cmap="YlOrBr", square=True, cbar_kws={'label': 'Varianza'},
            linewidths=0.5, linecolor='black', vmin=0, vmax=np.max(varianzas),
            mask=(varianzas == 0))  
plt.title("Mapa de calor de la varianza de los pixeles")
plt.xlabel("Píxeles (eje X)")
plt.ylabel("Píxeles (eje Y)")
plt.show()


#%%
mascara_cero1 = (varianzas != 0)

plt.figure(figsize=(8, 8))
sns.heatmap(varianzas, cmap="YlOrBr", square=True, cbar_kws={'label': 'Varianza','shrink': 0.8},
            linewidths=0.5, linecolor='black', mask=mascara_cero1)
plt.title("Mapa de calor de la varianza de los píxeles (Mascara cero)")
plt.xlabel("Píxeles (eje X)")
plt.ylabel("Píxeles (eje Y)")
plt.show()

pixeles_con_varianza_cero = np.sum(varianzas == 0)

print(f"Número de píxeles con varianza cero: {pixeles_con_varianza_cero}")


#%% 

data_num = tmnist_data.drop(columns=['names'])

agrupadas = data_num.groupby("labels").mean()
varianzas = data_num.groupby("labels").var()

promedios([1, 3, 8], agrupadas)
diferencias([1, 3, 8], agrupadas)

promedios([8, 9, 6, 3], agrupadas)
diferencias([8, 9, 6, 3], agrupadas, filas=2)

promedios([1, 7], agrupadas)
diferencias([1, 7], agrupadas)

#%%

visualizar_ejemplos_varianza(data_num, [0, 5, 9])

#%%
###########################################
# PARTE 2: Clasificacion Binaria          #
###########################################

#%%

data_binaria = tmnist_data[tmnist_data['labels'].isin([0, 1])]

conteo_clases = data_binaria['labels'].value_counts()
print("Cantidad de muestras por clase en el subconjunto binario:\n", conteo_clases)

#%%
X = data_binaria.iloc[:, 2:].astype(float)  
y = data_binaria['labels']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

varianzas_bin = X_train.var(axis=0)

varianzas2 = X_train.var(axis=0).values.reshape(28, 28) 

plt.figure(figsize=(8, 8))
sns.heatmap(varianzas2, cmap="YlOrBr", square=True, cbar_kws={'label': 'Varianza','shrink': 0.8},
            linewidths=0.5, linecolor='black')
plt.title("Mapa de calor de la varianza de los píxeles en el conjunto de entrenamiento")
plt.xlabel("Píxeles (eje X)")
plt.ylabel("Píxeles (eje Y)")
plt.show()

mascara_ceros = (varianzas2 != 0)

plt.figure(figsize=(8, 8))
sns.heatmap(varianzas2, cmap="YlOrBr", square=True, cbar_kws={'label': 'Varianza','shrink': 0.8},
            linewidths=0.5, linecolor='black', mask=mascara_ceros)
plt.title("Mapa de calor de la varianza de los píxeles en el conjunto de entrenamiento (con máscara de ceros)")
plt.xlabel("Píxeles (eje X)")
plt.ylabel("Píxeles (eje Y)")
plt.show()

pixeles_con_varianza_cero = np.sum(varianzas2 == 0)

print(f"Número de píxeles con varianza cero: {pixeles_con_varianza_cero}")

    
#%%
varianzas_ordenadas = varianzas_bin.sort_values(ascending=False,)

top_3 = varianzas_ordenadas.iloc[:3].index
top_50_52 = varianzas_ordenadas.iloc[49:52].index
top_100_102 = varianzas_ordenadas.iloc[99:102].index
print(top_3, top_50_52, top_100_102)

#%% 

atributos_seleccionados = [
    varianzas_bin.nlargest(534).index,
    [435, 491, 463],
    [357, 548, 402],
    [523, 238, 629]
]

atributos_seleccionados = [
    X_train.columns[atributos] if isinstance(atributos[0], int) else atributos
    for atributos in atributos_seleccionados
]


for atributos in atributos_seleccionados:
    resultados = []
    for k in range(1, 21):
        metricas = evaluar_knn(X_train, X_test, y_train, y_test, atributos, k)
        resultados.append({
            "k": k,
            "train_exactitud": metricas["train"][0],
            "train_precision": metricas["train"][1],
            "train_recall": metricas["train"][2],
            "train_f1": metricas["train"][3],
            "test_exactitud": metricas["test"][0],
            "test_precision": metricas["test"][1],
            "test_recall": metricas["test"][2],
            "test_f1": metricas["test"][3],
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_resultados['k'], df_resultados['train_exactitud'], marker='o', label="Train")
    plt.plot(df_resultados['k'], df_resultados['test_exactitud'], marker='o', label="Test")
    plt.title("Exactitud vs. k")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.xticks(ticks=df_resultados['k'])
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(df_resultados['k'], df_resultados['train_precision'], marker='o', label="Train")
    plt.plot(df_resultados['k'], df_resultados['test_precision'], marker='o', label="Test")
    plt.title("Precisión vs. k")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Precisión")
    plt.legend()
    plt.xticks(ticks=df_resultados['k'])
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(df_resultados['k'], df_resultados['train_recall'], marker='o', label="Train")
    plt.plot(df_resultados['k'], df_resultados['test_recall'], marker='o', label="Test")
    plt.title("Recall vs. k")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("Recall")
    plt.legend()
    plt.xticks(ticks=df_resultados['k'])
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(df_resultados['k'], df_resultados['train_f1'], marker='o', label="Train")
    plt.plot(df_resultados['k'], df_resultados['test_f1'], marker='o', label="Test")
    plt.title("F1-score vs. k")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("F1-score")
    plt.legend()
    plt.xticks(ticks=df_resultados['k'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

#%%

rangos_atributos = range(1, 20, 1)  

resultados = analizar_por_atributos(X_train, X_test, y_train, y_test, varianzas, rangos_atributos)

plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(resultados["n_atributos"], resultados["train_exactitud"], marker='o', label="Train Exactitud", color='blue')
plt.plot(resultados["n_atributos"], resultados["test_exactitud"], marker='o', label="Test Exactitud", color='red')
plt.xticks(ticks=resultados["n_atributos"], labels=resultados["n_atributos"])
plt.xlabel("Número de atributos seleccionados")
plt.ylabel("Exactitud")
plt.title("Exactitud vs Número de atributos")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(resultados["n_atributos"], resultados["train_f1"], marker='o', label="Train F1", color='blue')
plt.plot(resultados["n_atributos"], resultados["test_f1"], marker='o', label="Test F1", color='red')
plt.xticks(ticks=resultados["n_atributos"], labels=resultados["n_atributos"])
plt.xlabel("Número de atributos seleccionados")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Número de atributos")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(resultados["n_atributos"], resultados["train_recall"], marker='o', label="Train Recall", color='blue')
plt.plot(resultados["n_atributos"], resultados["test_recall"], marker='o', label="Test Recall", color='red')
plt.xticks(ticks=resultados["n_atributos"], labels=resultados["n_atributos"])
plt.xlabel("Número de atributos seleccionados")
plt.ylabel("Recall")
plt.title("Recall vs Número de atributos")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(resultados["n_atributos"], resultados["train_precision"], marker='o', label="Train Precisión", color='blue')
plt.plot(resultados["n_atributos"], resultados["test_precision"], marker='o', label="Test Precisión", color='red')
plt.xticks(ticks=resultados["n_atributos"], labels=resultados["n_atributos"])
plt.xlabel("Número de atributos seleccionados")
plt.ylabel("Precisión")
plt.title("Precisión vs Número de atributos")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
###########################################
# PARTE 3: Clasificacion Multiclase       #
###########################################

#%%

# Inciso a: Separamos el conjunto de datos en desarrollo (dev) (80%) y validación (held-out) (20%)
X = tmnist_data.drop(["labels", "names"], axis=1)  # Eliminar la columna 'labels' y conservar solo las columnas de píxeles
y = tmnist_data.labels  # Seleccionar la columna 'labels' como la variable objetivo

X_dev, X_held_out, y_dev, y_held_out = train_test_split(X, y, random_state=1, test_size=0.2)

#%%

# Inciso b: Ajustar un modelo de árbol de decisión con distintas profundidades (1-10)

# Listas para almacenar la profundidad del árbol, exactitud en train y exactitud en test
max_depths = []
train_accuracy_scores = []
test_accuracy_scores = []

# Probar con distintas profundidades de 1 a 10
for depth in range(1, 11):
    max_depths.append(depth)
    
    # Crear y ajustar el modelo de árbol de decisión con la profundidad especificada
    arbol_elegido = DecisionTreeClassifier(max_depth=depth)
    arbol_elegido.fit(X_dev, y_dev)
    
    # Predicción en el conjunto de entrenamiento (train)
    y_train_pred = arbol_elegido.predict(X_dev)
    train_accuracy = accuracy_score(y_dev, y_train_pred)
    train_accuracy_scores.append(train_accuracy)
    
    # Predicción en el conjunto de validación (test)
    y_test_pred = arbol_elegido.predict(X_held_out)
    test_accuracy = accuracy_score(y_held_out, y_test_pred)
    test_accuracy_scores.append(test_accuracy)
    
    # Mostrar resultados para cada profundidad
    print(f"Profundidad {depth}:")
    print(f"  - Exactitud en Train: {train_accuracy:.4f}")
    print(f"  - Exactitud en Test:  {test_accuracy:.4f}")

# Graficar la exactitud en función de la profundidad para train y test
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracy_scores, marker='o', label="Exactitud en Train", color='blue')
plt.plot(max_depths, test_accuracy_scores, marker='o', label="Exactitud en Test", color='orange')
plt.title("Exactitud en Train y Test vs. Profundidad del Árbol")
plt.xlabel("Profundidad del Árbol")
plt.ylabel("Exactitud")
plt.legend()
plt.grid(True)
plt.show()

#%%

# Inciso c: Validación cruzada para distintos árboles de decisión con hiperparámetros variados

# Definir los valores de profundidad y los criterios de división
profundidades = list(range(1, 11))  # Profundidades de 1 a 10
criterios = ['gini', 'entropy']  # Criterios de división
k_folds = 5
kf = KFold(n_splits=k_folds)

# Crear matrices para almacenar resultados: una por criterio
resultados_train = {crit: np.zeros((k_folds, len(profundidades))) for crit in criterios}
resultados_test = {crit: np.zeros((k_folds, len(profundidades))) for crit in criterios}

# Realizar validación cruzada para cada criterio
for criterio in criterios:
    for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
        # Dividir los datos en conjunto de entrenamiento y prueba para el fold actual
        X_train, X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        y_train, y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
        
        # Probar cada valor de profundidad
        for j, profundidad in enumerate(profundidades):
            # Entrenar el árbol de decisión con el criterio y la profundidad actual
            arbol = DecisionTreeClassifier(max_depth=profundidad, criterion=criterio)
            arbol.fit(X_train, y_train)
            
            # Evaluar el modelo en el conjunto de entrenamiento
            score_train = arbol.score(X_train, y_train)
            resultados_train[criterio][i, j] = score_train
            
            # Evaluar el modelo en el conjunto de prueba
            score_test = arbol.score(X_test, y_test)
            resultados_test[criterio][i, j] = score_test

# Calcular la media de la exactitud para cada profundidad en train y test
media_resultados_train = {crit: resultados_train[crit].mean(axis=0) for crit in criterios}
media_resultados_test = {crit: resultados_test[crit].mean(axis=0) for crit in criterios}

# Crear un DataFrame con los resultados para mejor visualización
df_resultados = pd.DataFrame({
    'Profundidad': profundidades,
    'Exactitud Media (Train - Gini)': media_resultados_train['gini'],
    'Exactitud Media (Test - Gini)': media_resultados_test['gini'],
    'Exactitud Media (Train - Entropy)': media_resultados_train['entropy'],
    'Exactitud Media (Test - Entropy)': media_resultados_test['entropy'],
})

# Identificar la mejor configuración para ambos criterios
mejor_gini = df_resultados.loc[df_resultados['Exactitud Media (Test - Gini)'].idxmax()]
mejor_entropy = df_resultados.loc[df_resultados['Exactitud Media (Test - Entropy)'].idxmax()]

print("Mejor configuración para Gini:")
print(mejor_gini)
print("\nMejor configuración para Entropy:")
print(mejor_entropy)

# Mostrar los resultados detallados para cada profundidad
print("\nResultados detallados de la validación cruzada:")
print(df_resultados)

# Graficar la exactitud en función de la profundidad para ambos criterios
plt.figure(figsize=(12, 8))
plt.plot(profundidades, media_resultados_test['gini'], marker='o', label="Test - Gini", color='blue')
plt.plot(profundidades, media_resultados_test['entropy'], marker='o', label="Test - Entropy", color='orange')
plt.plot(profundidades, media_resultados_train['gini'], linestyle='--', label="Train - Gini", color='blue')
plt.plot(profundidades, media_resultados_train['entropy'], linestyle='--', label="Train - Entropy", color='orange')
plt.title("Exactitud Media en Train y Test vs. Profundidad del Árbol (Gini vs Entropy)")
plt.xlabel("Profundidad del Árbol")
plt.ylabel("Exactitud Media")
plt.legend()
plt.grid(True)
plt.show()
#%%

# Inciso d: Entrenar el mejor modelo en el conjunto de desarrollo completo

# Seleccionar la mejor configuración (profundidad y criterio) basándonos en la mejor exactitud en Test
# Para esto, seleccionamos la fila con la mejor exactitud media en test
mejor_configuracion = df_resultados.loc[df_resultados[['Exactitud Media (Test - Gini)', 'Exactitud Media (Test - Entropy)']].max(axis=1).idxmax()]

# Obtener la mejor profundidad y criterio
mejorProfundidad = int(mejor_configuracion['Profundidad'])
mejorCriterio = 'gini' if mejor_configuracion['Exactitud Media (Test - Gini)'] >= mejor_configuracion['Exactitud Media (Test - Entropy)'] else 'entropy'

# Entrenar el modelo con la mejor configuración en el conjunto de desarrollo completo
mejor_modelo = DecisionTreeClassifier(max_depth=mejorProfundidad, criterion=mejorCriterio)
mejor_modelo.fit(X_dev, y_dev)

# Evaluación en el conjunto held-out
y_pred_held_out = mejor_modelo.predict(X_held_out)
accuracy_held_out = accuracy_score(y_held_out, y_pred_held_out)

# Imprimir resultados
print(f"Exactitud en el conjunto held-out con la mejor profundidad ({mejorProfundidad}) y criterio ({mejorCriterio}): {accuracy_held_out:.4f}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_held_out, y_pred_held_out)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.title("Matriz de Confusión para Clasificación Multiclase (Conjunto Held-out)")
plt.show()

# Análisis de las clases más confundidas
# Inicializar una lista para almacenar los pares de clases más confundidas
clases_confundidas = []

# Recorrer la matriz de confusión para encontrar las celdas fuera de la diagonal
for i in range(10):  # Asumiendo que tienes 10 clases (0-9)
    for j in range(10):
        if i != j:  # Solo analizamos las celdas fuera de la diagonal
            if conf_matrix[i, j] > 0:  # Si hay confusión entre las clases
                clases_confundidas.append((i, j, conf_matrix[i, j]))

# Ordenar las clases más confundidas por la cantidad de confusión
clases_confundidas.sort(key=lambda x: x[2], reverse=True)

# Mostrar las clases más confundidas
print("Clases más confundidas:")
for conf in clases_confundidas:
    print(f"Clases {conf[0]} y {conf[1]} se confundieron {conf[2]} veces.")
