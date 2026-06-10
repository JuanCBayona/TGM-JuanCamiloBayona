# EsneyShift++ con UNI

Fue implementada la versión **EsneyShift++** utilizando el modelo fundacional **UNI** para extracción de características histopatológicas.

El modelo **H0-mini** requiere autorización explícita de sus autores en HuggingFace para poder ser descargado. Al momento de realizar estos experimentos, dicho acceso aún no había sido concedido, por lo que todas las evaluaciones fueron realizadas utilizando UNI.

## Experimento base: conjunto de entrenamiento vs sí mismo

Como validación inicial del pipeline, se comparó el conjunto de entrenamiento (Rings + CoCaHis) contra sí mismo.

### Resultados

| Métrica | Valor |
|----------|----------:|
| KL | 0.000000 |
| JS | 0.000000 |
| EMD | 0.000000 |
| MMD | 0.000000 |
| Fréchet | 0.000000 |
| KS | 0.001950 |
| NLL | 116.338217 |

Estos resultados confirman que la implementación se comporta como se espera.

Las métricas KL, JS, EMD, MMD y Fréchet alcanzan valores prácticamente nulos, indicando que ambas distribuciones son indistinguibles.

La distancia KS presenta un valor residual muy pequeño debido a efectos numéricos y al proceso de reducción de dimensionalidad empleado antes del cálculo.

La métrica Negative Log-Likelihood (NLL) merece una interpretación especial. En este caso, el valor:

```text
NLL = 116.34
```

debe considerarse como el valor de referencia (*baseline*) del sistema.

Un valor de NLL cercano a este indica que las muestras evaluadas son altamente compatibles con la distribución del conjunto de entrenamiento. Valores superiores indican que las muestras evaluadas son progresivamente menos probables bajo el modelo estadístico ajustado sobre los datos reales.

Por tanto, el objetivo no es obtener un NLL cercano a cero, sino un NLL lo más cercano posible al valor de referencia obtenido al comparar el conjunto consigo mismo.

---

## Análisis visual

### PCA

Para PCA, los puntos naranjas se ubican exactamente sobre los puntos azules. Esto indica que ambos conjuntos son idénticos, como era de esperarse al comparar un conjunto de datos contra sí mismo.

<img width="790" height="587" alt="imagen" src="https://github.com/user-attachments/assets/71e0ce20-3ae3-4655-98f3-66298f9c3e88" />

### UMAP

En UMAP las distribuciones también aparecen superpuestas. Debido a la naturaleza no lineal del algoritmo, algunos puntos pueden no coincidir exactamente, pero ambas distribuciones ocupan esencialmente la misma región del espacio de características.

<img width="801" height="621" alt="imagen" src="https://github.com/user-attachments/assets/33ae43c1-69e0-419f-8c4b-a7be55befc52" />

---

# Experimento FractalGen

Se entrenó FractalGen durante 685 épocas utilizando el conjunto Rings + CoCaHis.

## Evaluación cualitativa

Visualmente, las muestras generadas son ruidosas y se evidencia su diferencia respecto a las de entrenamiento.

<img width="585" height="341" alt="imagen" src="https://github.com/user-attachments/assets/7e0ce074-3f01-4fe3-a49e-04d937c191e0" />

---

## Evaluación cuantitativa

| Métrica | Valor |
|----------|----------:|
| KL | 4.079819 |
| JS | 0.138103 |
| EMD | 0.013040 |
| MMD | 0.003829 |
| Fréchet | 1828.888922 |
| KS | 0.210363 |
| NLL | 255.829935 |

---

## PCA

La separación entre los conjuntos es evidente. Las muestras generadas forman un grupo compacto localizado principalmente a la izquierda del espacio proyectado, mientras que las muestras reales ocupan una región claramente diferenciada.

<img width="789" height="589" alt="imagen" src="https://github.com/user-attachments/assets/68b829de-eee4-4329-ae7f-a3343a92574c" />

---

## UMAP

UMAP confirma la misma tendencia observada mediante PCA. Las muestras generadas forman un clúster compacto y separado del conjunto de entrenamiento.

<img width="794" height="585" alt="imagen" src="https://github.com/user-attachments/assets/a603285e-5ccc-491c-a5e6-6e386334257f" />

---

# Interpretación de las métricas

Las métricas obtenidas indican que las muestras generadas por FractalGen no siguen la misma distribución que las muestras reales del conjunto de entrenamiento.

### KL Divergence

La divergencia KL pasa de aproximadamente cero a:

```text
KL = 4.08
```

lo que indica diferencias importantes entre las distribuciones de características extraídas por UNI.

### Jensen-Shannon Divergence

La divergencia JS aumenta hasta:

```text
JS = 0.138
```

confirmando que ambas distribuciones contienen información diferente.

### Earth Mover's Distance (EMD)

La distancia EMD aumenta desde cero hasta:

```text
EMD = 0.013
```

indicando desplazamientos sistemáticos entre ambas distribuciones.

### Maximum Mean Discrepancy (MMD)

Aunque el valor:

```text
MMD = 0.0038
```

permanece relativamente bajo, sigue siendo claramente superior al valor de referencia.

Esto sugiere que las muestras generadas conservan ciertas características globales del dominio histopatológico, pero no reproducen completamente la distribución real.

### Fréchet Distance

La distancia de Fréchet aumenta desde aproximadamente cero hasta:

```text
Fréchet = 1828.89
```

lo que evidencia diferencias significativas entre las medias y covarianzas de ambas distribuciones en el espacio de características de UNI.

### Kolmogorov-Smirnov (KS)

El incremento desde:

```text
KS = 0.00195
```

hasta:

```text
KS = 0.21036
```

indica que las distribuciones marginales de las características presentan discrepancias importantes.

### Negative Log-Likelihood (NLL)

La métrica NLL proporciona una de las evidencias más claras de la diferencia entre ambos conjuntos.

| Comparación | NLL |
|-------------|----------:|
| Train vs Train | 116.34 |
| Train vs FractalGen | 255.83 |

Recordando que valores menores indican mayor compatibilidad con la distribución real, observamos que las muestras generadas presentan un NLL más de dos veces superior al valor de referencia.

En otras palabras, las características extraídas de las imágenes generadas son considerablemente menos probables bajo el modelo probabilístico ajustado utilizando las muestras reales.

La diferencia:

```text
255.83 - 116.34 = 139.49
```

es suficientemente grande como para concluir que las imágenes generadas se encuentran fuera de la región de alta densidad ocupada por los datos reales en el espacio de representación de UNI.

---

# Conclusión

Tanto las métricas cuantitativas como las proyecciones PCA y UMAP indican que FractalGen logra generar imágenes con apariencia histológica plausible, pero no reproduce completamente la distribución de características presente en el conjunto de entrenamiento.

La formación de un clúster compacto y claramente separado en PCA y UMAP sugiere además una cobertura limitada del espacio de variación histopatológica real, indicando una diversidad insuficiente de las muestras generadas respecto a los datos originales.

De manera consistente, todas las métricas de distribución aumentan significativamente respecto al escenario de referencia (*train vs train*), siendo especialmente notable el incremento observado en la métrica NLL, que pasa de **116.34** a **255.83**, indicando una pérdida considerable de compatibilidad estadística con la distribución de entrenamiento.



