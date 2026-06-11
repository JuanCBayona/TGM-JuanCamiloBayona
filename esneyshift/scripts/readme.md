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

# Comparación entre ImageNet y CIFAR-10

Se realizó un segundo experimento utilizando los modelos **ResNet50** y **UNI** como extractores de características. Los conjuntos de datos comparados fueron **ImageNet** y **CIFAR-10**, utilizando 500 imágenes de cada uno.

## Resultados cuantitativos

| Métrica | ResNet50 | UNI | Mejor |
|----------|----------:|----------:|:------:|
| KL Divergence ↓ | **0.917576** | 1.324928 | ResNet50 |
| JS Divergence ↓ | 0.082245 | **0.058502** | UNI |
| EMD ↓ | 0.008040 | **0.005909** | UNI |
| MMD ↓ | 0.004537 | **0.004000** | UNI |
| Fréchet Distance ↓ | **119.936750** | 1113.064191 | ResNet50 |
| KS Statistic ↓ | 0.157360 | **0.121200** | UNI |
| Negative Log-Likelihood ↓ | 228.832231 | **143.585602** | UNI |
| Cosine Similarity ↑ | 0.381647 | **0.554521** | UNI |

En términos generales, ambas redes detectan diferencias significativas entre las distribuciones de ImageNet y CIFAR-10. UNI presenta mejores resultados en la mayoría de las métricas, mientras que ResNet50 obtiene una distancia de Fréchet considerablemente menor.

## Histogramas

Las distribuciones obtenidas mediante UNI presentan una apariencia más cercana a una distribución Gaussiana.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/a02fbe4d-8475-4102-9677-87bdab6d7f3c">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/ffcb500d-f936-4a6e-b9df-d3ee62be3796">

</td>
</tr>
</table>

## PCA

Las proyecciones PCA muestran una clara separación entre ambas distribuciones para los dos extractores de características.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/c4b535df-e85e-4fe7-90e0-2597a03a1609">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/60c4f0e6-8840-4b90-9d59-709f288657c8">

</td>
</tr>
</table>

## UMAP

UMAP también evidencia una separación consistente entre los conjuntos de datos.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/9ea94387-49e2-4a9b-8e01-7bbe4d204b57">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/fdaa89c7-4e94-47ad-89bf-6e469c91b707">

</td>
</tr>
</table>

---

# Comparación condicional

Se realizó un segundo experimento para evaluar comparaciones condicionadas. Se utilizaron:

- El conjunto de entrenamiento empleado en Fractal.
- El conjunto de validación normalizado mediante Reinhard.
- El conjunto de validación original utilizado como referencia para la normalización.

La intención fue comparar imágenes que mantienen la misma estructura histológica pero presentan diferencias de coloración.

## Resultados cuantitativos

| Métrica | ResNet50 | UNI | Mejor |
|----------|----------:|----------:|:------:|
| KL Divergence ↓ | **3.773083** | 4.212745 | ResNet50 |
| JS Divergence ↓ | **0.130876** | 0.135359 | ResNet50 |
| EMD ↓ | **0.005251** | 0.006464 | ResNet50 |
| MMD ↓ | 0.018870 | **0.018867** | UNI |
| Fréchet Distance ↓ | **35.161207** | 535.583863 | ResNet50 |
| KS Statistic ↓ | 0.186631 | **0.177764** | UNI |
| Negative Log-Likelihood ↓ | **76.553647** | 128.039033 | ResNet50 |
| Cosine Similarity ↑ | **0.932009** | 0.798396 | ResNet50 |
| MSE ↓ | 1230.114516 | 1230.114516 | Igual |
| PSNR ↑ | 18.090802 | 18.090802 | Igual |
| SSIM ↑ | 0.873172 | 0.873172 | Igual |

A diferencia del experimento anterior, en este caso las imágenes comparten la misma estructura histológica, por lo que se espera una mayor similitud entre distribuciones. Ambas redes reflejan esta situación mediante mayores valores de similitud coseno y menores distancias globales.

## Histogramas

Los histogramas generados por ambos modelos mantienen una forma aproximadamente Gaussiana.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/0f3d6487-e08a-4842-9a4c-17f565e20165">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/5e84fe02-94cd-4868-9cc5-84cdfc067f92">

</td>
</tr>
</table>

## PCA

En PCA las distribuciones aparecen considerablemente más cercanas que en el experimento ImageNet–CIFAR10, especialmente para UNI.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/6f21416b-b0ca-48a6-8b8a-bfa537dc6ff7">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/96f18a57-ecc8-4a45-a917-caac93b4aa79">

</td>
</tr>
</table>

## UMAP

UMAP continúa mostrando separación entre dominios, aunque menor que la observada entre ImageNet y CIFAR-10.

<table>
<tr>
<td width="50%" align="center">

**ResNet50**

<img src="https://github.com/user-attachments/assets/f07d34d3-469b-48d1-9151-335872ad7201">

</td>
<td width="50%" align="center">

**UNI**

<img src="https://github.com/user-attachments/assets/e243ee54-9666-460d-9161-3514771ce9f7">

</td>
</tr>
</table>

## Conclusiones

- Ambos extractores distinguen claramente ImageNet y CIFAR-10 mediante histogramas, PCA y UMAP.
- UNI obtiene mejores resultados en la mayoría de las métricas de divergencia durante la comparación no condicional.
- ResNet50 presenta distancias de Fréchet significativamente menores en ambos experimentos.
- En las comparaciones condicionadas, las métricas reflejan una mayor similitud entre dominios debido a la preservación de las estructuras histológicas.
- PCA evidencia una proximidad considerable entre distribuciones condicionadas, mientras que UMAP continúa detectando diferencias de dominio residuales.



NECESITO AUN AGREGAR DATASETS DESDE KAGGLE O DRIVE.

FALTA COMPARAR TAMBIEN DISTANCIA ENTRE 




