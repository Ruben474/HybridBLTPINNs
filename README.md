# HybridBLTPINNs

## Objetivo científico

**Solving the Viscous Burgers' Equation: A Hybrid Approach Combining Boundary Layer Theory and Physics-Informed Neural Networks.**

Autores identificados en el README original: Oscar Martínez Núñez, Ana Magnolia Marín Ramírez y Rubén Darío Ortiz Ortiz, Universidad de Cartagena, Grupo Ondas, Instituto de Matemáticas Aplicadas. Correspondencia: rortizo@unicartagena.edu.co.

El proyecto explora soluciones de Burgers viscosa mediante diferencias finitas, PINNs y perfiles analíticos denominados BLT (teoría de capa límite) en el cuaderno. En varias variantes, la hibridación consiste en usar el perfil BLT como condición inicial. Esto no equivale por sí solo a implementar un término adicional de regularización.

**Estado de la documentación, 22 de octubre de 2026:** fuente y salidas históricas inspeccionadas en [`3cbf75a97577be8a0c500ee59451e41e3e3bcec9`](https://github.com/Ruben474/HybridBLTPINNs/tree/3cbf75a97577be8a0c500ee59451e41e3e3bcec9), commit «Se creó con Colab» del 26 de octubre de 2024. El README original sí existe en esa versión y contiene título, autores y resumen en macros LaTeX. No se ha reejecutado el entrenamiento para esta documentación. Se distingue evidencia guardada de reproducibilidad pendiente.

## Ecuación y problemas tratados

El residuo implementado mediante diferenciación automática es:

$$u_t+u u_x-\varepsilon u_{xx}=0.$$

Las celdas PINN 34 y 35 usan $x\in[-1,1]$, $t\in[0,1]$, $\varepsilon=0.01$, $u(-1,t)=1$, $u(1,t)=-1$ y

$$u(x,0)=-\tanh\!\left(\frac{x}{2\varepsilon}\right).$$

Todos los índices de celda de esta guía cuentan desde cero e incluyen Markdown; no son los números de ejecución de Jupyter.

El perfil $u_*(x)=-\tanh(x/(2\varepsilon))$ satisface la EDP estacionaria, pero sus valores exactos en los extremos son $\pm\tanh(1/(2\varepsilon))$, no exactamente $\pm1$. Por tanto, es una referencia con discrepancia de frontera para ese problema finito, aunque la diferencia sea muy pequeña para ciertas viscosidades. No es la solución exacta de cualquier condición inicial o de frontera del cuaderno.

**Variantes verificadas:** se cambian viscosidades, perfiles iniciales (incluidos gaussianos y escalones), fronteras y métodos. En particular, la celda FD 5 usa $\varepsilon=0.1$ y fronteras variables $\sin(\pi t)$ y $\cos(2\pi t)$; no corresponde al mismo problema que las celdas 34–35. También hay perfiles con otra escala, como `-tanh(x/sqrt(epsilon))`, que no deben intercambiarse con el anterior.

## Estructura y datos

| Archivo versionado | Función |
| --- | --- |
| [solucionburgersredesneuronales.ipynb](solucionburgersredesneuronales.ipynb) | 66 celdas con experimentos FD, PINN, BLT, gráficos y salidas guardadas |
| `README.md` | Descripción científica e instrucciones |

No hay CSV de entrada, pesos ni dependencias fijadas como archivos independientes en la versión revisada. Los datos de entrenamiento se generan desde geometría, EDP y condiciones. Los CSV leídos por las comparaciones son salidas intermedias que deben generarse previamente, no un conjunto externo incluido.

| Salida generada | Productor y columnas relevantes |
| --- | --- |
| `pinns_solution_t_0.5.csv` | celda 34: `x`, `u_pinns(x,t=0.5)` |
| `hybrid_pinns_solution_t_0.5.csv` | celda 35: `x`, `u_hybridpinns(x,t=0.5)`; otras celdas reutilizan el nombre |
| `finite_differences_solution_t_0.5.csv` | varios bloques FD: `x`, `u(x,t=0.5)`; parámetros dependen del último productor ejecutado |
| `tanh_function_data.csv` | bloques de perfil analítico: `x`, `u(x)`; verificar su escala antes de comparar |

Los nombres reutilizados pueden sobrescribir experimentos con distinta viscosidad. Para archivar una ejecución, usar una carpeta por configuración y conservar una copia del cuaderno ejecutado.

## Ejecutar en Google Colab

[Abrir cuaderno](https://colab.research.google.com/github/Ruben474/HybridBLTPINNs/blob/main/solucionburgersredesneuronales.ipynb).

1. Guarda una copia y comienza con un entorno limpio. GPU es opcional para estos experimentos; registrar el dispositivo utilizado.
2. El cuaderno instala `deepxde` y `tensorflow` sin fijar versiones. Una preparación orientativa para las celdas seleccionadas es:

   ```python
   %pip install deepxde tensorflow numpy pandas matplotlib
   ```

   Antes de importar DeepXDE, puede seleccionarse explícitamente el backend TensorFlow compatible con la API histórica:

   ```python
   import os
   os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
   ```

   Esta selección es una propuesta de entorno, no una reproducción certificada del entorno original. Registrar las versiones resueltas; la compatibilidad de las versiones actuales queda por verificar.
3. Para una ruta acotada, ejecutar la celda **34** (PINN) o **35** (etiquetada híbrida) en sesiones limpias separadas. Ambas son bloques completos: red `[2,50,50,50,1]`, `tanh`, inicialización `Glorot uniform`, 400 puntos interiores, 40 de frontera, 40 iniciales, Adam con tasa `0.001` y 5000 épocas. Evalúan 200 puntos a $t=0.5$ y guardan los CSV indicados arriba.
4. Para el experimento histórico con norma de error impresa, ejecutar por separado la celda **12**: difusión `0.05`, muestreo `600/60/60`, misma arquitectura, Adam `0.0005` durante 5000 épocas y después L-BFGS. Su título gráfico dice `epsilon = 0.1`, pero el valor efectivo del código es `0.05`.
5. Descargar CSV, registros y figuras antes de cerrar Colab. No ejecutar todas las celdas como una única cadena: hay redefiniciones, comparaciones que preceden a productores y dependencias de variables de ejecuciones previas.

Las celdas 34 y 35 imponen matemáticamente el mismo perfil inicial, arquitectura, muestreo y optimizador. Sus etiquetas no establecen una diferencia metodológica controlada; diferencias numéricas entre ellas pueden deberse al entrenamiento aleatorio. No usar esa pareja por sí sola como prueba de ventaja de la hibridación.

## Métricas de error

El cuaderno utiliza dos cantidades diferentes bajo el nombre «L2». Con $e_i=u_i-u_{*,i}$:

| Ubicación | Definición | Interpretación |
| --- | --- | --- |
| Celdas 7 y 12 | $\sqrt{\Delta x\sum_i e_i^2}$ | aproximación discreta de norma espacial $L^2$ |
| Celdas 25, 26 y 32 | $\sqrt{N^{-1}\sum_i e_i^2}$ | RMSE, sin peso espacial |

No son intercambiables. La pérdida de entrenamiento (residuo y condiciones) tampoco es el error de solución. Antes de comparar CSV se deben comprobar igualdad de `x`, tiempo, viscosidad, condición inicial y fronteras. Para perfiles gaussianos, escalonados o fronteras variables se necesita una referencia que resuelva ese mismo problema.

## Resultados reproducibles: evidencia disponible

| Evidencia guardada | Valor | Estado |
| --- | --- | --- |
| Celda 7: FD, `nu=0.01`, 200 nodos, `dt=0.001`, `T=0.5`, referencia tanh | norma espacial discreta `0.008611637523541573` | salida histórica; no reejecutada aquí |
| Celda 12: variante BLT–PINN, `epsilon=0.05`, Adam + L-BFGS, evaluación `t=0.5` | norma espacial discreta `0.0009326370634177836` | salida histórica; no reejecutada aquí |

Estas filas corresponden a viscosidades diferentes: **no forman una comparación de métodos**. Las salidas embebidas no garantizan que correspondan a la última edición de cada celda.

Las celdas 25–26 miden intervalos vacíos y los describen como tiempos simulados; la 27 usa `sleep(0.1)`. Esas cifras no miden coste de FD ni entrenamiento y no deben citarse como evidencia de eficiencia. Los registros de entrenamiento contienen otros tiempos históricos, pero no un benchmark controlado común.

### Protocolo para completar una reproducción

1. Fijar commit, backend, versiones, semilla, dispositivo y una sola configuración del problema. Guardar `%pip freeze` y la configuración junto a los resultados; las rutas PINN anteriores no fijan una semilla.
2. Ejecutar cada método desde cero. Si se cambia una frontera para hacerla compatible con el perfil exacto, identificarlo como experimento nuevo.
3. Generar FD con la misma EDP y condiciones; comprobar estabilidad temporal y convergencia con refinamiento espacial y temporal. No reutilizar automáticamente el CSV de la celda 5.
4. Calcular ambas métricas con nombres explícitos sobre puntos alineados y publicar los CSV que permiten recalcularlas.
5. Medir con un cronómetro alrededor del cálculo real, separando compilación, entrenamiento e inferencia; repetir varias semillas y reportar dispersión.

### Pendiente por completar

- Entorno fijado y ejecución limpia comprobada; varias llamadas usan `dde.maps.FNN` y `epochs`, propias del estilo de API utilizado en el cuaderno.
- Separación de variantes, eliminación de sobrescrituras y nombres de salida que incluyan viscosidad y condiciones.
- Definición operativa inequívoca del método híbrido frente a la PINN de comparación.
- Tabla comparativa del mismo problema, con referencia validada, tiempos reales y múltiples semillas.
- Archivo de resultados y pesos, vínculo bibliográfico definitivo y licencia explícita. El objetivo del README original no se presenta como una conclusión científica ya demostrada.
