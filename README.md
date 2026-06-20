Copa FutBotMX 2026 - Reto de Visión por Computadora 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Descripción del Proyecto
	
Este repositorio contiene nuestra entrega oficial para la Copa FutBotMX. Nuestro proyecto utiliza el Segment Anything Model 3 (SAM 3) de Meta para segmentar, rastrear y analizar de forma automatizada los partidos de fútbol robótico.
	
El objetivo principal es aplicar conceptos de visión por computadora para identificar el balón y los robots, generando estadísticas útiles y una narrativa visual del juego.

# Enfoque y Arquitectura
	
Nuestra solución está diseñada mediante el siguiente flujo de trabajo:

1. Detección y Segmentación (SAM 3): Utilizamos comandos y prompts) para indicarle al modelo cómo identificar de manera precisa a los robots y el balón en el campo de juego.
	
2. Seguimiento Visual (OpenCV): Implementamos OpenCV para dibujar cajas delimitadoras alrededor de los elementos detectados, permitiendo visualizar su trayectoria a lo largo de los fotogramas del video.
	
3. Análisis de Datos y Estadísticas: Desarrollamos una lógica de seguimiento que no solo observa el movimiento, sino que también clasifica a los robots por equipo y contabiliza métricas clave del partido, tales como:
	
   •	Número total de pases.

   •	Conteo de goles.
	
   •	Separación de estadísticas por equipo.

### Video de Prueba del Sistema

Para visualizar el funcionamiento del arbitraje automatizado en tiempo real, hemos integrado el siguiente clip de demostración:


<img width="400" height="450" alt="Ejemplo 1" src="https://github.com/user-attachments/assets/cbdd85cc-e5de-4ddd-8106-c8893315299a" /> <img width="400" height="450" alt="Ejemplo 2" src="https://github.com/user-attachments/assets/8799fd2c-e0a6-45cb-83c9-2c2b67098e37" />



---


##  Demostración del Sistema y Arbitraje

Conseguir este nivel de rastreo fluido requirió superar múltiples limitantes de hardware y software. Para renderizar este video, nuestro script segmenta el video original en lotes, limpiando la memoria VRAM de la GPU fotograma por fotograma. Utilizamos la inferencia nativa de **SAM 3** combinada con dilatación morfológica de máscaras en **OpenCV** para garantizar que los choques entre el balón y los robots sean hiper-sensibles, logrando un arbitraje automatizado preciso, libre de falsos positivos y con un rendimiento estable



https://github.com/user-attachments/assets/0025f9e9-4d9f-4d71-8366-d9a50e1b1724




##  Generación de Mapas de Calor (Heatmaps) y Telemetría
Al extraer matemáticamente el "Centro de Masa" de las siluetas amorfas generadas por la IA, el sistema logra exportar coordenadas X/Y precisas. Esto permite generar mapas de calor dinámicos para analizar el posicionamiento y el flujo del partido en tiempo real:




https://github.com/user-attachments/assets/ac0f7df8-2393-4226-8c3c-c43423b1ae10


## Radar de Posicionamiento y Telemetría
Además de los mapas de calor, implementamos un **Radar de telemetría en tiempo real**. Este componente proyecta la posición relativa de los robots y el balón en un plano 2D simplificado, eliminando el ruido visual de la cancha real. Esto nos permite observar la distribución espacial de los equipos y la dirección de juego con una precisión técnica superior, facilitando el análisis táctico automatizado.



https://github.com/user-attachments/assets/5adf93ff-818e-44f9-8df7-4081ee024a8c

## Demostración Completa del Sistema (2 Minutos)

<div align="center">
  <video src="https://github.com/Gerardo-MauricioGP/FutBotMX-Amateur-SAM3/blob/main/assets/Demostraci%C3%B3n%20Completa%20del%20Sistema.mp4" width="600" controls></video>
</div>

## Demostración en Instagram

Para ver el sistema en acción con un formato más dinámico y rápido, puedes revisar nuestro clip donde mostramos el rendimiento de la IA y el seguimiento de los robots en tiempo real:

<div align="center">
  <a href="https://www.instagram.com/reel/DZxuDYQgvVF/?igsh=MTUzeW53YjgyOGY2aA==" target="_blank">
    <img src="https://img.shields.io/badge/Video_de_Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Botón de Instagram" />
  </a>
</div>

# Informe Técnico: Optimización y Resolución de Problemas 

### Gestión de Memoria Gráfica (VRAM) y Rendimiento de Hardware

**1. Saturación de VRAM y acumulación de datos:** Nos enfrentamos a que el modelo SAM 3 generaba una cantidad enorme de información interna por cada fotograma analizado. Mantener todos estos datos acumulados de forma permanente saturaba rápidamente la memoria de nuestra tarjeta gráfica (VRAM), lo que nos provocaba ralentizaciones severas y el congelamiento total del sistema durante los videos largos.
   * **Solución:** Implementamos una extracción manual de datos fuera de la tarjeta gráfica utilizando el comando `.cpu().numpy()` para trasladar las matrices a la RAM del ordenador. Además, aplicamos una limpieza agresiva mediante `torch.cuda.empty_cache()` para borrar los cálculos intermedios fotograma por fotograma.

**2. Fugas de memoria y enlaces residuales (Ghost Links):** A pesar de nuestras limpiezas básicas, notamos que el sistema dejaba procesos muertos y enlaces invisibles conectados a la GPU tras procesar cada bloque de imágenes. Esta fuga de memoria asfixiaba progresivamente nuestra tarjeta gráfica, provocando el colapso del sistema al llegar al Lote 12.
   * **Solución:** Aplicamos una desconexión matemática estricta a nivel de código mediante `.detach().clone().cpu()` para cortar cualquier enlace residual. Esto lo complementamos con una secuencia de destrucción total de variables (usando `del` y `gc.collect()`) al finalizar cada lote procesado.

**3. Acumulación de sesiones huérfanas en VRAM:** Descubrimos que cada vez que inicializábamos el análisis de un nuevo lote mediante el comando `start_session`, el modelo reservaba un bloque de memoria de aproximadamente 570 MB. Al terminar el lote, esta sesión no se autodestruía, llenando nuestra memoria silenciosamente.
   * **Solución:** Inyectamos el comando inverso `reset_session` explícitamente al finalizar el bucle de cada lote. Con esto destruimos el estado interno de la IA y logramos devolver la memoria bloqueada a la reserva principal de la tarjeta gráfica.

**4. Fragmentación de memoria inicial (CUDA Out of Memory):** Observamos que la acción de borrar y crear variables constantemente provocaba que nuestra memoria gráfica se fragmentara en huecos pequeños e inutilizables, causándonos errores críticos de "Memoria Insuficiente" (OOM) antes de siquiera empezar a analizar el video.
   * **Solución:** Configuramos la variable de entorno especial `expandable_segments:True` en la inicialización de nuestro entorno. Esto instruyó al asignador de memoria de PyTorch a agrupar los datos en segmentos dinámicos, evitando los huecos y previniendo las caídas.

**5. Carga matemática excesiva en 32 bits:** Operar el modelo de inteligencia artificial en precisión matemática estándar de 32 bits nos exigía demasiados recursos, excediendo por completo la capacidad física de nuestra memoria de video.
   * **Solución:** Implementamos la técnica de "Inferencia de Precisión Mixta" utilizando `torch.autocast(dtype=torch.float16)`. Esto obligó a la tarjeta gráfica a procesar las matrices matemáticas pesadas en 16 bits, reduciendo nuestro consumo a la mitad sin perder capacidad de segmentación.

**6. Estrangulamiento térmico (Thermal Throttling):** Tras 15 minutos continuos de trabajo al 100% de su capacidad, nuestra tarjeta gráfica se sobrecalentaba peligrosamente. Para evitar daños físicos, el hardware reducía automáticamente su velocidad de procesamiento.
   * **Solución:** Aceptamos esta limitación física de hardware como parte del proceso. Decidimos asumir un mayor tiempo de renderizado (aumentando de 14 a 22 minutos) a cambio de priorizar y garantizar la estabilidad final de nuestro análisis.

**7. Cuello de botella en la transferencia de datos (PCIe):** Transferir datos constantemente desde la tarjeta gráfica hacia la RAM del sistema nos exigía utilizar los carriles de la placa base, generando micro-latencias que se acumulaban fotograma tras fotograma.
   * **Solución:** Asumimos este retardo acumulativo como un peaje de rendimiento estrictamente necesario para nuestro proyecto. Comprendimos que la única alternativa era mantener los datos en la GPU, lo cual nos llevaba irremediablemente a un colapso total de la memoria.

---

### Optimización de RAM y Procesamiento de Video

**8. Colapso de RAM por carga en alta calidad:** Intentar decodificar y cargar el archivo de video completo en alta resolución (1080p) desde el inicio resultó ser demasiado pesado. Esta acción saturaba nuestros 12.6 GB de RAM de forma instantánea, provocando cierres abruptos en el servidor.
   * **Solución:** Diseñamos una estrategia de "Carga Diferida" (Lazy Loading) mediante FFmpeg para extraer fotogramas individuales previamente. De esta manera, hicimos que nuestro entorno de Python únicamente cargara en la memoria principal una sola matriz de imagen a la vez.

**9. Saturación de RAM por peso acumulado:** A pesar de la carga diferida, el peso acumulado del historial de las imágenes seguía ahogando nuestra RAM poco a poco. En secuencias largas, nuestro sistema terminaba colapsando por el exceso de metadatos retenidos.
   * **Solución:** Implementamos un sistema de "Procesamiento por Lotes" (Video Chunking) para dividir nuestras cargas de trabajo. Dividimos el video en carpetas aisladas de 100 a 500 imágenes que procesamos de forma independiente, purgando la RAM al terminar cada bloque.

**10. Retención de memoria inactiva por el sistema operativo:** Nos dimos cuenta de que el sistema operativo Linux (utilizado en Google Colab) mantenía en su caché profunda la memoria que nuestro código en Python ya había liberado, provocándonos falsos errores de memoria llena.
    
* **Solución:** Utilizamos el comando de bajo nivel `libc.malloc_trim(0)` a través de la librería `ctypes`. Esto nos permitió saltar las restricciones de Python y forzar directamente al núcleo de Linux a devolvernos la memoria inactiva a la reserva central.

**11. Recarga ineficiente del modelo de IA:** Inicializar nuestro pesado modelo neuronal dentro del ciclo de procesamiento por lotes nos creaba un cuello de botella masivo, ya que obligaba al disco duro a leer gigabytes de datos repetidamente.
    
* **Solución:** Movimos la inicialización de la IA a la cabecera de nuestro script principal. Esto nos permitió cargar el modelo en la memoria RAM una sola vez, dejándolo suspendido e inactivo para reutilizarlo instantáneamente en todos los lotes posteriores.

**12. Mezcla de archivos residuales:** Nuestro entorno virtual a veces conservaba imágenes de pruebas anteriores. Esto causaba que los nuevos fotogramas se mezclaran con resoluciones distintas, generando conflictos y datos corruptos para la IA.

* **Solución:** Inyectamos un comando destructivo de terminal (`!rm -rf {ruta_directorio}/*`) en el bloque inicial. Al ejecutarlo antes de la extracción de FFmpeg, garantizamos un lienzo de carpetas completamente limpio.

---

### Lógica de Rastreo e Inteligencia Artificial

**13. Inexactitud de rastreadores tradicionales:** En un intento por ahorrar memoria gráfica, probamos utilizar algoritmos clásicos de seguimiento (como CSRT y MIL). Sin embargo, estos rastreadores nos fracasaron totalmente al confundir a los robots o perder el balón cuando se cruzaban rápido.

 * **Solución:** Descartamos por completo las opciones tradicionales para priorizar la exactitud sobre la velocidad. Le otorgamos el control total a la IA (SAM 3) para evaluar cada fotograma basándose en su comprensión semántica, evitando perder el rastro durante las oclusiones.

**14. Conflicto de identificadores (ID Collisions):** Intentamos que la IA buscara simultáneamente a los robots, el balón y la portería para ahorrar tiempo inyectando todos los comandos de golpe. Esto sobreescribía las variables internas del modelo, devolviéndonos videos corruptos o sin resultados.

* **Solución:** Creamos un "Bucle Híbrido" para aislar completamente nuestros flujos de datos. Programamos el script para escanear primero a los robots, forzar una amnesia en el modelo con `reset_session`, y luego volver a analizar la imagen para el balón sin mezclar las peticiones.

**15. Pérdida de precisión en cajas delimitadoras:** Al forzar a la IA a dividir su atención para buscar múltiples objetos a la vez (Single Pass), notamos que la nitidez de sus mapas visuales se degradaba, produciendo hitboxes imprecisas.

* **Solución:** Determinamos mantener pasadas de rastreo completamente independientes para cada objeto. Aunque esto nos penalizaba en tiempo, nos garantizó que la IA se enfocara al 100% en un solo tipo de geometría a la vez.

**16. Sobrecarga por la tasa de fotogramas (FPS) y recursos limitados:** Analizar el video a su tasa original exigía un nivel de procesamiento que superaba por completo los recursos de nuestro hardware. Al intentar procesar tantas imágenes por segundo, el sistema se nos saturaba rápidamente y no lográbamos terminar el análisis.

 * **Solución:** Optamos por reducir deliberadamente la extracción del video a solo 10 cuadros por segundo (FPS) utilizando FFmpeg para ajustarnos a nuestra capacidad real de hardware. Aunque esto generó un movimiento menos fluido, redujo drásticamente nuestra carga matemática y nos permitió finalizar el rastreo.

**17. Caída del script por formato de datos (AttributeError):** De manera impredecible, las salidas de nuestra API alternaban entre devolvernos datos en formato de GPU (Tensores) y formato de CPU (NumPy), colapsando nuestro código al aplicar funciones incompatibles.

 * **Solución:** Añadimos una validación condicional dinámica en el código utilizando `if torch.is_tensor()`. Esto nos aseguró que las conversiones matemáticas de formato solo se aplicaran si los datos estaban realmente alojados en la tarjeta gráfica.

**18. Riesgo de descalificación por reglas de competencia:** Evaluamos integrar sistemas avanzados de rastreo externo (como DeepSORT), pero el reglamento de la "Categoría Amateur" nos prohibía estrictamente el uso de redes neuronales secundarias.

* **Solución:** Purgamos por completo todas las redes secundarias y librerías de seguimiento externas. Limitamos nuestro proyecto a utilizar exclusivamente la inferencia nativa de SAM 3 combinada con matemáticas de álgebra básica permitidas en la categoría.

---

### Preprocesamiento de Imagen y Estética

**19. Análisis de fondos irrelevantes:** Notamos que la inteligencia artificial desperdiciaba valioso tiempo de procesamiento mapeando detalladamente las texturas de la tribuna, el público y las paredes negras fuera de nuestra área de juego.

* **Solución:** Automatizamos un paso de preprocesamiento usando las herramientas algorítmicas `cv2.inRange` y `cv2.boundingRect` de OpenCV. Esto nos permitió detectar el verde del campo y recortar físicamente el video para aislar únicamente la cancha antes de pasárselo a la IA.

**20. Fallos en detección del terreno por iluminación:** Las sombras duras o luces sobreexpuestas hacían que nuestro algoritmo de recorte fallara constantemente, ya que la búsqueda de colores en formato RGB no lograba identificar la zona de juego bajo cambios bruscos de luz.

* **Solución:** Migramos nuestro análisis de imagen inicial al espectro de color avanzado HSV. Al calibrar rangos estrictos, logramos que el algoritmo identificara correctamente el color verde de la cancha sin importar la intensidad de las sombras.

**21. Recortes perjudiciales en extremidades:** Nuestro algoritmo automático generaba un recorte tan exacto al borde del campo que terminaba amputando visualmente partes de los robots si caminaban por la línea de banda.

 * **Solución:** Aplicamos una ecuación matemática simple para restar y sumar 100 píxeles a las coordenadas del recorte. Esto nos creó un "margen de seguridad" invisible que mantuvo a los robots completamente dentro del marco.

**22. Estética arruinada por filtros de oscurecimiento:** Nuestro intento inicial para ocultar los fondos fue aplicar un filtro "Blackout" que pintaba de negro al público. Aunque esto aceleraba la IA, destruía el realismo y la estética que queríamos presentar en el proyecto final.

* **Solución:** Abandonamos el oscurecimiento destructivo de píxeles y optamos por la técnica de recorte físico por coordenadas espaciales. Con esto logramos reducir el procesamiento mientras manteníamos un aspecto natural y profesional.

---

### Matemáticas de Colisión y Renderizado Final

**23. Asignación aleatoria de IDs:** El sistema nos asignaba identificadores numéricos de forma aleatoria (ej. 17, 18 y 19), lo cual nos rompía constantemente la lógica del código encargada de asignar nuestros colores fijos por equipo.

* **Solución:** Normalizamos los identificadores en tiempo real mediante una fórmula sencilla, restando su valor mínimo de rastreo (`tracker_id - np.min(tracker_id)`). Esto nos garantizó que la numeración siempre se reiniciara a 0, 1 y 2 de forma consistente.

**24. Falta de registro en colisiones de gol:** A nivel matemático, la máscara 2D que generábamos para el balón a veces resultaba demasiado pequeña, impidiendo que se cruzara con los píxeles de la portería y provocando que no registráramos goles legítimos.

* **Solución:** Aplicamos una operación morfológica de "Dilatación" (`cv2.dilate`). Esto nos permitió engordar artificialmente la máscara del balón con píxeles extra, haciendo que nuestro cálculo de choque fuera hipersensible y detectara el gol certeramente.

**25. Generación de telemetría sin coordenadas exactas:** El modelo SAM 3 nos devolvía matrices gigantes con siluetas amorfas en lugar de coordenadas (X, Y) exactas, haciéndonos imposible generar mapas de calor.

* **Solución:** Programamos una función geométrica a medida usando los métodos `np.mean` y `np.where`. Esto nos permitió calcular de manera matemática el Centro de Masa absoluto de cada silueta, dándonos el punto coordenado perfecto para nuestra telemetría.

**26. Inestabilidad por máscaras fragmentadas:** En los cruces corporales durante el partido, la IA nos devolvía siluetas fragmentadas, provocando que nuestras cajas delimitadoras temblaran violentamente en el video final.

* **Solución:** El cálculo matemático del "Centro de Masa" que diseñamos logró estabilizar nuestras referencias visuales. Al ser un promedio aritmético, nuestro punto de rastreo no saltaba repentinamente aunque la silueta del robot estuviera ocluida.

**27. Caídas del script por divisiones entre cero:** Cuando el balón salía de la cámara, el sistema reportaba una matriz vacía. Sin embargo, al intentar calcular nuestro centroide sobre algo inexistente, el programa nos arrojaba un error fatal por división entre cero (`NaN`).

* **Solución:** Protegimos todo nuestro bloque de operaciones matemáticas con una compuerta condicional de validación estricta (`if len(indices_x) > 0`). Esto aseguró que la geometría solo se calculara si nosotros comprobábamos previamente la existencia de píxeles válidos.
    
**28. Falsos positivos continuos en el conteo de toques de balón:** Al registrar las interacciones mediante intersección de máscaras, notamos que el sistema sumaba decenas de toques si un robot simplemente se quedaba parado junto al balón, inflando nuestro marcador.

* **Solución:** Implementamos una lógica de validación por "estado y movimiento". Primero, establecimos un umbral de desplazamiento matemático (`MOVEMENT_THRESHOLD`) para registrar un toque únicamente si el balón presentaba movimiento físico. Segundo, creamos un conjunto en memoria (`set`) para llevar el registro exacto de qué robot mantenía el contacto, evitando sumar puntos duplicados.

**29. Conteo múltiple de goles en una sola anotación:** Debido a que nuestro cálculo de solapamiento (IoU) se evalúa cuadro por cuadro, el sistema nos sumaba un gol nuevo en cada fotograma mientras el balón permaneciera dentro de la red.

* **Solución:** Implementamos un control de estado temporal utilizando la variable `ball_in_rectangle_prev_frame`. Esta memoria a corto plazo nos permitió comparar el estado del fotograma actual con el anterior, asegurando que contabilizáramos el "Gol" de forma única.

**30. Superposición y nula legibilidad de los marcadores en pantalla:** Al intentar dibujar nuestros textos de puntaje sobre el video, los colores cambiantes de la cancha camuflaban las letras y desalineaban los marcadores.

* **Solución:** Programamos un sistema de visualización dinámica (Scoreboard). Para la legibilidad, usamos la función `getTextSize` creando cajas de fondo negro opaco que se ajustan automáticamente al texto. Para el posicionamiento, calculamos las coordenadas algebraicamente en base a la resolución del video, asegurando que nuestro marcador global se mantuviera perfectamente centrado y sin chocar.
   

# Requisitos de Hardware y Software

Este proyecto fue desarrollado y optimizado para ejecutarse en la nube.
	
• Entorno: Google Colab
	
• Hardware: GPU NVIDIA T4
	
• Versión de Python: 3.10+

# Dependencias Principales:

  `torch` y `torchvision` (Requeridos para ejecutar el modelo SAM 3)

  `opencv-python` (Requeridos para ejecutar el modelo SAM 3)
 
  `numpy`

  `Sam3` (Repositorio oficial de Meta)

# Instrucciones de Instalación y Reproducción 

Para replicar nuestro entorno y obtener los resultados, sigue estos pasos:

1. Apertura directa
   
Puedes abrir y ejecutar nuestro proyecto directamente en la nube haciendo clic aquí:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gerardo-MauricioGP/FutBotMX-Amateur-SAM3/blob/main/FutBotMX_Final.ipynb)

**Ejecución Automática en la Nube:**
   * **Entorno:** Sube o abre el archivo `FutBotMX-Amateur-SAM3` en Google Colab.
   * **Configuración:** Ve a `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` y asegúrate de seleccionar **GPU (T4)**.
   * **Secuencia:** Ejecuta las celdas en orden secuencial.
   * **Validación de seguridad:** Durante la ejecución, el sistema buscará el token de acceso en la nube. Si no lo detecta automáticamente, desplegará un cuadro de texto pidiéndote que ingreses el token.
   * **Procesamiento:** Una vez validado, el script comenzará de inmediato el preprocesamiento, análisis e inferencia de los mapas y marcadores.


## Equipo de Desarrollo

Este proyecto fue desarrollado para la **Copa FutBotMX 2026** por el equipo:

* **Alan Eduardo López Escobedo**
* **Carlos Gabriel Caamal Magaña**
* **Rylan Transito Chan**
* **Gerardo Mauricio Galván Pablo**

# Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white)





