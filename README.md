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



https://github.com/user-attachments/assets/6ea09b3c-5034-44bd-be81-093d024f408c


# Informe Técnico: Optimización y Resolución de Problemas 

## Gestión de Memoria Gráfica (VRAM) y Rendimiento de Hardware

1. **Saturación de VRAM y acumulación de datos:** El modelo SAM 3 generaba una cantidad enorme de información interna por cada fotograma analizado. Mantener todos estos datos acumulados de forma permanente saturaba rápidamente la memoria de la tarjeta gráfica (VRAM). Esto provocaba ralentizaciones severas y el congelamiento total del sistema durante los videos largos.
   * **Solución:** Se implementó una extracción manual de datos fuera de la tarjeta gráfica utilizando el comando `.cpu().numpy()` para trasladar las matrices a la RAM del ordenador. Además, se aplicó una limpieza agresiva mediante `torch.cuda.empty_cache()` para borrar los cálculos intermedios fotograma por fotograma.

2. **Fugas de memoria y enlaces residuales (Ghost Links):** A pesar de las limpiezas básicas, el sistema dejaba procesos muertos y enlaces invisibles conectados a la GPU tras procesar cada bloque de imágenes. Esta fuga de memoria asfixiaba progresivamente y de forma indetectable la tarjeta gráfica, provocando el colapso del sistema en el Lote 12.
   * **Solución:** Se aplicó una desconexión matemática estricta a nivel de código mediante `.detach().clone().cpu()` para cortar cualquier enlace residual. Esto se complementó con una secuencia de destrucción total de variables (usando `del` y `gc.collect()`) al finalizar cada lote procesado.

3. **Acumulación de sesiones huérfanas en VRAM:** Cada vez que el modelo inicializaba el análisis de un nuevo lote mediante el comando `start_session`, reservaba un bloque de memoria de aproximadamente 570 MB. Al terminar el lote, esta sesión no se autodestruía, llenando la memoria silenciosamente hasta agotar el espacio disponible.
   * **Solución:** Se inyectó el comando inverso `reset_session` explícitamente al finalizar el bucle de cada lote. Esto destruyó el estado interno de la IA y devolvió la memoria bloqueada a la reserva principal de la tarjeta gráfica.

4. **Fragmentación de memoria inicial (CUDA Out of Memory):** La acción de borrar y crear variables constantemente provocaba que la memoria gráfica se fragmentara en huecos pequeños e inutilizables. Esta desorganización causaba errores críticos de "Memoria Insuficiente" (OOM) antes de siquiera empezar a analizar el video a profundidad.
   * **Solución:** Se configuró la variable de entorno especial `expandable_segments:True` en la inicialización del entorno. Esto instruyó al asignador de memoria de PyTorch a agrupar los datos en segmentos dinámicos, evitando los huecos y previniendo las caídas del sistema.

5. **Carga matemática excesiva en 32 bits:** Operar el modelo de inteligencia artificial en precisión matemática estándar de 32 bits requería demasiados recursos. Este nivel de detalle excedía por completo la capacidad física de la memoria de video disponible para procesar múltiples imágenes de manera fluida.
   * **Solución:** Se implementó la técnica de "Inferencia de Precisión Mixta" utilizando `torch.autocast(dtype=torch.float16)`. Esto obligó a la tarjeta gráfica a procesar las matrices matemáticas pesadas en 16 bits, reduciendo el consumo a la mitad sin que la IA perdiera su capacidad de segmentación.

6. **Estrangulamiento térmico (Thermal Throttling):** Tras 15 minutos continuos de trabajo al 100% de su capacidad, la tarjeta gráfica se sobrecalentaba peligrosamente. Para evitar daños físicos en el silicio, el hardware reducía automáticamente su velocidad de procesamiento en el servidor.
   * **Solución:** Se aceptó esta limitación física de hardware ineludible como parte del proceso. Se asumió un mayor tiempo de renderizado (aumentando de 14 a 22 minutos) a cambio de priorizar y garantizar la estabilidad final del análisis.

7. **Cuello de botella en la transferencia de datos (PCIe):** Transferir datos constantemente desde la tarjeta gráfica hacia la RAM del sistema exigía utilizar los carriles de la placa base. Este viaje físico de la información generaba micro-latencias que se acumulaban fotograma tras fotograma.
   * **Solución:** Se asumió este retardo acumulativo como un peaje de rendimiento estrictamente necesario para el proyecto. La única alternativa era mantener los datos en la GPU, lo cual desembocaba irremediablemente en un colapso total de la memoria.

---

## Optimización de RAM y Procesamiento de Video

8. **Colapso de RAM por carga en alta calidad:** Intentar decodificar y cargar el archivo de video completo en alta resolución (1080p) desde el inicio era demasiado pesado para el sistema. Esta acción saturaba los 12.6 GB de RAM del entorno de forma instantánea, provocando cierres abruptos y fallos del servidor.
   * **Solución:** Se implementó una estrategia de "Carga Diferida" (Lazy Loading) mediante FFmpeg para extraer fotogramas individuales previamente. De esta manera, el entorno de Python únicamente cargaba en la memoria principal una sola matriz de imagen a la vez.

9. **Saturación de RAM por peso acumulado:** A pesar de la carga diferida, el peso acumulado del historial de las imágenes seguía ahogando la RAM poco a poco. En secuencias de video largas, la memoria del sistema terminaba colapsando irremediablemente por el exceso de metadatos retenidos.
   * **Solución:** Se implementó el sistema de "Procesamiento por Lotes" (Video Chunking) para dividir las cargas de trabajo. El video se dividió en carpetas aisladas de 100 a 500 imágenes que se procesaban de forma independiente, purgando la RAM al terminar cada bloque.

10. **Retención de memoria inactiva por el sistema operativo:** El sistema operativo Linux (utilizado en Google Colab) mantenía en su caché profunda la memoria que Python ya había liberado. Esta desincronización provocaba falsos errores de memoria llena que detenían la ejecución del análisis.
    * **Solución:** Se utilizó el comando de bajo nivel `libc.malloc_trim(0)` a través de la librería `ctypes` de Python. Esto saltó las restricciones del lenguaje de programación y forzó directamente al núcleo de Linux a devolver la memoria inactiva a la reserva central.

11. **Recarga ineficiente del modelo de IA:** Inicializar el pesado modelo de la red neuronal dentro del ciclo de procesamiento por lotes creaba un cuello de botella masivo. Obligaba al disco duro a leer gigabytes de datos repetidamente por cada subcarpeta procesada, ralentizando todo el flujo de trabajo.
    * **Solución:** Se movió la inicialización de la IA a la cabecera del script principal. Esto permitió cargar el modelo pesado en la memoria RAM una sola vez, dejándolo suspendido e inactivo para ser reutilizado instantáneamente por todos los lotes posteriores.

12. **Mezcla de archivos residuales:** El entorno virtual interactivo a veces conservaba imágenes de ejecuciones de pruebas anteriores. Esto causaba que los nuevos fotogramas se mezclaran con resoluciones distintas de otras pruebas, generando conflictos y datos corruptos para la IA.
    * **Solución:** Se inyectó un comando destructivo de terminal de Linux (`!rm -rf {ruta_directorio}/*`) en el bloque inicial. Al ejecutarlo antes de la extracción de FFmpeg, se garantizó un lienzo de carpetas completamente limpio y sin archivos residuales.

---

## Lógica de Rastreo e Inteligencia Artificial

13. **Inexactitud de rastreadores tradicionales:** En un intento por ahorrar memoria gráfica, se utilizaron algoritmos clásicos de seguimiento (como CSRT y MIL) para predecir el movimiento. Estos rastreadores fracasaron totalmente al confundir a los robots o perder el balón cuando se cruzaban rápidamente en la pantalla.
    * **Solución:** Se descartaron por completo las opciones tradicionales para priorizar la exactitud sobre la velocidad. Se le otorgó el control total a la IA (SAM 3) para evaluar cada fotograma, basándose en su comprensión semántica para no perder el rastro durante las oclusiones.

14. **Conflicto de identificadores (ID Collisions):** Se intentó que la IA buscara simultáneamente a los robots, el balón y la portería para ahorrar tiempo inyectando todos los comandos de texto de golpe. Esto sobreescribía las variables internas del modelo, devolviendo resultados vacíos o videos corruptos sin máscaras.
    * **Solución:** Se creó un "Bucle Híbrido" para aislar completamente los flujos de datos. El script escaneaba primero a los robots, forzaba una amnesia en el modelo con `reset_session`, y luego volvía a analizar la imagen para el balón sin mezclar las peticiones matemáticas.

15. **Pérdida de precisión en cajas delimitadoras:** Forzar a la IA a dividir su mecanismo de atención para buscar múltiples objetos diferentes a la vez (Single Pass) degradaba la nitidez de sus mapas visuales. Esto producía cajas delimitadoras imprecisas que capturaban partes incorrectas de la imagen y arruinaban las colisiones.
    * **Solución:** Se determinó mantener pasadas de rastreo completamente independientes para cada objeto detectado. Aunque penalizaba el tiempo de procesamiento, esto garantizaba que la IA se enfocara al 100% en un solo tipo de geometría a la vez.

**16. Sobrecarga por la tasa de fotogramas (FPS) y recursos limitados**
Analizar el video a una alta tasa de fotogramas exigía un nivel de procesamiento que superaba por completo los recursos que nuestro hardware posee. Al intentar procesar tantas imágenes por segundo, el sistema se saturaba rápidamente debido a nuestra capacidad limitada y no lograba terminar el análisis de forma estable.
* **Solución:** Optamos por reducir deliberadamente la extracción del video a solo 10 cuadros por segundo (FPS) utilizando FFmpeg para ajustarnos a la capacidad real de nuestro hardware. Aunque esto generó un movimiento menos fluido, redujo drásticamente la carga matemática y nos permitió finalizar el rastreo completo sin colapsar.

17. **Caída del script por formato de datos (AttributeError):** De manera impredecible, las salidas de la API alternaban entre devolver datos matemáticos en formato de GPU (Tensores) y formato crudo de CPU (NumPy). Cuando el código asumía el formato incorrecto, el programa colapsaba violentamente al aplicar funciones incompatibles.
    * **Solución:** Se añadió una validación condicional dinámica en el código utilizando `if torch.is_tensor()`. Esto aseguró que las conversiones matemáticas de formato solo se aplicaran si los datos estaban realmente alojados en la tarjeta gráfica.

18. **Riesgo de descalificación por reglas de competencia:** La integración de sistemas avanzados de rastreo externo (como DeepSORT o ByteTrack) para agilizar el proceso de video suponía un gran riesgo. El reglamento de la "Categoría Amateur" prohibía estrictamente el uso de redes neuronales secundarias para asistir el análisis, arriesgando la descalificación.
    * **Solución:** Se purgaron por completo todas las redes secundarias y librerías de seguimiento externas de terceros. El proyecto se limitó a utilizar exclusivamente la inferencia nativa de SAM 3 combinada con matemáticas de álgebra básica permitidas en la categoría.

---

## Preprocesamiento de Imagen y Estética

19. **Análisis de fondos irrelevantes:** El modelo de inteligencia artificial desperdiciaba valioso tiempo de procesamiento y gigabytes de memoria gráfica analizando zonas visuales inútiles. Esto incluía mapear detalladamente las texturas de la tribuna, el público y las paredes negras fuera del área real de juego.
    * **Solución:** Se automatizó un paso de preprocesamiento usando las herramientas algorítmicas `cv2.inRange` y `cv2.boundingRect` de OpenCV. Esto permitió detectar el verde del campo y recortar físicamente el video para aislar únicamente la cancha antes de pasárselo a la IA.

20. **Fallos en detección del terreno por iluminación:** Las sombras duras o luces sobreexpuestas en diferentes zonas de la cancha hacían que el algoritmo de recorte fallara constantemente. La búsqueda de colores tradicional en formato RGB no lograba identificar correctamente la zona de juego bajo cambios bruscos de iluminación.
    * **Solución:** Se migró el análisis de imagen inicial al espectro de color avanzado HSV, el cual aísla el pigmento base de la luminosidad. Al calibrar rangos estrictos, el algoritmo pudo identificar el color verde de la cancha sin importar la intensidad de las sombras.

21. **Recortes perjudiciales en extremidades:** El algoritmo automático generaba un recorte geométrico tan exacto al borde del campo de juego que terminaba amputando visualmente partes del cuerpo de los robots. Si un robot caminaba por la línea de banda, la IA lo perdía de vista al no tener su silueta completa en el plano.
    * **Solución:** Se aplicó una ecuación matemática simple para restar y sumar 100 píxeles a las coordenadas iniciales y finales del recorte. Esto creó un "margen de seguridad" invisible que mantuvo a los robots completamente dentro del marco.

22. **Estética arruinada por filtros de oscurecimiento:** Un intento inicial para ocultar los fondos distractores fue aplicar un filtro tipo "Blackout" que pintaba de negro absoluto al público y a las paredes. Aunque esto aceleraba el proceso interno de la IA, destruía por completo el realismo y la estética del video requerida para la entrega final.
    * **Solución:** Se abandonó el oscurecimiento destructivo de píxeles en favor de la técnica de recorte físico por coordenadas espaciales. Esto logró reducir el procesamiento computacional mientras mantenía un aspecto natural y profesional del entorno de fútbol.

---

## Matemáticas de Colisión y Renderizado Final

23. **Asignación aleatoria de IDs:** El sistema de la IA asignaba identificadores numéricos de forma aleatoria en cada ejecución de prueba (por ejemplo, numerando a los robots del equipo rojo como 17, 18 y 19). Esta irregularidad numérica rompía constantemente la lógica del código encargada de asignar colores fijos y rastrear los puntos de cada bando.
    * **Solución:** Se normalizaron los identificadores en tiempo real mediante una fórmula matemática sencilla, restando su valor mínimo de rastreo (`tracker_id - np.min(tracker_id)`). Esto garantizó que la numeración siempre se reiniciara a 0, 1 y 2 de forma consistente.

24. **Falta de registro en colisiones de gol:** A nivel matemático, la máscara 2D generada para el balón a veces resultaba demasiado pequeña o ajustada al contorno. Esto causaba que no se cruzara con los suficientes píxeles del área de la portería, impidiendo que el motor de físicas registrara un gol legítimo.
    * **Solución:** Se aplicó una operación morfológica de visión computacional expansiva llamada "Dilatación" (`cv2.dilate`). Esto engordó artificialmente la máscara del balón con píxeles extra, haciendo que el cálculo de choque fuera hiper-sensible y detectara el gol certeramente.

25. **Generación de telemetría sin coordenadas exactas:** El modelo SAM 3 no está diseñado para devolver puntos centrales exactos (coordenadas X, Y), sino que devuelve matrices gigantes con siluetas amorfas de los objetos. Esto hacía imposible generar de forma automatizada la telemetría de movimiento o los mapas de calor requeridos.
    * **Solución:** Se programó una función geométrica a medida usando los métodos de array `np.mean` y `np.where`. Esto permitió calcular de manera matemática el Centro de Masa absoluto de cada silueta, convirtiéndola en un punto coordenado puntual perfecto para la telemetría.

26. **Inestabilidad por máscaras fragmentadas:** En los cruces corporales intensos durante el partido, cuando un robot tapaba parcialmente a otro, la IA devolvía siluetas fragmentadas o cortadas a la mitad. Esto provocaba que las cajas delimitadoras clásicas temblaran violentamente o cambiaran de tamaño erráticamente en el video final.
    * **Solución:** El cálculo matemático del "Centro de Masa" logró estabilizar de forma natural las referencias visuales. Al ser un promedio aritmético de todos los píxeles, el punto central del rastreo no saltaba repentantemente aunque la silueta del robot estuviera incompleta por la oclusión.

27. **Caídas del script por divisiones entre cero:** Cuando el balón o un objeto salía temporalmente de la cámara, el sistema reportaba correctamente una matriz vacía. Sin embargo, intentar calcular el centroide matemático de una matriz inexistente generaba un error fatal por división entre cero (`NaN`), cerrando abruptamente el programa en plena generación del video.
    * **Solución:** Se protegió todo el bloque de operaciones matemáticas con una compuerta condicional de validación estricta (`if len(indices_x) > 0`). Esto aseguró que la geometría de colisión solo se calculara si se comprobaba previamente la existencia de píxeles válidos en la pantalla.


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

1. Clonar el repositorio:
   ```bash git
   clone  https://colab.research.google.com/drive/1UrF0WutbZuiNGDLzT8S2aRW3u8OyqIDc?usp=sharing https://colab.research.google.com/drive/1euw_ejfaQBKeaXIYLDEllAd_hDxm09-k?usp=sharing
 
 cd futbotmx-sam3


# Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=flat&logo=github&logoColor=white)


# Proyecto_SAM3
#Primer copia del proyecto en GitHub 
#Aqui se va a crear la documentacion
https://colab.research.google.com/drive/1UrF0WutbZuiNGDLzT8S2aRW3u8OyqIDc?usp=sharing
https://colab.research.google.com/drive/1euw_ejfaQBKeaXIYLDEllAd_hDxm09-k?usp=sharing
