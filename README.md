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

# Para replicar nuestro entorno y obtener los resultados, sigue estos pasos:

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
