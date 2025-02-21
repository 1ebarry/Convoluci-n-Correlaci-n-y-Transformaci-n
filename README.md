<h1 align="center">Convolución, Correlación y Transformación </h1>
<p align="center"> </p>
<p align="center"><img src="https://cdn.prod.website-files.com/649475f4d573d5d9d1522680/649475f4d573d5d9d1522c35_analisis-de-fourier-y-wavelets-para-el-procesamiento-de-imagenes.jpg"/></p> 

# Introducción
Este proyecto contiene la implementación y análisis de operaciones fundamentales en el procesamiento de señales, incluyendo convolución, correlación y transformadas en el dominio de la frecuencia, calculamos la convolución de una señal con un sistema tanto manualmente como en Python, así como la correlación entre señales periódicas Además, se incluye el análisis de una señal descargada desde PhysioNet para el estudio de sus características en el dominio del tiempo y frecuencia mediante la Transformada de Fourier.


# Análisis Manual de la Convolución Discreta entre Señal y Sistema

En este análisis se estudió el efecto de la convolución discreta entre una señal de entrada x(n) respuesta al impulso de un sistema h(n)  representadas por los dígitos de un código y una cédula, para comprender mejor este concepto cada integrante del grupo realizó el ejercicio con su respectiva información personal.

## Integrante 1   
Señal Resultante de la convolución:
![WhatsApp Image 2025-02-20 at 10 55 51 PM](https://github.com/user-attachments/assets/151b44b3-ab53-4e49-bede-d4f5bfeb97af)    

Graficas X1 y X2:  
![image](https://github.com/user-attachments/assets/1c7086a8-b563-419b-b773-28ecb64363c3)  

## Integrante 2     
Señal Resultante de la convolución:
![WhatsApp Image 2025-02-20 at 11 16 45 PM](https://github.com/user-attachments/assets/3f18a868-310f-4da5-8b0f-968e86c58b83)   

Graficas X1 y X2:  
![WhatsApp Image 2025-02-20 at 11 16 23 PM](https://github.com/user-attachments/assets/b9a8cc44-8e73-4533-87c5-5b38853f4372)  

## Integrante 3   
Señal Resultante de la convolución:  
![WhatsApp Image 2025-02-20 at 11 20 19 PM](https://github.com/user-attachments/assets/758ff5d0-6400-4df5-ac68-175f3afd73e6)  

Graficas X1 y X2:  
![WhatsApp Image 2025-02-20 at 11 21 30 PM](https://github.com/user-attachments/assets/461069d2-ad11-4cde-ab20-f19173e0b265)  


# Correlación  Manual entre Señales Senoidal y Cosenoidal
Se calcula el coeficiente de correlación (fuerza de relación entre dos señales) reemplazando valores para n desde 0 hasta 8 en las señales x1(n) y x2(n) para determinar sus variaciones y similitudes al pasar un determinado tiempo.
![WhatsApp Image 2025-02-20 at 11 22 43 PM](https://github.com/user-attachments/assets/ec9eb3f0-d6be-42a5-b4bd-cc88bdbfc3e0)  


# Guía de usuario 
En el encabezado del código encontramos la inicialización de las librerias correspondientes para el correcto funcionamiento del código
 ```pyton 
import os  # ubicacion archivo
import wfdb  # señal
import matplotlib.pyplot as plt  # graficas
import numpy as np # operaciones matemáticas 
from scipy.stats import norm, gaussian_kde # estadísticas avanzadas y Función de Probabilidad
import statistics # cálculos estadísticos básicos
from scipy.fftpack import fft #para el uso de la transformada rápida de Fourier
from scipy.signal import welch #para estimar la Densidad Espectral de Potencia
```
Es importante modificar la ruta de ubicación, se aconceja tener los archivos ".dat" y ".hea" junto al archivo del código en una misma carpeta para su correcta compilación
 ```pyton
os.chdir(r'C:\Users\Esteban\Pictures\Convolucion,Correlacion y,Transformaci-n')
datos, info = wfdb.rdsamp('a01', sampfrom=50, sampto=1000)
datos = np.array(datos).flatten()
```
## Análisis de la Convolución Discreta entre Señal y Sistema

Posteriormente, se implementó el ejercicio usando  la operación predefinida  de convolución discreta en Python  "np.convolve"   por lo cuál  se pudo analizar cómo la señal de entrada fue modificada por la respuesta al impulso y por ultimo obteniendo la representación gráfica y secuencial para relacionarlo con los resultados manuales.

```pyton
x = np.array([5,6,0,0,4,9,6])  # Señal 1 código inicializado
h = np.array([1,0,2,1,3,9,2,6,7,8])  # Señal 2 cédula inicializada

y = np.convolve(x, h, mode='full')
print("Resultado de la convolución Integrante 1/2/3:", y)
```
Ecuación de la convolución discreta
<p align="center"><img src="https://dademuchconnection.wordpress.com/wp-content/uploads/2020/09/null-70.png?w=300"/></p>



##  Correlación de Pearson entre Señales Senoidal y Cosenoidal 
Se realiza el coeficiente de  correlación entre dos señales periódicas una función coseno y una función seno, ambas con una frecuencia de 100 Hz y muestreadas con un período de Ts= 1.25ms, es importante tener en cuenta que si se obtiene un valor de 1 indica una correlación positiva perfecta, -1 indica una correlación negativa perfecta y 0 sugiere que no hay relación lineal. 
Las señales dadas están definidas como:
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/Se%C3%B1ales%20senoidal%20%20y%20cosenoidal.png?raw=true"/></p>

donde la correlación entre x1(n) y x2(n) se define como: 
<p align="center"><img src="https://www.hubspot.com/hs-fs/hubfs/F%C3%B3rmula%20de%20la%20correlaci%C3%B3n.png?width=400&height=141&name=F%C3%B3rmula%20de%20la%20correlaci%C3%B3n.png"/></p>

Representamos la formula de correlación de Pearson respecto a nuestras señales como: 
```pyton
def correlacion_pearson(x1, x2):  "Convierte los datos en arreglos Numpy para facilitar a python las opreaciones matemáticas"
    x1 = np.array(x1)
    x2 = np.array(x2)     
    mean_x1 = np.mean(x1)     "Promedio de los arreglos"
    mean_x2 = np.mean(x2)

 "covarizanza con los promedios para representar el numerador en la formula"
    numerador = np.sum((x1 - mean_x1) * (x2 - mean_x2))  

"multiplicación desviacion estandar correspondiente a las dos señales como denominador"
    denominador = np.sqrt(np.sum((x1 - mean_x1)**2)) * np.sqrt(np.sum((x2 - mean_x2)**2))  

    "evitar desviación por 0"
    return 0 if denominador == 0 else numerador / denominador

"Definir las señales"
x1 = np.array([1, 0.972, 0.890, 0.760, 0.587, 0.382, 0.155, -0.079, -0.309])
x2 = np.array([0, -0.233, -0.454, -0.649, -0.809, -0.924, -0.987, -0.996, -0.950])
r = correlacion_pearson(x1, x2)
```
# Caracterización y Transformación de Señal ECG
Se incluye una señal de ECG digitalizada continua, un conjunto de anotaciones de apnea identificadas como derivadas por expertos humanos sobre la base de la respiración registrada simultáneamente y las señales relacionadas y un conjunto de anotaciones de QRS generadas por máquina(simulación) en las que todos los latidos, independientemente del tipo, se han etiquetado como normales.
Se caracterizó la señal en el dominio del tiempo mediante el cálculo de estadísticos descriptivos como la media, mediana y desviación estándar, permitiendo analizar su comportamiento y variabilidad. Además, se determinó la frecuencia de muestreo para evaluar la resolución temporal y verificar el cumplimiento del criterio de Nyquist. Finalmente, se representó gráficamente la señal para identificar patrones, tendencias y posibles interferencias.

<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/Se%C3%B1al%20ECG.png?raw=true"/></p>

Los estadisticos realizados con funciones predefinidas correspondientes para la media, mediana, Coeficiente de variación  y desviación estandar , donde usamos la biblioteca "numpy" la cual nos proporciona calculos eficientes para ciertas operaciones matemáticas y "info['fs']" para obtener la frecuencia de muestreo guardada en los datos de la señal
```pyton
mean = np.mean(datos)
print(f"Media Numpy: {mean}")
desviacion_muestral = np.std(datos, ddof=1)
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")
fs = info['fs']
N = len(datos)
t = np.arange(N) / fs
```
## Transformada Rápida de Fourier
Se aplicó la Transformada de Fourier (FFT) a la señal para analizar su contenido en el dominio de la frecuencia. Posteriormente, se graficó la transformada, mostrando la distribución de las frecuencias presentes en la señal. Además, se representó la densidad espectral de potencia, lo que permitió visualizar la contribución energética de cada frecuencia y facilitar la interpretación del comportamiento espectral de la señal.
<p align="center"><img src="https://cdn.svantek.com/wp-content/uploads/2023/08/FFT-Fast-Fourier-Transform-Formula-300x98.jpg"/></p>

Primero, se utiliza "fft(datos)", que convierte la señal del dominio del tiempo al dominio de la frecuencia para analizar su contenido espectral, "np.fft.fftfreq(N, d=1/fs)" calcula el las frecuencias de la FFT, donde N es la cantidad de muestras y fs la frecuencia de muestreo hallada anteriormente, solo se toman las frecuencias positivas con "freqs[:N//2]" pues la función detecta que la FFT es simétrica , se extrae la mágnitud con la función "(fft_values[:N//2])" y se grafica la señal.

```pyton
fft_values = fft(datos)  "fft funcion que evaluacion fourier en funcion del tiempo" 
freqs = np.fft.fftfreq(N, d=1/fs) "calcula la frecuencia dependiendo de n  y periodo de muestreo"

plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(fft_values[:N//2]), label="Transformada de Fourier", color='m')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|Frecuencia| (Hz)")
plt.title("Transformada de Fourier de la Señal EMG")
plt.legend()
plt.grid()
plt.show()
```
### Explicación Magnitud
La siguiente función predefinida "fft_values" representa muchos arreglos de números complejos que se hallan a partir de la fft, estos se ven representados por magnitudes y fases, en este caso nos interesa la magnitud para saber la energia que tiene cada frecuencia y determinar los estadisticos, "np.abs" nos ayuda a determinar cuánta energía tiene cada frecuencia sin considerar la fase.

```pyton
magnitudes = np.abs(fft_values)
```
<p align="center"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBsnC1OGC_tbo92E-GsxqtgYOGlgJiyfhAQcah5xwnZJee_LyQRdTzttNO7-VUfPGYHw&usqp=CAU"/></p>

# Densidad Espectral
Ya con el conocimiento previo de como surgen las magnitudes se puede hallar la densidad espectral que para este caso es de potencia, se identifica  las frecuencias donde se concentra la mayor cantidad de energía, y se grafica la PSD, en el eje horizontal se encuentra la frecuencia en Hertz (Hz), mientras que el eje vertical muestra la densidad de potencia en unidades de 𝑚𝑉2/𝐻𝑧

```pyton
psd = np.abs(fft_values) ** 2 / N  "Calcula la magnitud de fft y lo eleva al cuadrado para obtener la magnitud de la potencia"

def suavizar(y, box_pts=10):  "def suavizar ayuda a eliminar el ruido representado en la señal"
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')
psd_suave = suavizar(psd, box_pts=10)
```
# Análisis Estadístico en función de la Frecuencia 
Se realiza este análisis por que permite caracterizar la distribución espectral de la señal, primero la frecuencia media se calcula ponderando las frecuencias por la magnitud de la Transformada de Fourier, la frecuencia mediana es el valor que divide la distribución de energía en dos partes iguales, la desviación estándar mide la dispersión de las frecuencias alrededor de la media, finalmente el histograma de frecuencias muestra la distribución de la energía en distintos rangos de frecuencia.

```pyton
media_fft = np.mean(magnitudes)         "se aplica los estadisticos con funciones predefinidas a la magnitud previamente obtenida"
mediana_fft = np.median(magnitudes)
desviacion_fft = np.std(magnitudes)

frecuencia_media = np.sum(freqs[:N//2] * magnitudes[:N//2]) / np.sum(magnitudes[:N//2])  "Frecuencia media (ponderada por la magnitud de la FFT)"

acumulada = np.cumsum(magnitudes[:N//2])
frecuencia_mediana = freqs[:N//2][np.where(acumulada >= acumulada[-1]/2)[0][0]]    "Frecuencia mediana (basada en la suma acumulada de magnitudes)"

desviacion_frecuencia = np.sqrt(np.sum(((freqs[:N//2] - frecuencia_media)**2) * magnitudes[:N//2]) / np.sum(magnitudes[:N//2])) "Desviación estandar"
```
Los calculos anteriores son en base a las formulas matemáticas correspondientes a :

### Frecuencia Media (ponderada por la magnitud de la FFT)
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/Frecuencia%20Media%20(ponderada%20por%20la%20magnitud%20de%20la%20FFT).png?raw=true"></p>

### Frecuencia Mediana (basada en la suma acumulada de magnitudes)

<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/Frecuencia%20Mediana%20(basada%20en%20la%20suma%20acumulada%20de%20magnitudes).png?raw=true"/></p>

### Desviación Estandar de la Frecuencia
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/Desviaci%C3%B3n%20Est%C3%A1ndar%20de%20la%20Frecuencia.png?raw=true"></p>


# Resultados 

<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20INT1.png?raw=true"></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20INT2.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20INT3.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20CORRELACION.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/HISTOGRAMA%20SE%C3%91AL%20ECG.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20TRANSFORMADA.png?raw=true  "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/GRAFICA%20DENSIDAD%20ESPECTRAL.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/HISTOGRAMA%20FRECUENCIA%20ESPECTRAL.png?raw=true "></p>
<p align="center"><img src="https://github.com/1ebarry/Convoluci-n-Correlaci-n-y-Transformaci-n/blob/main/RESULTADOSS.png?raw=true "></p>


## Información adicional


Por último y no menos importante vemos que el coeficiente de correlación de Pearson tiene un valor (0.8467185111073606) casi cercano a 1 por lo que nos indica que existe una correlación positiva, podria decirse semi perfecta o casi perfecta.... Por otro lado vemos que nuestra señal cumple satisfactoriamente con el criterio de Nyquist el cual nos dice  que la frecuencia de muestreo  𝑓𝑠(100Hz) es al menos el doble de la máxima frecuencia 𝑓𝑚𝑎𝑥(50Hz) Esto garantiza que la señal puede ser reconstruida sin pérdida de información ni aliasing.




## Bibliografía
[1] PhysioNet Support, MIT. (2013). *Dataset Description*. Recuperado de [PhysioNet](https://www.physionet.org)

## Licencia 
Open Data Commons Attribution License v1.0

## Temas:
# 📡 Procesamiento de Señales  
- Adquisición y preprocesamiento  
- Convolución y correlación  
- Estadísticas básicas  

# 🔊 Análisis en Frecuencia  
- Transformada de Fourier  
- Densidad Espectral de Potencia (PSD)  

# 📊 Estadísticos Descriptivos  
- Frecuencia media y mediana  
- Desviación estándar  
- Histograma de frecuencias  

# 🖥️ Código e Implementación  
- Explicación del código  
- Ejecución y ejemplos  
- Mejoras y optimización  


