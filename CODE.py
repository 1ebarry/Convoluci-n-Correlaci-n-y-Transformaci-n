import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import wfdb
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.stats import norm, gaussian_kde
import statistics


# Datos y gráficos Lizeth


x = np.array([5,6,0,0,4,9,6])  # Señal 1
h = np.array([1,0,2,1,3,9,2,6,7,8])  # Señal 2

y = np.convolve(x, h, mode='full')
print("Resultado de la convolución Lizeth:", y)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.stem(x)
plt.title("Señal x1 Lizeth")
plt.xlabel("n")
plt.ylabel("x1")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(h)
plt.title("Señal x2 Lizeth")
plt.xlabel("n")
plt.ylabel("x2")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(y)
plt.title("Resultado de la convolución Y[n] Lizeth")
plt.xlabel("n")
plt.ylabel("Y(n)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Datos y gráficos Esteban


g = np.array([5,6,0,0,6,1,5])  # Señal 1
z = np.array([1,0,2,5,5,2,6,2,3,9])  # Señal 2

y_esteban = np.convolve(g, z, mode='full')
print("Resultado de la convolución Esteban:", y_esteban)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.stem(g)
plt.title("Señal e1 Esteban")
plt.xlabel("n")
plt.ylabel("g")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(z)
plt.title("Señal e2 Esteban")
plt.xlabel("n")
plt.ylabel("z")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(y_esteban)
plt.title("Resultado de la convolución Y[n] Esteban")
plt.xlabel("n")
plt.ylabel("Y(n)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Datos y gráficos Valentina


q = np.array([5,6,0,0,6,4,9])  # Señal 1
n = np.array([1,0,2,6,5,5,2,1,8,2])  # Señal 2

y_valentina = np.convolve(q, n, mode='full')
print("Resultado de la convolución Valentina:", y_valentina)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.stem(q)
plt.title("Señal v1 Valentina")
plt.xlabel("n")
plt.ylabel("q")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(n)
plt.title("Señal v2 Valentina")
plt.xlabel("n")
plt.ylabel("n")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(y_valentina)
plt.title("Resultado de la convolución Y[n] Valentina")
plt.xlabel("n")
plt.ylabel("Y(n)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Correlación de Pearson


def correlacion_pearson(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    ##media de las señales
    mean_x1 = np.mean(x1)
    mean_x2 = np.mean(x2)
    
    ##numerador: covarianza
    numerador = np.sum((x1 - mean_x1) * (x2 - mean_x2))
    
    # Denominador: producto de las desviaciones estándar
    denominador = np.sqrt(np.sum((x1 - mean_x1)**2)) * np.sqrt(np.sum((x2 - mean_x2)**2))
    
    #evitar desviación por 0
    return 0 if denominador == 0 else numerador / denominador

#Definir las señales
x1 = np.array([1, 0.972, 0.890, 0.760, 0.587, 0.382, 0.155, -0.079, -0.309])
x2 = np.array([0, -0.233, -0.454, -0.649, -0.809, -0.924, -0.987, -0.996, -0.950])

r = correlacion_pearson(x1, x2)
print(f"Correlación de Pearson: {r}")

plt.figure(figsize=(11, 10))
plt.plot(x1, x2, 'o')  #usar o para mostrar unicamente puntos 
plt.title("Correlación entre x1 y x2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.tight_layout()
plt.show()


# Lectura y análisis de la señal ECG/EMG


os.chdir(r'C:\Users\Esteban\Pictures\Convolucion,Correlacion y,Transformaci-n')
datos, info = wfdb.rdsamp('a01', sampfrom=50, sampto=1000)
datos = np.array(datos).flatten()



mean = np.mean(datos)
print(f"Media Numpy: {mean}")
desviacion_muestral = np.std(datos, ddof=1)
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")

fs = info['fs']
N = len(datos)
t = np.arange(N) / fs

plt.figure(figsize=(10, 4))
plt.plot(t, datos, label="Señal EMG", color='c')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal ECG Apnea")
plt.legend()
plt.grid()
plt.show()


# Histograma y función de probabilidad sobre la señal original
plt.figure(figsize=(12, 6))
counts, bins, _ = plt.hist(datos, bins=30, density=True, color='b', alpha=0.7, edgecolor='black', label="Histograma")
centers = (bins[:-1] + bins[1:]) / 2
plt.plot(centers, counts, 'r-', label="Función de Probabilidad")
plt.xlabel("Amplitud de la Señal (mV)")
plt.ylabel("Probabilidad")
plt.title("Histograma y Función de Probabilidad de la Señal Original")
plt.legend()
plt.grid()
plt.show()

#Transformada de Fourier

fft_values = fft(datos)  #fft funcion que evaluacion fourier en funcion del tiempo 
freqs = np.fft.fftfreq(N, d=1/fs) #calcula la frecuencia dependiendo de n  y periodo de muestreo

plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(fft_values[:N//2]), label="Transformada de Fourier", color='m')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|Frecuencia| (Hz)")
plt.title("Transformada de Fourier de la Señal EMG")
plt.legend()
plt.grid()
plt.show()


# Análisis de la magnitud de la FFT


magnitudes = np.abs(fft_values)
# fft_values es un arreglo de números complejos que se hallan a partir de la fft,
# estos se ven representados por magnitudes y fases, en este caso nos interesa la magnitud
#para saber la energia que tiene cada frecuencia, np.abs nos ayuda a determinar cuánta energía
 #tiene cada frecuencia sin considerar la fase.




# Densidad espectral de potencia (PSD)


psd = np.abs(fft_values) ** 2 / N  ##Calcula la magnitud de los datos obtenidos con fft.
                                    #ylo eleva al cuadrado para obtener la magnitud de la potencia
def suavizar(y, box_pts=10):  #suavizar ayuda a eliminar el ruido representado en la señal 
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

psd_suave = suavizar(psd, box_pts=10)



# Estadísticas de la FFT


media_fft = np.mean(magnitudes)         #aplica los estadisticos con funciones predefinidas 
                                            #a la magnitud previamente obtenida
mediana_fft = np.median(magnitudes)
desviacion_fft = np.std(magnitudes)

print(f"Media: {media_fft:.4f}")
print(f"Mediana: {mediana_fft:.4f}")
print(f"Desviación estándar: {desviacion_fft:.4f}")




# Parámetros de muestreo
fs = info['fs']  # obtiene la frecuencia de muestreo del archivo de datos cargado.
N = len(datos)  # devuelve cuántos puntos de datos tiene la señal cargada.
t = np.arange(N) / fs  # factor de conversion para pasar a tiempo 




# Densidad Espectral de Potencia (PSD) con método de Welch

#La función welch de scipy.signal calcula la PSD usando el método de Welch,

freqs_psd, psd = welch(datos, fs=fs, nperseg=256) #nperseg Divide la señal en segmentos de longitud (256 muestras).
                    #Calcula la FFT de cada segmento y obtiene su potencia. promedia todas las potencias para eliminar ruido
plt.figure(figsize=(10, 4))
plt.semilogy(freqs_psd, psd, label="Densidad Espectral de Potencia", color='g')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (mV²/Hz)")
plt.title("Densidad Espectral de Potencia")
plt.legend()
plt.grid()
plt.show()


# Estadísticos Descriptivos en función de la frecuencia


# 1. Frecuencia media (ponderada por la magnitud de la FFT)
frecuencia_media = np.sum(freqs[:N//2] * magnitudes[:N//2]) / np.sum(magnitudes[:N//2])

# 2. Frecuencia mediana (basada en la suma acumulada de magnitudes)
acumulada = np.cumsum(magnitudes[:N//2])
frecuencia_mediana = freqs[:N//2][np.where(acumulada >= acumulada[-1]/2)[0][0]]

# 3. Desviación estándar de la frecuencia
desviacion_frecuencia = np.sqrt(np.sum(((freqs[:N//2] - frecuencia_media)**2) * magnitudes[:N//2]) / np.sum(magnitudes[:N//2]))

# 4. Histograma de frecuencias (ponderado por la magnitud)

plt.figure(figsize=(12, 6))
counts_freq, bins_freq, _ = plt.hist(freqs[:N//2], bins=30, weights=magnitudes[:N//2], color='b', alpha=0.7, edgecolor='black', label="Histograma Ponderado")
centers_freq = (bins_freq[:-1] + bins_freq[1:]) / 2

pdf_freq = counts_freq / np.sum(counts_freq)  # Normalización para probabilidad
plt.plot(centers_freq, pdf_freq, 'r-', linewidth=2, label="Función de Probabilidad")

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Probabilidad Ponderada")
plt.title("Histograma y Función de Probabilidad de Frecuencias Ponderado por Magnitud")
plt.legend()
plt.grid()
plt.show()


# Mostrar resultados estadísticos
print(f"Frecuencia Media: {frecuencia_media:.4f} Hz")
print(f"Frecuencia Mediana: {frecuencia_mediana:.4f} Hz")
print(f"Desviación Estándar de la Frecuencia: {desviacion_frecuencia:.4f} Hz")
