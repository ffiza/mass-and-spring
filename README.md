<div align="center">
    <h1>Animaciones de masas y resortes</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Códigos de Python para resolver y animar la dinámica de masas y resortes. El script principal es `code/simulation.py`, que simula la evolución de un sistema de masas y resortes usando un [integrador temporal tipo *leapfrog*](https://en.wikipedia.org/wiki/Leapfrog_integration). El script `code/animation.py` puede utilizarse para generar una animación de los resultados obtenidos utilizando PyGame. El script `code/utils.py` contiene algunas funciones útiles y `code/ui/` contiene scripts para la interfaz gráfica de PyGame. Todas las unidades están en el sistema internacional.

Para correr estos códigos es necesario instalar las siguientes librerías: [NumPy](https://numpy.org/), [PyGame](https://pyga.me/docs/), [pandas](https://pandas.pydata.org/docs/index.html), [tqdm](https://tqdm.github.io/) y [PyYAML](https://pypi.org/project/PyYAML/), lo cual puede hacerse mediante el comando:

```bash
pip install numpy pygame-ce pandas tqdm pyyaml
```

# Clonar o descargar este repositorio

Para tener una copia de este repositorio en tu sistema, podés clonar el mismo usando

```bash
git clone https://github.com/ffiza/mass-and-spring.git
```

o bien descargar el repositorio comprimido en formato `ZIP` yendo a `Code > Download ZIP`.

# Cómo animar las simulaciones incluidas

Para animar las simulaciones inlcuidas en este repositorio (cuyos resultados están almacenados en el directorio `results/`, en una archivo `CSV` por simulación) sólo es necesario ejecutar

```bash
python code/animation.py --result [nombre_resultados]
```

Por ejemplo, para animar los resultados dentro del directorio `run_05`, ejecutar

```bash
python code/animation.py --result run_05
```

La simulación incia pausada y puede activarse presionando `P`. Además, `R` resetea la animación y `D` muestra o esconde el panel del debugger.

# Cómo ejecutar tus propias simulaciones

Para ejecutar tus propias simulaciones, es necesario que crees una nueva carpeta dentro del directorio `configs/` y agregues archivos de configuración según se describe en la sección siguiente. Luego, simplemente podés correr

```bash
python code/simulation.py --config [nombre_carpeta_config]
```

Python va a correr toda la simulación (tené en cuenta que cuantas más partículas haya más lenta va a ser la ejecución) y guardar los resultados en el directorio `results/`, en un archivo `CSV` con el mismo nombre de la carpeta donde pusiste los archivos de configuración.

Finalmente, para animar tus resultados, podés correr

```bash
python code/animation.py --result [nombre_carpeta_config]
```

# Sobre los archivos de configuración

Para correr tus propias simulaciones tenés que crear una nueva carpeta en `configs/` que debe contener varios archivos relacionados con la configuración. Los mismos son los siguientes.

- `physics.yml`: Archivo que contiene el paso temporal de la simulación (`TIMESTEP`), la cantidad de pasos (`N_STEPS`), la aceleración de la gravedad (`GRAV_ACCELERATION`) y el coeficiente de fricción (`FRICTION_COEF`).
- `ic.csv`: Archivo que contiene la condición inicial de cada partícula (posiciones y velocidades) y su masa. Además, el campo `DynamicParticle` indica si la partícula es dinámica (`1`) o estática (`0`), en cuyo caso se mantendrá fija durante toda la simulación.
- `elastic_constants.csv`: Archivo que contiene la constante elástica de cada resorte. En esta matriz, el par `(i, j)` indica el valor de la constante elástica del resorte que une la partícula `i` con la partícula `j`. Tanto la diagonal como el triángulo inferior deben ser nulos porque la fuerza elástica es antisimétrica.
- `natural_lengths.csv`: Archivo que contiene la longitud natural de cada resorte. En esta matriz, el par `(i, j)` indica el valor de la longitud natural del resorte que une la partícula `i` con la partícula `j`. Tanto la diagonal como el triángulo inferior deben ser nulos porque la fuerza elástica es antisimétrica.

Una vez creados los archivos de configuración, se puede correr la simulación utilizando

```bash
python code/simulation.py --config [nombre_carpeta_config]
```

Esto generará un archivo con el nombre `nombre_carpeta_config` en `results/`. Luego, para generar una animación en PyGame usar:

```bash
python code/animation.py --result [nombre_carpeta_config]
```

# Ejemplos

El repositorio incluye algunos ejemplos, sus condiciones iniciales y los resultados. A continuación se muestran algunas imagenes de las simulaciones incluidas.

<p align="center">
    <a href="https://i.imgur.com/XHzp6aF.png"><img src="https://i.imgur.com/XHzp6aF.png" width=300></a>
    <a href="https://i.imgur.com/PdtuuLV.png"><img src="https://i.imgur.com/PdtuuLV.png" width=300></a>
    <a href="https://i.imgur.com/JCKt7aD.png"><img src="https://i.imgur.com/JCKt7aD.png" width=300></a>
    <a href="https://i.imgur.com/jKlT3YP.png"><img src="https://i.imgur.com/jKlT3YP.png" width=300></a>
    <a href="https://i.imgur.com/8FteqXH.png"><img src="https://i.imgur.com/8FteqXH.png" width=300></a>
    <a href="https://i.imgur.com/26XcjxW.png"><img src="https://i.imgur.com/26XcjxW.png" width=300></a>
</p>
