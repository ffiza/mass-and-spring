<div align="center">
    <h1>Animaciones de masas y resortes</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Código de Python para resolver la dinámica de masas y resortes. El script principal es `code/simulation.py`. `code/utils.py` contiene funciones útiles y `code/animation.py` genera una animación utilizando PyGame. Todas las unidades están en el sistema internacional.

Para correr tus propias simulaciones tenés que crear una nueva carpeta en `configs/` que debe contener varios archivos relacionados con la configuración.

- `physics.yml`: Archivo que contiene el paso temporal de la simulación, la cantidad de pasos, la aceleración de la gravedad y el coeficiente de fricción.
- `ic.csv`: Archivo que contiene la condición inicial de cada partícula. Además, el campo `DynamicParticle` indica si la partícula es dinámica (`1`) o estática (`0`), en cuyo caso se mantendrá fija durante toda la simulación.
- `elastic_constants.csv`: Archivo que contiene la constante elástica de todos los resortes. En esta matriz, el par `(i, j)` indica el valor de la constante elástica del resorte que une la partícula `i` con la partícula `j`. La diagonal y el triángulo inferior deben ser nulos porque la fuerza elástica es antisimétrica.
- `natural_lengths.csv`: Análogo a `elastic_constants.csv` pero para la longitud natural de cada resorte.

Una vez creados los archivos de configuración, se puede correr la simulación utilizando

```
python code/simulation.py --config [nombre_carpeta_config]
```

Esto generará un archivo con el nombre `nombre_carpeta_config` en `results/`. Luego, para generar una animación en PyGame usar:

```
python code/animation.py --result [nombre_carpeta_config]
```

### Ejemplos

El repositorio incluye algunos ejemplos, sus condiciones iniciales y los resultados. A continuación se muestran algunas imagenes de las simulaciones incluidas.

<p align="center">
    <a href="https://i.imgur.com/XHzp6aF.png"><img src="https://i.imgur.com/XHzp6aF.png" width=300></a>
    <a href="https://i.imgur.com/PdtuuLV.png"><img src="https://i.imgur.com/PdtuuLV.png" width=300></a>
    <a href="https://i.imgur.com/JCKt7aD.png"><img src="https://i.imgur.com/JCKt7aD.png" width=300></a>
    <a href="https://i.imgur.com/jKlT3YP.png"><img src="https://i.imgur.com/jKlT3YP.png" width=300></a>
    <a href="https://i.imgur.com/8FteqXH.png"><img src="https://i.imgur.com/8FteqXH.png" width=300></a>
    <a href="https://i.imgur.com/26XcjxW.png"><img src="https://i.imgur.com/26XcjxW.png" width=300></a>
</p>
