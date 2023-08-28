## ArgentinIA

### Instalación y Uso

- Ambiente virtual

Crear un ambiente virtual para compilar y ejecutar el comando:
```
$ cd <CARPETA>
$ python -m venv .venv
```
Activar el ambiente virtual
```
$ source .venv/bin/activate
```
Instalar bibliotecas
```
$ pip install -r src/requirements.txt
$ pip install -r src/yolov7/requirements.txt
```
- Instalar Tesseract, swig y JamSpell
```
$ sudo sh install.sh
$ cd src/JamSpell
$ python3 setup.py build_ext
$ python3 setup.py build_py
$ python3 setup.py install
```
- Uso

Para procesar las imágenes ubicadas la carpeta de origen input_data/ y guardar los resultados en la carpeta de destino out_data/, ejecutar:
```
$ python3 run.py
```
o desde el intérprete de python:
```python
procesar_imgs(“input_data”, “out_data”)
```

En caso de querer usar otros directorios, usar los flags -e y -s respectivamente. Por ejemplo, si quiero leer imágenes desde el directorio entrada/ y que la salida sea en salida/, ejecutar 
```
$ python3 run.py -e entrada/ -s salida/ o procesar_imgs(“entrada”, “salida”).
```

Adicionalmente, se puede pasar por parámetro un modelo de corrector ortográfico distinto al default (herramientas/model_juridico.bin) con el flag –modelo-corrector o procesar_imgs(“entrada”, “salida”, “path del modelo”).


