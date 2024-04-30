## Instalar las demos

Abrir el repository https://github.com/ravignad/analisis-estadistico-de-datos en un navegador

Bajar el archivo zip  analisis-estadistico-de-datos-main.zip

Mover analisis-estadistico-de-datos-main.zip a una carpeta

Descomprimir analisis-estadistico-de-datos-main.zip. 

'unzip analisis-estadistico-de-datos-main'

Se genera el directorio analisis-estadistico-de-datos-main con que contiene todas las demos

cd analisis-estadistico-de-datos-main

Crear un entorno virtual Python

'python3 -m venv venv'

Activar el entorno virtual

'source venv/activate/bin'

Instalar las dependencias:

'python3 -m pip install -r requirements.txt'

Crear kernel Jupyter 

python3 -m ipykernel install --user --name=analisis-datos

Correr Jupyter

'jupyter notebook'
