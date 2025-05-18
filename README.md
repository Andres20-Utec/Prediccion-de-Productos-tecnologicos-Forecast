# Proyecto UTEC Overall - Guía de Ejecución

Este proyecto tiene como objetivo [Descripción breve del propósito del proyecto].

## Estructura del proyecto

La estructura básica del proyecto es la siguiente

```bash
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── dataTableFinal.csv
│   └── data_prueba.txt
├── notebooks
│   └── Copia_de_Tienda_Familia.ipynb
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── data
    ├── evaluation
    ├── features
    ├── models
    └── utils
```

### Requisitos previos

- **Python** 3.x (3.9 por defecto)
- **Pip** para manejar las dependencias
- **Virtualenv** (recomendado para aislar el entorno del proyecto)

### Configuración del entorno

#### 1. Clonar el repositorio

Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/usuario/proyecto_utec_overall.git
cd proyecto_utec_overall
# Crear el entorno virtual
python3 -m venv overall_dev

# Activar el entorno virtual
# En macOS/Linux
source overall_dev/bin/activate

# En Windows
overall_dev\Scripts\activate

# Ejecutar el archivo setup.py para instalar dependencias y variables
pip install .

# Ejecutar make para validar el flujo de ejecucion
make
```