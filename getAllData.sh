pip install -U gdown
apt-get update && apt-get install unzip
rm -r ./data
mkdir data
gdown https://drive.google.com/uc?id=1PDiWRox4FAP_T7y2_uBdkDMQBFCeIaV7
unzip Chennai-Dataset.zip -d ./data
gdown https://drive.google.com/uc?id=1K9oQ0vzfIzosKsHDJE7g_HRLQB2vKl9a
unzip CityScape-Dataset.zip -d ./data
gdown https://drive.google.com/uc?id=1EV-03rvQPP9mqp7P8Ep18isSZBdFiJos
unzip Chennai-Dataset-Unified.zip -d ./data
gdown https://drive.google.com/uc?id=1Q9NDeex5XJrCzVseXnYNfuj3HBmyaF4c
unzip CityScape-Dataset-Unified.zip -d ./data
gdown https://drive.google.com/uc?id=1uv60hVJPYJ3SCo9FaBpIpIXWHinhiinQ
unzip Chennai-Dataset-KFold.zip -d ./data