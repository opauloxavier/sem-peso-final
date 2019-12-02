import numpy as np
import pandas as pd
import glob
import string
import codecs
from tqdm import tqdm
from PIL import Image

path = './dataset/*.png'
files = glob.glob(path)

# itera nos arquivos e cria o dataframe, com '0' nos bytes


def contructDataframe(file_list):
    data = []
    for file in tqdm(file_list):
        data.append((file, file.split("/")[-1].split("__")[0], 0))
    return pd.DataFrame(data, columns=['path', 'label', 'bytes'])


df = contructDataframe(files)

# array que terá o mesmo numero de linhas do dataframe, para substituição da coluna "bytes"
byteArray = []
for index, row in tqdm(df.iterrows()):
    # converte a imagem para escala L e usa como dithering o algoritmo de FLOYDSTEINBERG
    img = Image.open(row["path"]).convert("L", dither=Image.FLOYDSTEINBERG)
    # transforma os bytes em array
    arr = np.array(img)
    # faz do array flat
    flat_arr = arr.ravel().tolist()
    # da append na variavel de byteArray
    byteArray.append(flat_arr)

# substitui a coluna "bytes" no df pelos bytes das imagens
df['bytes'] = byteArray

# dropa a coluna "path" por não ter mais uso e salva um csv localmente
df.drop(columns=['path']).to_csv(
    'coil.csv', sep=',', index=False, encoding='utf-8')
