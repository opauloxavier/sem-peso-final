import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import string
import codecs
from tqdm import tqdm
from PIL import Image

path = './dataset/*.png'
# list files
files = glob.glob(path)


def contructDataframe(file_list):
    data = []
    for file in tqdm(file_list):
        data.append((file, file.split("/")[-1].split("__")[0], 0))
    return pd.DataFrame(data, columns=['path', 'label', 'bytes'])


df = contructDataframe(files)
byteArray = []
for index, row in tqdm(df.iterrows()):
    img = Image.open(row["path"]).convert("L", dither=Image.FLOYDSTEINBERG)
    arr = np.array(img)
    flat_arr = arr.ravel()
    byteArray.append(flat_arr)

df['bytes'] = byteArray

df.drop(columns=['path'])

df.to_csv('coil.csv', sep=',', encoding='utf-8')
