import numpy as np
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.filters import threshold_local
from sklearn.model_selection import KFold
import glob
import string
import codecs
from tqdm import tqdm
from PIL import Image, ImageFilter
import wisardpkg as wp

path = './dataset/*.png'
files = glob.glob(path)

# itera nos arquivos e cria o dataframe, com '0' nos bytes

# fixa o seed
random.seed(30)


def filter(img):
    local_threshold = threshold_local(
        np.array(img), 11, 'gaussian', param=3, offset=2)
    binary_local = img > local_threshold

    binaryImage = Image.fromarray(np.array(binary_local))

    return binaryImage


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

    # SEM ADAPTIVE MEAN ThRESHOLD
    # binaryImage = Image.open(row["path"]).convert(
    # "1", dither=Image.FLOYDSTEINBERG)
    # COM FILTRO
    binaryImage = filter(img)

    # transforma os bytes em array
    arr = np.array(binaryImage)
    # faz do array flat
    flat_arr = arr.ravel().tolist()
    # da append na variavel de byteArray
    byteArray.append(flat_arr)

# substitui a coluna "bytes" no df pelos bytes das imagens
df['bytes'] = byteArray

# dropa a coluna "path" por não ter mais uso e salva um csv localmente
df.drop(columns=['path']).to_csv(
    'coil.csv', sep=',', index=False, encoding='utf-8')

n_folds = 3
address = 15
labels = df['label'].values
minScore = 0.1
discriminatorLimit = 3
threshold = 24

print("Número de folds: ", n_folds)
print("Address Size: ", address)

kfold = KFold(n_folds, shuffle=True)

acuracia_list = []
acertos_list = []
for train_ix, test_ix in kfold.split(byteArray):
    wsd = wp.Wisard(address, ignoreZero=False, verbose=False)
    # wsd = wp.ClusWisard(address, minScore, threshold,
    #                     discriminatorLimit, verbose=False)

    trainX, trainY, testX, testY = np.take(
        byteArray, train_ix, 0), np.take(labels, train_ix, 0), np.take(byteArray, test_ix, 0), np.take(labels, test_ix, 0)
    print("vai treinar")
    wsd.train(trainX, trainY)
    print("treinou")
    out = wsd.classify(testX)
    print("classificou")

    # acertos=0
    # for i, d in enumerate(testY):
    #     if(out[i] == testY[i]):
    #         acertos = acertos + 1

    print("Matriz de confusão:")
    print(confusion_matrix(testY, out))

    # array = confusion_matrix(testY, out)
    # df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],
    #                 columns = [i for i in "0123456789"])
    # plt.figure(figsize = (10,10))
    # sn.heatmap(df_cm, annot=True, cmap="RdPu")
    # plt.show()

    # print("Acurácia: ", (acertos/len(out))*100)
    ac_score = accuracy_score(testY, out, normalize=True)
    acerto = accuracy_score(testY, out, normalize=False)
    acuracia_list.append(ac_score)
    acertos_list.append(acerto)
    patterns = wsd.getMentalImages()
    print("Imagem Mental:")
    # for key in patterns:
    #     #print(key, patterns[key])
    #       pixels = np.array(patterns[key], dtype='uint8')
    #       pixels = pixels.reshape((28, 28))
    #       plt.imshow(pixels, cmap='viridis')
    #       plt.show()
    # pixels0 = np.array(patterns["0"])
    # pixels0 = pixels0.reshape((28, 28))
    # plt.imshow(pixels0, cmap='viridis')
    # plt.show()

    # pixels1 = np.array(patterns["1"])
    # pixels1 = pixels1.reshape((28, 28))
    # plt.imshow(pixels1, cmap='viridis')
    # plt.show()

    # pixels2 = np.array(patterns["2"])
    # pixels2 = pixels2.reshape((28, 28))
    # plt.imshow(pixels2, cmap='viridis')
    # plt.show()

    # pixels3 = np.array(patterns["3"])
    # pixels3 = pixels3.reshape((28, 28))
    # plt.imshow(pixels3, cmap='viridis')
    # plt.show()

    # pixels4 = np.array(patterns["4"])
    # pixels4 = pixels4.reshape((28, 28))
    # plt.imshow(pixels4, cmap='viridis')
    # plt.show()

    # pixels5 = np.array(patterns["5"])
    # pixels5 = pixels5.reshape((28, 28))
    # plt.imshow(pixels5, cmap='viridis')
    # plt.show()

    # pixels6 = np.array(patterns["6"])
    # pixels6 = pixels6.reshape((28, 28))
    # plt.imshow(pixels6, cmap='viridis')
    # plt.show()

    # pixels7 = np.array(patterns["7"])
    # pixels7 = pixels7.reshape((28, 28))
    # plt.imshow(pixels7, cmap='viridis')
    # plt.show()

    # pixels8 = np.array(patterns["8"])
    # pixels8 = pixels8.reshape((28, 28))
    # plt.imshow(pixels8, cmap='viridis')
    # plt.show()

    # pixels9 = np.array(patterns["9"])
    # pixels9 = pixels9.reshape((28, 28))
    # plt.imshow(pixels9, cmap='viridis')
    # plt.show()

    print("Numero de acertos:", acerto)
    print("Porcentagem em acuracy_score:", ac_score)
    print("Porcentagem em acuracy_score com % :", ac_score*100)

print("Media Acurácia:", np.mean(acuracia_list))
print("Média Variância:", np.var(acuracia_list))
print("Média Desvio Padrão:", np.std(acuracia_list))
print("Média de acertos:", np.mean(acertos_list))
