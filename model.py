import tensorflow as tf
import os
from util import preprocess, convert_time
from keras_contrib.layers import CRF
import numpy as np
import json
import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print(
            'memory growth:',
            tf.config.experimental.get_memory_growth(
                physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


allow_ext = [".wav", ".mp3", ".ogg"]
quality_index = {}
current_dir = os.getcwd()

sess = tf.Session()
graph = tf.get_default_graph()

keras.backend.set_session(sess)
model = keras.models.load_model(
    current_dir + "/model/chord_estimation_model.h5",
    compile=False,
    custom_objects={
        "CRF": CRF,
        "mish": mish})


with open(current_dir + "/index.json", mode="r") as f:
    quality_index = json.load(f)


def predict(path):
    print(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("正しくないファイルパスが入力されました。")

    if not any([os.path.splitext(path)[-1].lower() == ext for ext in allow_ext]):
        raise ValueError("処理できる音声ファイルは " + ",".join(allow_ext) + "のいずれかです。")

    try:
        filename = os.path.splitext(os.path.basename(path))[0]

        print("音声読み込み中...")
        S, bins_per_seconds = preprocess(path, mono=False, hop_length=512)

        print("モデル予測中...")
        global sess
        global graph
        
        with graph.as_default():
            keras.backend.set_session(sess)
            pred = model.predict(np.expand_dims(S, 0))

        print("変換中...")
        times = convert_time(pred, bins_per_seconds, quality_index)

        print("ファイル保存中...")
        with open(current_dir + "/label/{}.txt".format(filename), mode="w") as f:
            for t in times:
                if t[2] != "N.C.":
                    f.write("{}	{}	{}\n".format(t[0], t[1], t[2]))
        print("/label/{}.txtに保存しました。".format(filename))

        return times

    except Exception as e:
        raise e
