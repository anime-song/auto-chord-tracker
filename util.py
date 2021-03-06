import librosa
import numpy as np

from dlchord import Chord


def convert_time(pred, bins_per_seconds, chord_index, min_time=0.1):
    result = pred[0][0]
    bass_result = pred[1][0]

    times = []
    tones = [
        "C",
        "Db",
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
    ]

    s_time = 0
    e_time = 0
    last = 0
    now_chord = ""
    chord = ""

    for i, t in enumerate(result):
        if np.argmax(bass_result[i]) != 0:
            last = i

    for i, t in enumerate(result):
        bass_line = np.argmax(bass_result[i])

        if bass_line != 0:
            chord = chord_index[str(np.argmax(t))]
            if chord != "N.C.":
                if Chord(chord).bass != (bass_line - 1):
                    chord += "/{}".format(tones[bass_line - 1])

        if i == 0:
            now_chord = chord
            continue

        if now_chord != chord or i == last or i == len(result) - 1:
            e_time = (i / bins_per_seconds)

            if abs(s_time - e_time) > min_time:
                if len(times) > 1:
                    if times[-1][-1] == now_chord:
                        times[-1][-2] = e_time
                    else:
                        try:
                            modify_chord = Chord(now_chord).modify().chord
                        except ValueError:
                            modify_chord = now_chord
                        
                        times.append([s_time, e_time, modify_chord])
                else:
                    try:
                        modify_chord = Chord(now_chord).modify().chord
                    except ValueError:
                        modify_chord = now_chord
                    times.append([s_time, e_time, modify_chord])

                s_time = e_time
            now_chord = chord

    return times


def standard(x):
    x = (x - np.mean(x)) / np.std(x)
    return x


def preprocess(path, sr=22050, hop_length=512, mono=False):
    y, sr = librosa.load(path, sr=sr, mono=mono)

    S_l = np.abs(
        librosa.cqt(
            y[0],
            sr=sr,
            hop_length=hop_length,
            n_bins=12 * 3 * 7,
            bins_per_octave=12 * 3)).astype("float32")
    S_r = np.abs(
        librosa.cqt(
            y[1],
            sr=sr,
            hop_length=hop_length,
            n_bins=12 * 3 * 7,
            bins_per_octave=12 * 3)).astype("float32")
    S = np.array((S_l.T, S_r.T))
    S = standard(S)
    
    S = np.concatenate((S[0], S[1], (S[0] - S[1])), axis=-1)
    bins_per_seconds = (S.shape[-2] / (y.shape[-1] / sr))

    return S, bins_per_seconds
