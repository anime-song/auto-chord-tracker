import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fl
from tkinter import messagebox as mb

import threading
import model
import pywfd

times = []


def openfile():
    filepath = fl.askopenfilename(
        filetypes=[(
            "音声ファイル",
            "*.wav;*.mp3;*.ogg")])
    pb1.start(20)
    try:
        global times
        times = model.predict(filepath)

    except Exception as e:
        mb.showinfo("エラー", e)

    pb1.stop()
    button_audio_select.configure(state=tk.NORMAL)
    lock.release()


def savewfd():
    if not times:
        mb.showinfo("エラー", "先に解析を行ってください")
    else:
        filepath = fl.askopenfilename(filetype=[("WFDファイル", "*.wfd")])

        try:
            wfd = pywfd.load(filepath)
            chords = wfd.chords.label_to_array(times)
            wfd.chords = chords
            pywfd.write(filepath, wfd)

        except ValueError:
            mb.showinfo("エラー", "WFDファイルの読み込みに失敗しました。 コード解析を実行していない可能性があります。")
        except Exception as e:
            mb.showinfo("エラー", e)
            

def callback():
    if lock.acquire(blocking=False):
        button_audio_select.configure(state=tk.DISABLED)
        th = threading.Thread(target=openfile)
        th.start()
    else:
        mb.showinfo("処理中です。")


global button_audio_select, lock, pb1

lock = threading.Lock()
root = tk.Tk()
root.title("Auto-Chord-Tracker")

frame1 = tk.LabelFrame(root, text="コード検出", foreground="green")
frame1.grid(row=0, sticky="we")

frame2 = tk.LabelFrame(root, text="WFD出力", foreground="green")
frame2.grid(row=1, sticky="we")

frame3 = tk.LabelFrame(root, text="進捗", foreground="green")
frame3.grid(row=2, sticky="we")

entry_audio_path = tk.Entry(frame1, width=40)
entry_audio_path.insert(tk.END, "パスを入力してください")
entry_audio_path.grid(row=0, column=0, padx=5, pady=15)

button_audio_select = tk.Button(frame1, text="選択&解析", width=10, command=callback)
button_audio_select.grid(row=0, column=1)

entry_wfd_path = tk.Entry(frame2, width=40)
entry_wfd_path.insert(tk.END, "パスを入力してください")
entry_wfd_path.grid(row=0, column=0, padx=5, pady=15)

button_wfd_select = tk.Button(frame2, text="選択&出力", width=10, command=savewfd)
button_wfd_select.grid(row=0, column=1)

pb1 = ttk.Progressbar(frame3, orient=tk.HORIZONTAL, length=350, mode='indeterminate')
pb1.configure(maximum=50, value=0)
pb1.grid(row=0, column=0)

root.mainloop()
