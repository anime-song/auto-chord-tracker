和音自動認識を高精度で可能なモデル。
学習データには日本のポップソング約70曲を採譜し、使用しています。

# 動作するために必要なもの
- ffmpeg [https://www.ffmpeg.org/download.html] (mp3を読み込むために必要です。)

# 出力形式

出力されたラベルはAudacityで視覚化できます。

    開始時間  終了時間  和音名
                .
                .
                .

## wavetoneへの出力

解析したい音声ファイルを任意の設定で、wavetoneで解析します。
オフセット、拍子、テンポを合わせ、最後にコード解析を実行して保存します。

# ダウンロード

[Release](https://github.com/anime-song/auto-chord-tracker/releases)よりダウンロードできます。