# Anime Cutout Lab

動画内のキャラクターを囲み、ブラウザ内で背景を抜いて WebM に書き出す静的 Web アプリです。動画はサーバーへアップロードされません。

## 公開

GitHub Pages の Source を `Deploy from a branch`、Branch を `main / (root)` に設定してください。

## ローカル起動

```powershell
python -m http.server 4173
```

`http://localhost:4173` を開きます。

## 現在の方式と制約

このバージョンは最初のフレームをブラシで手動マスクし、Cutie Video Object Segmentationで後続フレームへ伝播します。480×272版の4つのONNXモデルをONNX Runtime Webで実行します。初回利用時は合計約148MBのモデルデータを取得します。

Chrome / EdgeのWebGPU対応環境を推奨します。WebGPUでモデルを初期化できない場合はWASMへフォールバックしますが、処理は大幅に遅くなります。

利用者が権利を持つ、または利用許諾を得た動画のみ使用してください。
