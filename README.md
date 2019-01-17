# Context_aware_NMT
An explanation of this repository written in English is here.

## 概要
文脈つきニューラル機械翻訳（Context-aware NMT）の実装です。  
Bawden et al.（NAACL2018）の手法に少し改変を加えたものになっています。  
これを用いた研究については、言語処理学会第25回年次大会で発表しました。  

## 動作環境


## 使い方
事前学習を行う場合は、以下の手順で
1. python pretrain.py train config_file
2. python pretrain.py dev config_file

事前学習を行ったら、文脈つきNMTの学習をします。

1. python cnmt.py train config_file
2. python cnmt.py dev config_file
3. python cnmt.py test config_file test_epoch


## おことわり
このプログラムは、大半を[YukioNMT](https://github.com/yukio326/nmt-chainer)をもとに作成しました。  
特に、`pretrain.py`はほぼこちらの`nmt.py`と同じものです。  
ありがとうございました。