# Context-aware NMT

## 概要
文脈つきニューラル機械翻訳（Context-aware NMT）の実装です。  
Bawden et al.（NAACL2018）の手法に少し改変を加えたものになっています。  
これを用いた研究については、言語処理学会第25回年次大会で発表しました。  

## 動作環境
Python 3.5.1 <    
必要なものは以下の通りです。
- Chainer 3.3.0
- Cupy 2.3.0
- gensim 3.3.0  

動かなければ `pip install -r requirements.txt` をお試しください。

## 使い方

#### Step1
文書の境界ごとに、以下のように空行を挟んでください。  

```:train.ja
我輩 は 猫 で ある 。
（中略）
ありがたい ありがたい 。

メロス は 激怒 し た 。
（中略）  
勇者 は 、 ひどく 赤面 し た 。
```

#### Step2
[Luong+, EMNLP2015] と共通部分について事前学習します。  
config_fileの書き方は[readme_config](https://github.com/hargon24/Context_aware_NMT/blob/master/config_README.md)を参考にしてください。

```
python pretrain.py train config_file [restart_epoch]  
python pretrain.py dev config_file [start_epoch] [end_epoch]
```
DevのBLEUが最も良いEpochを選び、その番号をconfig_fileに書き込みます。  
Testは任意で実行してください。
```
python pretrain.py test config_file test_epoch
```

#### Step3
文脈つきNMTの学習を実行


```
python cnmt.py train config_file [restart_epoch] 
python cnmt.py dev config_file [start_epoch] [end_epoch] 
python cnmt.py test config_file test_epoch  
```

## おことわり
このプログラムの大半は[YukioNMT](https://github.com/yukio326/nmt-chainer)をもとに作成しました。  
特に、`pretrain.py`はほぼこちらの`nmt.py`と同じものです。  
ありがとうございました。