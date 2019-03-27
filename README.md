# Context-aware NMT

## 概要
マルチエンコーダ型の文脈つきニューラル機械翻訳（Context-aware NMT）の実装です。  
[Bawden+, NAACL2018] の手法に少し改変を加えたものになっています。  
これを用いた研究については、言語処理学会第25回年次大会で発表しました。[[Paper](http://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/P1-23.pdf)]  

## 動作環境
Python 3.5.1 以上   
必要なものは以下の通りです。
- Chainer 3.3.0
- Cupy 2.3.0
- gensim 3.3.0  

`pip install -r requirements.txt` で入ると思いますが、入らないようでしたらご連絡ください。

## 使い方

#### Step1
対訳コーパスのファイルの前処理をします。  
文書の境界ごとに、以下のように空行を挟んでください。  
このとき、言語対ごとに文書数（=空行数）が同じになるよう注意してください。

```
我輩 は 猫 で ある 。
（中略）
ありがたい ありがたい 。

メロス は 激怒 し た 。
（中略）  
勇者 は 、 ひどく 赤面 し た 。
```

#### Step2
[Luong+, EMNLP2015] と共通部分について事前学習します。  
事前学習しない場合は何もせずStep3へ進みます。  
config_fileの書き方は[config_readme](https://github.com/hargon24/Context_aware_NMT/blob/master/config_readme.md)や[sample.config](https://github.com/hargon24/Context_aware_NMT/blob/master/sample.config)を参考にしてください。  
```
python pretrain.py train config_file [restart_epoch]  
python pretrain.py dev config_file [start_epoch [end_epoch] ]
```
devのBLEUが最も良いEpochを選び、その番号をconfig_fileに書き込みます。  
testは任意で実行してください。
```
python pretrain.py test config_file test_epoch
```

#### Step3
文脈つきNMTの学習を実行します。  

```
python cnmt.py train config_file [restart_epoch] 
python cnmt.py dev config_file [start_epoch [end_epoch] ] 
python cnmt.py test config_file test_epoch  
```

## おことわり
このプログラムは[YukioNMT](https://github.com/yukio326/nmt-chainer)をもとに作成しました。  
特に`pretrain.py`に関しては、ほぼこちらの`nmt.py`と同じものです。  

## その他
このコードは自由に使っていただいて構いませんが、対外発表等にお使いの場合はこのページのURLを記載いただけますようお願いいたします。
