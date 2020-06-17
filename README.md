# Tencent-Ads-Algo-Comp-2020

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) 

Git repo for Tencent Advertisement Algorithm Competition 

* [Quick Start](#quick_start)
* [Script Documentation](#script_documentation)
  * [Model Training V2](#model_training2)
  * [Legacy - Model Training V1](#model_training1)
  * [Data Preparation](#data_preparation)
* [Materials](#material)

---

<a id='quick_start'></a>
## Quick Start

```bat
cd ./Script
. prerequisite.sh
python3 input_generate.py
python3 input_split.py fine
python3 train_w2v.py creative 128
python3 train_w2v.py ad 128
python3 train_w2v.py advertiser 128
python3 train_w2v.py product 128
python3 train_w2v.py industry 64
python3 train_w2v.py product_category 64
python3 train_v2_age_final_pre_ln_tf_multiInp.py 40 2048 100 1e-3
```

<a id='script_documentation'></a>
## Script Documentation

<a id='model_training2'></a>
### Model Training V2

* How to run training script

  Syntax: `python3 train_v2_{some script name}.py 40 2048 100 1e-3`
  > **Argument**:
  > 1. *(Required,INT)* target epoch to train
  > 2. *(Required,INT)* batch size for training
  > 3. *(Required,INT)* maximal length of input sequence, smaller length can help train withb larger batch size
  > 4. *(Required,FLOAT)* learning rate for adam optimizer
  > 5. *(Optional, INT)* If nothing specified then the model will be trained from scratch, otherwise it indicates the epoch to resume training
  > 6. *(Optional, INT)* If nothing specified then the model will be trained from scratch, otherwise it indicates the training file to resume training
  >    * Example: `9, 2` indicates resume training from epoch 9 file 2.
  
* Training script inventory
  ```
  |--Script
    |--data_loader_v2.py
    |
    |--clf_lstm.py             # Model based on stacked LSTM
    |--clf_gnmt.py             # Model based on GNMT (Google Neural Translation Machine)
    |--clf_tf_enc.py           # Model based on Encoder part of Transformer
    |--clf_esim.py             # Model based on ESIM (Enhanced Sequential Inference Model)
    |--clf_pre_ln_tf.py        # Model based on pre Layer Normalization Transformer
    |--clf_final.py            # Model for final submission
  ```

<a id='model_training1'></a>
### Legacy - Model Training V1

* How to run training script

  Syntax: `python3 train_{some script name}.py 0 10 256 100 1e-3 split`
  > **Argument**:
  > 1. *(Required,INT)* 0 means training from scratch and a positive number means loading the corresponding epoch and start training from there <br/>
  > 2. *(Required,INT)* number of epoches to train
  > 3. *(Required,INT)* batch size for training
  > 4. *(Required,INT)* maximal length of input sequence, smaller length can help train withb larger batch size
  > 5. *(Required,FLOAT)* learning rate for adam optimizer
  > 6. *(Optional)* If nothing specified then the model will be trained using unsplitted files. If `python3 input_split.py fine` has been executed and a value is specified the model will be trained using a list of splitted files. 
  
* Training script inventory
  ```
  |--Script
    |--data_loader.py
    |
    |--multi_seq_lstm_classifier.py
    |--train_age_multi_seq_lstm_classifier.py
    |--train_gender_multi_seq_lstm_classifier.py
    |
    |--transformer_encoder_classifier.py
    |--train_age_transformer_encoder_classifier_with_creative.py
    |
    |--GNMT_classifier.py
    |--train_age_GNMT_classifier_with_creative.py
    |
    |--multi_seq_GNMT_classifier.py
    |--train_age_multi_seq_GNMT_classifier.py
  ```
  
<a id='data_preparation'></a>
### Data Preparation

* **Step 1**: run 
```bat
cd ./Script
. prerequisite.sh
``` 

Note that if the instance has no public internet connection, download [train file](https://tesla-ap-shanghai-1256322946.cos.ap-shanghai.myqcloud.com/cephfs/tesla_common/deeplearning/dataset/algo_contest/train_preliminary.zip) and [test file](https://tesla-ap-shanghai-1256322946.cos.ap-shanghai.myqcloud.com/cephfs/tesla_common/deeplearning/dataset/algo_contest/test.zip) and put them under `/Script`. You should have the following files and directories after execution.

```
|--Script
  |--train_artifact
    |--user.csv
    |--click_log.csv
    |--ad.csv
  |--test_artifact
    |--click_log.csv
    |--ad.csv
  |--input_artifact
  |--embed_artifact
  |--model_artifact
  |--output_artifact
```

* **Step 2**: run 
```bat
python3 input_generate.py
python3 input_split.py
```

For machine with small memory please replace the second line with `python3 input_split.py fine`.You should have the following files after execution.

```
|--Script
  |--input_artifact
    |--train_idx_shuffle.npy
    |--train_age.npy
    |--train_gender.npy
    |--train_creative_id_seq.pkl
    |--train_ad_id_seq.pkl
    |--train_advertiser_id_seq.pkl
    |--train_product_id_seq.pkl
    |--test_idx_shuffle.npy
    |--test_creative_id_seq.pkl
    |--test_ad_id_seq.pkl
    |--test_advertiser_id_seq.pkl
    |--test_product_id_seq.pkl
  |--embed_artifact
    |--embed_train_creative_id_seq.pkl
    |--embed_train_ad_id_seq.pkl
    |--embed_train_advertiser_id_seq.pkl
    |--embed_train_product_id_seq.pkl
  |--model_artifact
  |--output_artifact
  |--train_artifact
  |--test_artifact
```

* **Step 3**: run 
```bat
python3 train_w2v.py creative 128
python3 train_w2v.py ad 128
python3 train_w2v.py advertiser 128
python3 train_w2v.py product 128
python3 train_w2v.py industry 64
python3 train_w2v.py product_category 64
```

You should have the following files after exection.

```
|--Script
  |--embed_artifact
    |--w2v_registry.json
    |--wv_registry.json
    |--creative_sg_embed_s256_{random token}
    |--...
  |--model_artifact
  |--input_artifact
  |--output_artifact
  |--train_artifact
  |--test_artifact
```

Note that `w2v_registry.json` stores all the w2v model artifact paths and `wv_registry.json` stores all the `KeyedVector` artifact paths.

<a id='material'></a>
## Materials

* 官方竞赛手册: [PDF](https://algo-1256087447.cos.ap-nanjing.myqcloud.com/admin/20200509/7da104bd074309285ab56a6e52150ba3.pdf)

* 赛题理解与思路: [思路byCHIZHU](https://mp.weixin.qq.com/s/ISQjOGcc_spSNVeeg75d8w), [分析by鱼遇](https://zhuanlan.zhihu.com/p/141288029), [思路by鱼遇](https://zhuanlan.zhihu.com/p/143185271)

* 往届回顾: [17~19](https://zhuanlan.zhihu.com/p/116907937)


