# Tencent-Ads-Algo-Comp-2020

Git repo for Tencent Advertisement Algorithm Competition 

## Script

### Model Training Utility

```
|--Script
  |--data_loader.py
  |--multi_seq_lstm_classifier.py
  |--train_age_multi_seq_lstm_classifier.py
```

* Multi-Seq-LSTM Classifier for Age Prediction

  Run `python3 train_age_multi_seq_lstm_classifier.py 0 10 1024 100 split`
  > **Argument**:
  > 1. *(Required,INT)* 0 means training from scratch and a positive number means loading the corresponding epoch and start training from there <br/>
  > 2. *(Required,INT)* number of epoches to train
  > 3. *(Required,INT)* batch size for training
  > 4. *(Required,INT)* maximal length of input sequence, smaller length can help train withb larger batch size
  > 5. *(Optional)* If nothing specified then the model will be trained using unsplitted files. If `python3 input_split.py fine` has been running and a value is specified the model will be trained using a list of splitted files. 




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
python3 train_w2v.py creative 256
python3 train_w2v.py ad 256
python3 train_w2v.py advertiser 128
python3 train_w2v.py product 128
```

You should have the following files after exection.

```
|--Script
  |--embed_artifact
    |--w2v_registry.json
    |--creative_embed_s256_{random token}
    |--ad_embed_s256_{random token}
    |--advertiser_embed_s128_{random token}
    |--product_embed_s128_{random token}
    |--embed_train_creative_id_seq.pkl
    |--embed_train_ad_id_seq.pkl
    |--embed_train_advertiser_id_seq.pkl
    |--embed_train_product_id_seq.pkl
  |--model_artifact
  |--input_artifact
  |--output_artifact
  |--train_artifact
  |--test_artifact
```

Note that `w2v_registry.json` stores all the w2v model artifact paths.


## Materials

* 官方竞赛手册: [PDF](https://algo-1256087447.cos.ap-nanjing.myqcloud.com/admin/20200509/7da104bd074309285ab56a6e52150ba3.pdf)

* 赛题理解与思路: [思路byCHIZHU](https://mp.weixin.qq.com/s/ISQjOGcc_spSNVeeg75d8w), [分析by鱼遇](https://zhuanlan.zhihu.com/p/141288029), [思路byby鱼遇](https://zhuanlan.zhihu.com/p/143185271), [Baseline 1](https://zhuanlan.zhihu.com/p/141842643), [Baseline 2](https://zhuanlan.zhihu.com/p/139270681), [Baseline 3](https://zhuanlan.zhihu.com/p/144346714)

* 往届回顾: [17~19](https://zhuanlan.zhihu.com/p/116907937)


