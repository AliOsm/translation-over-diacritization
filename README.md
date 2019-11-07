# Translation-over-Diacritization

This repository contains the implementation for the Translation-over-Diacritization technique descriped in our paper on Arabic Text Diacritization:

"[Neural Arabic Text Diacritization: State of the Art Results and a Novel Approach for Machine Translation](https://www.aclweb.org/anthology/D19-5229)", Ali Fadel, Ibraheem Tuffaha, Bara' Al-Jawarneh and Mahmoud Al-Ayyoub, [EMNLP-IJCNLP 2019](https://www.emnlp-ijcnlp2019.org).

The work uses diacritics to improve the results of Arabic->English translation while avoiding vocabulary sparsity that leads to out-of-vocabulary issues.

## 0. Prerequisites
- Tested with Python 3.6.8
- Install required packages listed in `requirements.txt` file
    - `pip install -r requirements.txt`
- Download and unzip the Arabic-English parallel corpora from [OPUS](http://opus.nlpl.eu) project and extract them in `data_dir/tmx` folder
  - `WEBSITE_LINK=https://object.pouta.csc.fi`
  - `wget "$WEBSITE_LINK"/OPUS-GlobalVoices/v2017q3/tmx/ar-en.tmx.gz -O GlobalVoices_v2017q3.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-MultiUN/v1/tmx/ar-en.tmx.gz -O MultiUN_v1.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-News-Commentary/v11/tmx/ar-en.tmx.gz -O News-Commentary_v11.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-Tatoeba/v2/tmx/ar-en.tmx.gz -O Tatoeba_v2.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-TED2013/v1.1/tmx/ar-en.tmx.gz -O TED2013_v1.1.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-Ubuntu/v14.10/tmx/ar-en.tmx.gz -O Ubuntu_v14.10.tmx.gz`
  - `wget "$WEBSITE_LINK"/OPUS-Wikipedia/v1.0/tmx/ar-en.tmx.gz -O Wikipedia_v1.0.tmx.gz`
  - `mv *.gz data_dir/tmx`
  - `gunzip data_dir/tmx/*.gz`
- Clone both [Shakkelha](https://github.com/AliOsm/shakkelha) and [mosesdecoder](https://github.com/moses-smt/mosesdecoder) dependency repositories
  - `git clone https://github.com/AliOsm/shakkelha.git`
  - `git clone https://github.com/moses-smt/mosesdecoder.git`

## 1. Data Extraction
To extract the data, run the following command:
```
python 1_extract_data.py
```

## 2. Data Preparing and Splitting
To prepare, segment (using Byte Pair Encoding), and split the data into training and testing, run the following command:
```
sh 2_prepare_data.sh
```

## 3. Remove Long Lines
Some lines gain a lot of tokens after the segmentation process in step 2, so run the following command to remove them:
```
python 3_remove_long_lines.py
```

## 4. Diacritize Arabic Data
To diacritize the Arabic data extracted in step 1, run the following command:
```
sh 4_diacritize_ar_data.sh
```

## 5. Merge Diacritics with Segmented Text
To merge the diacritics from the diacritized Arabic text generated from step 4 with the segmented Arabic text from step 2, run the following command:
```
python 5_merge_diacritics_with_bpe.py
```

## 6. Train the Model
To train the model, run the following command:
```
python 6_seq2seq.py --use-diacs True
python 6_seq2seq.py --use-diacs False
```
The value of the boolean parameter `USE_DIACS` determines whether to train the model with or without diacritics.

## 7. Detokenize Predicted Translations (Remove BPE special characters)
To detokenize the predicted translations, run the following command:
```
sh 7_detok_predictions.sh
```

## 8. Calculate BLEU scores
To calculate the BLEU scores, run the following command:
```
sh 8_calculate_bleu.sh
```

## Model Structure

The following figure illustrates our model structure
<p align="center">
  <img src="model_representation.png">
</p>

#### Note: All codes in this repository tested on [Ubuntu 18.04](http://releases.ubuntu.com/18.04)

## Contributors
1. [Ali Hamdi Ali Fadel](https://github.com/AliOsm).<br/>
2. [Ibraheem Tuffaha](https://github.com/IbraheemTuffaha).<br/>
3. [Mahmoud Al-Ayyoub](https://github.com/malayyoub).<br/>

## License
The project is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
