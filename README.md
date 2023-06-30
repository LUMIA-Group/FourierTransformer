# FourierTransformer
This is the official Pytorch implementation of paper [Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator](https://arxiv.org/abs/2305.15099)


## Install
* Pytorch version >= 1.13.0
* Fairseq version >= 0.12.3
```bash
git clone https://github.com/LUMIA-Group/FourierTransformer.git
cd FourierTransformer
pip install -e .
```
For faster training, install NVIDIA's apex library following [fairseq](https://github.com/facebookresearch/fairseq#requirements-and-installation).


## Experiments
```
# Download files for preprocessing

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

### Further Pretraining on Pile
1. Download the Pile dataset from [here](https://pile.eleuther.ai/).
2. Preprocess the Pile:
   ```
   # BPE
   for SPLIT in train val test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs pile/${SPLIT}.raw.en \
        --outputs pile/${SPLIT}.bpe.en \
        --keep-empty \
        --workers 120; \
   done

   # Binarize
   fairseq-preprocess \
    --only-source \
    --source-lang "en" \
    --srcdict dict.txt \
    --trainpref pile/train.bpe \
    --validpref pile/val.bpe \
    --testpref pile/test.bpe \
    --destdir pile-bin \
    --workers 60

   rename files in pile-bin by removing ".en".
   ```
3. [Script](https://github.com/LUMIA-Group/FourierTransformer/blob/main/Summarization/submits/futher_pretrain.sh) to further pretrain Fourier-Bart       (in our paper, we randomly sliced 10G data from the Pile to conduct further pretraining).

### CNN-Dailymail
1. Download, Preprocess and Binarize: 
  Follow [this script](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md).

2. Fine-tuning Fourier Transformer on CNN-DM summarization task:
   ```bash
   cd Summarization
   sh submits/cnn-dm.sh
   ```

3. Evaluate:  
   For calculating rouge, install files2rouge from [here](https://github.com/pltrdy/files2rouge).
   ```
   sh submits/eval-cnn-dm.sh
   ```

### ELIT5


### LRA
As mentioned in our paper, the code for LRA is build from [this repository](https://github.com/mlpen/Nystromformer/tree/main/LRA). Please follow the scripts there to prepare the datasets.

To run LRA experiments,
```
cd LRA/code
sh run_tasks.sh
```
Feel free to play with different settings by modifying [lra_config.py](https://github.com/LUMIA-Group/FourierTransformer/blob/main/LRA/code/lra_config.py)
