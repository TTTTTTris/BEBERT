## BEBERT: Efficient and robust binary ensemble BERT

Created by [jiayi Tian](https://ttttttris.github.io/), [Chao Fang](https://0-scholar-google-com.brum.beds.ac.uk/citations?hl=zh-CN&user=3wg-QTgAAAAJ), [Zhongfeng Wang](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=faC-qekAAAAJ&hl=zh-CN) from Nanjing University and [Haonan Wang](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=AQuXJEkAAAAJ&hl=zh-CN)the University of Southern California.

![model architecture](./assets/BEBERT_fig1.png)

## Introduction

This project is an implementation of our submitted ICASSP 2023 paper *BEBERT: Efficient and robust binary ensemble BERT* [[PDF](http://arxiv.org/abs/2210.15976)]. Pre-trained BERT models have achieved impressive accuracy on natural language processing (NLP) tasks, while the 
excessive amount of parameters hinders them from efficient deployment on edge devices. Binarization of the BERT models can significantly alleviate these issues but come with a severe accuracy drop compared with their full-precision counterparts. In this paper, to the best of our knowledge, we make the first attempt to employ ensemble learning on binary BERT models to improve accuracy and retain computational efficiency, yielding a superior model named Binary Ensemble BERT (BEBERT). Furthermore, we propose an efficient scheme to speed up the training process without compromising accuracy by removing the knowledge distillation procedures during ensemble. Experimental results on the GLUE benchmark show the proposed BEBERT significantly outperforms the state-of-the-art binary BERT models in both accuracy and robustness with a 2× speedup during the training process. It also exceeds the existing compressed BERTs in accuracy, and saves 15× and 13× in FLOPs and model size, respectively, over the full-precision BERT baseline.

## Dependencies
```bash
pip install -r requirements.txt
```

## Datasets

We train and test BinaryBERT on [GLUE benchmarks](https://github.com/nyu-mll/GLUE-baselines).
For data augmentation on GLUE, please follow the instruction in [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

## Execution

Our experiments are based on the fine-tuned full-precision DynaBERT, which can be found [here](https://drive.google.com/file/d/1pYApaDcse5QIB6lZagWO0uElAavFazpA/view?usp=sharing). Complete running scripts and more detailed tips are provided in `./scripts`. Note that There are two steps for execution in ensemble BinaryBERT, and the first step does not need ensemble. And before you ensemble the BinaryBERT in the second step, please train a half-sized ternary BERT first.

### Step one: Train a half-sized ternary BERT
This correponds to `./binarybert/scripts/ternary_glue.sh`.

### Step two: Apply TWS and BinaryBERT ensemble
This correponds to `./binarybert/scripts/benn_glue_{A,B,C}.sh`. 

## Acknowledgement

The original code is borrowed from [BinaryBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BinaryBERT) and [BiBERT](https://github.com/htqin/BiBERT).

## Citation

If you find our work useful in your research, please consider citing:

```shell
@misc{https://doi.org/10.48550/arxiv.2210.15976,
  doi = {10.48550/ARXIV.2210.15976},
  url = {https://arxiv.org/abs/2210.15976},
  author = {Tian, Jiayi and Fang, Chao and Wang, Haonan and Wang, Zhongfeng},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {BEBERT: Efficient and robust binary ensemble BERT},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
