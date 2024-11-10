# PEAR

## Environment

- pytorch
- transformers
- ...

## Quick Start

1. Download the pre-trained model - [chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm) and place it in `./pretrain`.

2. (optional) Adjust parameters in `run.py` and `script_run.sh`, especially the *output_dir*

3. Run
``` shell
bash script_run.sh
```

4. Check results on screen or learn details in the *output_dir*.

# CMIM23-NOM1-RA (dataset)

- It is the first high-quality restricted domain relational triple extraction dataset in the network operation and maintenance field.
- The dataset is placed in `./dataset/CMIM23-NOM1-RA`.

# Guidance for Baseline Reproduction

- We provide reproduction codes or reproduction guidelines for each baseline in (https://github.com/JYzzzzzz/PEAR-RTE-baselines). 

- We add explanation in the `README.md` file in the root path of each baseline. The explanation generally cover: 
	- (0) The original instructions in the source code README.
	- (1) What parts of the source code were modified.
	- (2) The training and validation procedures.
	- (3) The URL of the source code.


# Cite

... to be supplemented
