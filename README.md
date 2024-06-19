# GW-MoE

The official implementation of the paper "GW-MoE: Resolving Uncertainty in MoE Router with Global Workspace Theory"[![arxiv](https://img.shields.io/badge/Arxiv-2406.12375-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.12375).

## Installation

To install the required environment, execute `pip install -r requirements.txt`.

## Usage

`transformers` is the code we used to conduct experiments on Switch Transformer.

You can choose a task from ./tasks and run the corresponding bash script to conduct experiments. Before executing the following commands, make sure to set the dataset and model names or paths in the corresponding script.

For text classification:

```bash
# Run GWMoE on Switch Transformer for text classification:

bash ./tasks/text-classification/run_glue.sh
```

For summarization:

```bash
# Run GWMoE on Switch Transformer for summarization:

bash ./tasks/summarization/run_summarization.sh
```

For question-answering:

```bash
# Run GWMoE on Switch Transformer for question-answering:

bash switch_transformer/tasks/question-answering/run_seq2seq_qa.sh
```

## Citation

If you find our work helpful, please cite our paper:

```
@misc{wu2024gwmoe,
      title={GW-MoE: Resolving Uncertainty in MoE Router with Global Workspace Theory}, 
      author={Haoze Wu and Zihan Qiu and Zili Wang and Hang Zhao and Jie Fu},
      year={2024},
      eprint={2406.12375},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```





