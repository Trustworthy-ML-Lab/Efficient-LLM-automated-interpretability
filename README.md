# The Importance of Prompt Tuning for Automated Neuron Explanations

This is the official repository containing the code for our paper The Importance of Prompt Tuning for Automated Neuron Explanations (NeurIPS 2023 ATTRIB). [arXiv](https://arxiv.org/abs/2310.06200), [Project website](https://lilywenglab.github.io/Efficient-LLM-automated-interpretability/). We build heavily on OpenAIs [automated-interpretability](https://github.com/openai/automated-interpretability) and specifically analyze the importance of specific prompt types used to generate new explanations. 

We find that simpler, more intuitive prompts such as our summary can increase both computational efficiency and quality of the generated explanations. In particular our simple **Summary** prompt shown below can outperform the original while requiring 2.4 times less input tokens per neuron.

<img src=figs/prompt_comparison.png alt="Overview" width=850 height=266>

## Quickstart

1. Setup: first `cd neuron-explainer` and then install required packages by running `pip install -e .`
2. Change line 4 of `neuron_explainer/utils.py` to correspond to your OpenAI API key
3. Run `Experiments/save_descriptions.ipynb` to generate explanations for 5 sample neurons using our different prompts.
4. Compare results with neuron activations shown in [NeuronViewer](https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html)
5. To explain a different set of neurons, input a different csv to neurons_to_evaluate that similar to `inputs/test_neurons.csv` has the `layer` and `neuron` columns.

## Reproducing Our Experiments

1. All neuron descriptions generated for the paper are available in `Experiments\results`.
2. Simulate and Score experiments can be reproduced by running `Experiments\simulate_score.ipynb`. Note: the simulator model used originally, 'gpt-3.5-turbo-instruct', is no longer supported by the API in the required format and has been replaced with 'text-davinci-003' in the code. This change will likely decrease simulation quality and increase api costs.
3. To calculate Similarity to baseline explanation(AdaCS) to evaluate explanation quality, run `Experiments\ada_cs.ipynb`.
4. To explan Neuron Puzzles and calculate AdaCS similarity to their ground truth explanations, run `Experiments\puzzles.ipynb`
5. Finally our comparison of number of API tokens per explained neuron can be reproduced in `Experiments\simulate_score.ipynb`.

`Experiments\save_neuron_info.ipynb` and `Experiments\get_interpretable_neurons.ipynb` are used to collect NeuronViewer explanations and select which neurons we should explain.

## Overview of Neuron Explanation Pipeline

<img src=figs/animation.gif alt="Overview" width=850 height=478>

## Cite us

If you find this code useful, please cite:
```
@misc{lee2023importance,
      title={The Importance of Prompt Tuning for Automated Neuron Explanations}, 
      author={Justin Lee and Tuomas Oikarinen and Arjun Chatha and Keng-Chi Chang and Yilan Chen and Tsui-Wei Weng},
      year={2023},
      eprint={2310.06200},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
