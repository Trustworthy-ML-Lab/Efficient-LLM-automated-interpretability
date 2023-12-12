import os
import asyncio

os.environ["OPENAI_API_KEY"] = open(os.path.join(os.path.expanduser("~"), ".openai_api_key"), "r").read()[:-1]

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer, SummaryExplainer, HighlightExplainer, HighlightSummaryExplainer, AVHSExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  

import openai

async def get_explanation(mode = "Summary", neuron_record = None, 
                          to_print = False, explainer_model = "gpt-3.5-turbo", get_token_only = False):
    
    cutoff = neuron_record.quantile_boundaries[2]
    slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
    train_activation_records = neuron_record.train_activation_records(
      activation_record_slice_params=slice_params
    )
    valid_activation_records = neuron_record.valid_activation_records(
      activation_record_slice_params=slice_params
    )
    
    if mode=="Original":
        explainer = TokenActivationPairExplainer(
            model_name=explainer_model,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
            get_token_only = get_token_only,
            to_print = to_print,
        )
        
    elif mode=="Summary":
        explainer = SummaryExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=train_activation_records,
          cutoff=cutoff,
          num_samples=1,
          get_token_only = get_token_only,
          to_print = to_print
        )
    
    elif mode=="Highlight":
        explainer = HighlightExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=train_activation_records,
          cutoff=cutoff,
          num_samples=1,
          get_token_only = get_token_only,
          to_print = to_print,
        )
    
    elif mode=="HighlightSummary":
        explainer = HighlightSummaryExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=train_activation_records,
          cutoff=cutoff,
          num_samples=1,
          get_token_only = get_token_only,
          to_print = to_print,
        )

    elif mode=="AVHS":
        explainer = AVHSExplainer(
            model_name=explainer_model,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
            get_token_only = get_token_only,
            to_print = to_print
        )
        
    assert len(explanations) == 1
    explanation = explanations[0]
    return explanation

async def get_puzzle_explanation(mode = "Summary", explainer_model = "gpt-3.5-turbo", puzzle_activation_record = None):
    if mode=="Original":
        explainer = TokenActivationPairExplainer(
            model_name=explainer_model,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=puzzle_activation_record,
            max_activation=calculate_max_activation(puzzle_activation_record),
            num_samples=1,
        )
        
    elif mode=="Summary":
        explainer = SummaryExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=puzzle_activation_record,
          cutoff=1,
          num_samples=1,
        )
    
    elif mode=="Highlight":
        explainer = HighlightExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=puzzle_activation_record,
          cutoff=1,
          num_samples=1,
        )
    
    elif mode=="HighlightSummary":
        explainer = HighlightSummaryExplainer(
          model_name=explainer_model,
          prompt_format=PromptFormat.HARMONY_V4,
          max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
          all_activation_records=puzzle_activation_record,
          cutoff=1,
          num_samples=1,
        )

    elif mode=="AVHS":
        explainer = AVHSExplainer(
            model_name=explainer_model,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=1,
        )
        
        explanations = await explainer.generate_explanations(
            all_activation_records=puzzle_activation_record,
            max_activation=calculate_max_activation(puzzle_activation_record),
            num_samples=1,
        )
        
    assert len(explanations) == 1
    explanation = explanations[0]
    return explanation

async def get_score(explanation = None, layer = 0, neuron = 1, simulator_model = "gpt-3.5-turbo-instruct"):
  SIMULATOR_MODEL_NAME = simulator_model
  # Load a neuron record.
  neuron_record = load_neuron(layer, neuron)
  cutoff = neuron_record.quantile_boundaries[2]
  # Grab the activation records we'll need.
  slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
  train_activation_records = neuron_record.train_activation_records(
      activation_record_slice_params=slice_params
  )
  valid_activation_records = neuron_record.valid_activation_records(
      activation_record_slice_params=slice_params
  )

  simulator = UncalibratedNeuronSimulator(
      ExplanationNeuronSimulator(
          SIMULATOR_MODEL_NAME,
          explanation,
          max_concurrent=1,
          prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
      )
  )
  scored_simulation = await simulate_and_score(simulator, valid_activation_records)
  return float(f"{scored_simulation.get_preferred_score():.2f}")


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)
