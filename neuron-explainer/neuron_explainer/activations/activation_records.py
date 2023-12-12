"""Utilities for formatting activation records into prompts."""

import math
from typing import Optional, Sequence

from neuron_explainer.activations.activations import ActivationRecord

import string
ALPHABET = list(string.ascii_uppercase)

UNKNOWN_ACTIVATION_STRING = "unknown"

def relu(x: float) -> float:
    return max(0.0, x)


def calculate_max_activation(activation_records: Sequence[ActivationRecord]) -> float:
    """Return the maximum activation value of the neuron across all the activation records."""
    flattened = [
        # Relu is used to assume any values less than 0 are indicating the neuron is in the resting
        # state. This is a simplifying assumption that works with relu/gelu.
        max(relu(x) for x in activation_record.activations)
        for activation_record in activation_records
    ]
    return max(flattened)


def normalize_activations(activation_record: list[float], max_activation: float) -> list[int]:
    """Convert raw neuron activations to integers on the range [0, 10]."""
    if max_activation <= 0:
        return [0 for x in activation_record]
    # Relu is used to assume any values less than 0 are indicating the neuron is in the resting
    # state. This is a simplifying assumption that works with relu/gelu.
    return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record]


def _format_activation_record(
    activation_record: ActivationRecord,
    max_activation: float,
    omit_zeros: bool,
    hide_activations: bool = False,
    start_index: int = 0,
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    tokens = activation_record.tokens
    normalized_activations = normalize_activations(activation_record.activations, max_activation)
    if omit_zeros:
        assert (not hide_activations) and start_index == 0, "Can't hide activations and omit zeros"
        tokens = [
            token for token, activation in zip(tokens, normalized_activations) if activation > 0
        ]
        normalized_activations = [x for x in normalized_activations if x > 0]

    entries = []
    assert len(tokens) == len(normalized_activations)
    for index, token, activation in zip(range(len(tokens)), tokens, normalized_activations):
        activation_string = str(int(activation))
        if hide_activations or index < start_index:
            activation_string = UNKNOWN_ACTIVATION_STRING
        entries.append(f"{token}\t{activation_string}")
    return "\n".join(entries)


def format_activation_records(
    activation_records: Sequence[ActivationRecord],
    max_activation: float,
    *,
    omit_zeros: bool = False,
    start_indices: Optional[list[int]] = None,
    hide_activations: bool = False,
) -> str:
    """Format a list of activation records into a string."""
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [
                _format_activation_record(
                    activation_record,
                    max_activation,
                    omit_zeros=omit_zeros,
                    hide_activations=hide_activations,
                    start_index=0 if start_indices is None else start_indices[i],
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
        + "\n<end>\n"
    )


def _format_tokens_for_simulation(tokens: Sequence[str]) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation, suitable
    for use in prompts.
    """
    entries = []
    for token in tokens:
        entries.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")
    return "\n".join(entries)


def format_sequences_for_simulation(
    all_tokens: Sequence[Sequence[str]],
) -> str:
    """
    Format a list of lists of tokens into a string with each token marked as having an "unknown"
    activation, suitable for use in prompts.
    """
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [_format_tokens_for_simulation(tokens) for tokens in all_tokens]
        )
        + "\n<end>\n"
    )


def non_zero_activation_proportion(
    activation_records: Sequence[ActivationRecord], max_activation: float
) -> float:
    """Return the proportion of activation values that aren't zero."""
    total_activations_count = sum(
        [len(activation_record.activations) for activation_record in activation_records]
    )
    normalized_activations = [
        normalize_activations(activation_record.activations, max_activation)
        for activation_record in activation_records
    ]
    non_zero_activations_count = sum(
        [len([x for x in activations if x != 0]) for activations in normalized_activations]
    )
    return non_zero_activations_count / total_activations_count


def summarize_activation_records(
    activation_records: Sequence[ActivationRecord],
    cutoff: float
) -> str:
    """Format a list of activation records into a string."""
    return ("\n"+"".join(
                [
                _summarize_activation_record(
                    activation_record,
                    cutoff = cutoff,
                    index = i,
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
    )

def _summarize_activation_record(
    activation_record: ActivationRecord,
    cutoff: float,
    index: int
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    sample = activation_record
    out = "\n Text excerpt - {}:\n".format(ALPHABET[index])
    out += "<start>"
    out += "".join(sample.tokens)
    out += "<end> \n\n"
    out += "Highly activating tokens:\n"
    high_tokens = []
    for j, activation in enumerate(sample.activations):
        if activation > cutoff:
            high_tokens.append(sample.tokens[j])
    
    out += ','.join(high_tokens)
    out += "\n"
    return out

def highlight_activation_records(
    activation_records: Sequence[ActivationRecord],
    cutoff: float
) -> str:
    """Format a list of activation records into a string."""
    return ("\n"+"".join(
                [
                _highlight_activation_record(
                    activation_record,
                    cutoff = cutoff,
                    index = i,
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
    )

def _highlight_activation_record(
    activation_record: ActivationRecord,
    cutoff: float,
    index: int
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    sample = activation_record
    out = "\n Text excerpt - {}:\n".format(ALPHABET[index])
    out += "<start>"
    for i, token in enumerate(sample.tokens):
        if sample.activations[i] > cutoff:
            out += "["+token+"]"
        else:
            out += token
    out += "<end> \n"
    return out

def highlight_with_summary(
    activation_records: Sequence[ActivationRecord],
    cutoff: float
) -> str:
    """Format a list of activation records into a string."""
    return ("\n"+"".join(
                [
                _highlight_with_summary(
                    activation_record,
                    cutoff = cutoff,
                    index = i,
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
    )

def _highlight_with_summary(
    activation_record: ActivationRecord,
    cutoff: float,
    index: int
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    sample = activation_record
    high_tokens = []
    out = "\n Text excerpt - {}:\n".format(ALPHABET[index])
    out += "<start>"
    for i, token in enumerate(sample.tokens):
        if sample.activations[i] > cutoff:
            out += "["+token+"]"
            high_tokens.append(sample.tokens[i])
        else:
            out += token
    out += "<end> \n\n"
    out += "Highly activating tokens:\n"
    out += ','.join(high_tokens)
    out += "\n"
    return out

def AVHS_activation_records(
    activation_records: Sequence[ActivationRecord],
    max_activation: float
) -> str:
    """Format a list of activation records into a string."""
    return ("\n"+"".join(
                [
                _AVHS_activation_record(
                    activation_record,
                    max_activation = max_activation,
                    index = i,
                )
                for i, activation_record in enumerate(activation_records)
            ]
        )
    )

def _AVHS_activation_record(
    activation_record: ActivationRecord,
    max_activation: float,
    index: int
) -> str:
    """Format neuron activations into a string, suitable for use in prompts."""
    sample = activation_record
    norm_activations = normalize_activations(activation_record.activations, max_activation)
    sample.activations = norm_activations
    out = "\n Text excerpt - {}:\n".format(ALPHABET[index])
    out += "<start>"
    for i, token in enumerate(sample.tokens):
        if sample.activations[i] > 0:
            out += "["+token+"]"
        else:
            out += token
    out += "<end> \n\n"
    out += "Highly activating tokens:\n"
    high_tokens = []
    #here, add both activation token and value
    for j, activation in enumerate(sample.activations):
        if activation > 0:
            high_tokens.append(str(sample.tokens[j])+"\t"+str(activation))

    out += ','.join(high_tokens)
    out += "\n"
    return out