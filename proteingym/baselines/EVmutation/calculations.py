"""
High-level mutation calculation functions for EVmutation

.. todo::

    implement segment handling

Authors:
  Thomas A. Hopf
  Anna G. Green (generalization for multiple segments)
"""

import numpy as np
import pandas as pd
from evcouplings.utils.calculations import entropy_map


COMPONENT_TO_INDEX = {
    "full": 0,
    "couplings": 1,
    "fields": 2,
}


def extract_mutations(mutation_string, offset=0, sep=":"):
    """
    Turns a string containing mutations of the format I100V into a list of tuples with
    format (100, 'I', 'V') (index, from, to)

    Parameters
    ----------
    mutation_string : str
        Comma-separated list of one or more mutations (e.g. "K50R,I100V")
    offset : int, default: 0
        Offset to be added to the index/position of each mutation
    sep : str, default ","
        String used to separate multiple mutations

    Returns
    -------
    list of tuples
        List of tuples of the form (index+offset, from, to)
    """
    try:
        if mutation_string.lower() not in ["wild", "wt", ""]:
            mutations = mutation_string.split(sep)
            return list(map(
                lambda x: (int(x[1:-1]) + offset, x[0], x[-1]),
                mutations
            ))
    except AttributeError:
        return []


def predict_mutation_table(model, table, output_column="prediction_epistatic",
                           mutant_column="mutant", hamiltonian="full", offset=0,
                           segment=None, sep=":"):
    """
    Predicts all mutants in a dataframe and adds predictions
    as a new column.

    If mutant_column is None, the dataframe index is used,
    otherwise the given column.

    Mutations which cannot be calculated (e.g. not covered
    by alignment, or invalid substitution) using object are
    set to NaN.

    Parameters
    ----------
    model : CouplingsModel
        CouplingsModel instance used to compute mutation
        effects
    table : pandas.DataFrame
        DataFrame with mutants to which delta of
        statistical energy will be added
    mutant_column: str
        Name of column in table that contains mutants
    output_column : str
        Name of column in returned dataframe that will
        contain computed effects
    hamiltonian: {"full", "couplings", "fields"},
            default: "full"
        Use full Hamiltonian of exponential model (default),
        or only couplings / fields for statistical energy
        calculation.
    segment: str, default: None
        Specificy a segment identifier to use for the positions in the mutation
        table. This will only be used if the mutation table doesn't already have
        a segments column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with added column (mutant_column) that contains computed
        mutation effects
    """
    def _predict_mutant(m):
        # try:
            delta_E = model.delta_hamiltonian(m)
            return delta_E[_component]

        # except ValueError:
            # return np.nan

    # select Hamiltonian component for prediction
    if hamiltonian in COMPONENT_TO_INDEX:
        _component = COMPONENT_TO_INDEX[hamiltonian]
    else:
        raise ValueError(
            "Invalid selection for hamiltonian. "
            "Valid values are: " + ", ".join(COMPONENT_TO_INDEX)
        )

    # make sure there is a target sequence for which we
    # can compute statistical energy difference
    if not model.has_target_seq:
        raise ValueError(
            "CouplingsModel object does not have a target "
            "sequence (non-focus mode). "
            "Set target sequence, or rerun inference in focus mode."
        )
    pred = table.copy()

    # get column which contains mutations
    if mutant_column is None:
        mutations = pred.index
    else:
        mutations = pred.loc[:, mutant_column]

    # if there is a segment column, use that to apply
    # segment information to every mutation
    if "segment" in pred.columns and pred.loc[:, "segment"].notnull().all():
        segments = pred.loc[:, "segment"]

        # split each comma-delimited string of mutations into a list
        mutations_separated = [extract_mutations(x,offset=offset) for x in mutations] #map(extract_mutations, mutations)

        # split each comma-delimited string of segments into a list
        segments_separated = [x.split(",") for x in segments]
        mutation_list = []

        # create a list of mutation in the format
        # [[((segment, pos), aa_from, aa_to), ((segment, pos) aa_from, aa_to)], [((segment, pos) aa_from, aa_to)]]
        if len([segments_separated]) != len([mutations_separated]):
            raise(
                ValueError,
                "Number of mutations provided does not match number of segments of origin provided."
            )

        for segment_subset, mutation_subset in zip(segments_separated, mutations_separated):
            _mutation_list = [
                ((seg, pos), aa_from, aa_to) for
                (seg, (pos, aa_from, aa_to)) in zip(
                    segment_subset, mutation_subset
                )
            ]
            mutation_list.append(_mutation_list)

    # else if the segment argument was provided
    # designate that as the segment for every mutation
    elif segment is not None:
        mutations_separated = [extract_mutations(x,offset=offset) for x in mutations]  #map(extract_mutations, mutations)
        mutation_list = []
        for mutation_subset in mutations_separated:
            _mutation_list = [
                ((segment, pos), aa_from, aa_to) for
                (pos, aa_from, aa_to) in mutation_subset
            ]
            mutation_list.append(_mutation_list)

    else:
        mutation_list = [extract_mutations(x,offset=offset) for x in mutations]  #map(extract_mutations, mutations)
    # predict mutations and add to table
    pred.loc[:, output_column] = [
        _predict_mutant(m) for m in mutation_list
    ]

    return pred


def single_mutant_matrix(model, output_column="prediction_epistatic",
                         exclude_self_subs=True):
    """
    Create table with all possible single substitutions of
    target sequence in CouplingsModel object.

    Parameters
    ----------
    model : CouplingsModel
        Model that will be used to predict single mutants
    output_column : str, default: "prediction_epistatic"
        Name of column in Dataframe that will contain predictions
    exclude_self_subs : bool, default: True
        Exclude self-substitutions (e.g. A100A) from results

    Returns
    -------
    pandas.DataFrame
        DataFrame with predictions for all single mutants
    """
    res = []
    cons = entropy_map(model)

    # iterate all positions and substitutions per position
    for pos in model.index_list:
        for subs in model.alphabet:
            # do not predict gaps
            if subs in ["-", "."]:
                continue

            # exclude self-substitutions?
            if exclude_self_subs and subs == model.seq(pos):
                continue

            # if position is a tuple, it is in format
            # (segment_id, position). Else, there is
            # no segment information
            if isinstance(pos, tuple):
                position_str = pos[1]
                segment = pos[0]

            else:
                position_str = pos
                segment = np.nan

            wt = model.seq(pos)
            mutant = "{}{}{}".format(wt, position_str, subs)

            res.append(
                {
                    "segment": segment,
                    "mutant": mutant,
                    "pos": position_str,
                    "wt": wt,
                    "subs": subs,
                    "frequency": model.fi(pos, subs),
                    "column_conservation": cons[pos],
                    output_column: model.smm(pos, subs),
                }
            )

    pred = pd.DataFrame(res)
    return pred.loc[
        :, ["segment", "mutant", "pos", "wt", "subs", "frequency",
            "column_conservation", output_column]
    ]


def split_mutants(x, mutant_column="mutant"):
    """
    Splits mutation strings into individual columns in DataFrame
    (wild-type symbol(s), position(s), substitution(s), number of mutations).
    This function is e.g. helpful when computing average
    effects per position using pandas groupby() operations

    Parameters
    ----------
    x : pandas.DataFrame
        Table with mutants
    mutant_column : str, default: "mutant"
        Column which contains mutants, set to None
        to use index of DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns "num_subs", "pos", "wt"
        and "subs" that contain the number of mutations,
        and split mutation strings (if higher-order mutations,
        symbols/numbers are comma-separated)
    """
    def _split(mut_str):
        try:
            return sorted(extract_mutations(mut_str))
        except ValueError:
            return np.nan

    def _join(index):
        return [
            ",".join([str(subs[index]) for subs in mutant])
            for mutant in spl
        ]

    # get column which contains mutations
    if mutant_column is None:
        mutations = x.index
    else:
        mutations = x.loc[:, mutant_column]

    # extract wt/pos/subs where possible
    spl = mutations.map(_split)

    # then store in individual columns
    x.loc[:, "num_mutations"] = [len(mutant) for mutant in spl]
    for i, column in enumerate(["pos", "wt", "subs"]):
        x.loc[:, column] = _join(i)

    return x
