"""
Copyright (C) Tristan Bepler - All Rights Reserved
Author: Tristan Bepler <tbepler@gmail.com>
"""

from __future__ import division, print_function


def parse_stream(f, comment=b"#", upper=True):
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b">"):
            if name is not None:
                yield name, b"".join(sequence)
            name = line[1:]
            sequence = []
        else:
            if upper:
                sequence.append(line.upper())
            else:
                sequence.append(line)
    if name is not None:
        yield name, b"".join(sequence)


def parse(f, comment=b"#"):
    names = []
    sequences = []
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b">"):
            if name is not None:
                names.append(name)
                sequences.append(b"".join(sequence))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        names.append(name)
        sequences.append(b"".join(sequence))

    return names, sequences
