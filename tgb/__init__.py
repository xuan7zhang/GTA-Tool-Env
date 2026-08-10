"""TGB -- a tool-grounded benchmark for likelihood-based tool valuation.

The 8,500-task controlled set (dataset_full.json) establishes that gold
likelihood tracks *evidence* utility. It contains no tools: the evidence is
handed to the model already extracted. TGB is the layer above it -- real raw
inputs, real (deterministic) tool execution, several candidate coalitions per
task, and a closed-loop selection test -- so that the paper's tool-utility
claim rests on tools rather than on evidence strings.
"""
