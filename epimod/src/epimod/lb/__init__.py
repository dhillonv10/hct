"""LongevityBench (Insilico, Cell 2026) DNA-methylation tasks as numeric data.

The benchmark is distributed as chat prompts. These helpers parse the prompts back into numbers
(probe ids, beta values, labels, sample ids) so a numeric model can be scored on exactly the same
held-out items the language models were scored on.
"""
