import os
import json
import pandas as pd
from pathlib import Path

input_dir = Path("../generated_graphs_augmented_by_topic")
output_tsv_path = Path("../aba_triples.tsv")

triples = []

for file in input_dir.glob("*.json"):
    with open(file, 'r') as f:
        graph = json.load(f)
    
    assumptions = set(graph.get("assumptions", []))
    contraries = graph.get("contraries", {})
    rules = graph.get("rules", [])

    #add rules as triplets
    for rule in rules:
        head = rule.get("head")
        for body_literal in rule.get("body", []):
            triples.append((body_literal, "supports", head))

    #add contraries as attacks (undercut)
    for a, not_a in contraries.items():
        triples.append((a, "attacks", not_a))

df_triples = pd.DataFrame(triples, columns=["head", "relation", "tail"])
df_triples.to_csv(output_tsv_path, sep="\t", index=False, header=False)

output_tsv_path, df_triples.shape
