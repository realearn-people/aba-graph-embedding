import os
import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


source_dir = Path("./generated_graphs/")
output_dir = Path("./generated_graphs_augmented_by_topic")
output_dir.mkdir(exist_ok=True)

for f in output_dir.glob("*.json"):
    f.unlink()

distribution = {
    "small": 4000,
    "medium": 4000,
    "large": 2000
}
k_values_used = {
    "small": (2, 3),
    "medium": (4, 5),
    "large": (6, 8)
}

#Topics
theme_keywords = ["staff", "check-in", "check-out", "price"]

#Load graphs with their topics
def extract_theme(graph):
    text = " ".join(graph.get("language", []) + graph.get("assumptions", []) +
                   [r["head"] for r in graph.get("rules", []) if "head" in r]).lower()
    for kw in theme_keywords:
        if kw in text:
            return kw
    return None

#Load all valid graphs and group them by topic 
graphs_by_theme = defaultdict(list)
for path in source_dir.glob("graph_*.json"):
    with open(path) as f:
        g = json.load(f)
        theme = extract_theme(g)
        if theme:
            graphs_by_theme[theme].append(g)

#merge and clean 
def merge_graphs(graph_list, theme):
    merged = {
        "language": set(),
        "rules": [],
        "assumptions": set(),
        "contraries": {}
    }

    #Merge elements of each graph
    for g in graph_list:
        merged["language"].update(f"{theme}:{x}" for x in g.get("language", []))
        merged["assumptions"].update(f"{theme}:{x}" for x in g.get("assumptions", []))
        merged["rules"].extend({
            "head": f"{theme}:{r['head']}",
            "body": [f"{theme}:{b}" for b in r["body"]]
        } for r in g.get("rules", []))
        merged["contraries"].update({
            f"{theme}:{k}": f"{theme}:{v}" for k, v in g.get("contraries", {}).items()
        })

    #Clean : delete unconnected entities
    connected = set()
    for rule in merged["rules"]:
        connected.add(rule["head"])
        connected.update(rule["body"])
    merged["language"] = list(merged["language"].intersection(connected))
    merged["assumptions"] = list(merged["assumptions"].intersection(connected))
    merged["rules"] = [r for r in merged["rules"] if r["head"] in connected and all(b in connected for b in r["body"])]
    merged["contraries"] = {k: v for k, v in merged["contraries"].items() if k in connected and v in connected}

    return merged if len(merged["rules"]) >= 3 else None 

generated = 0
for category, count in distribution.items():
    k_range = k_values_used[category]
    for i in tqdm(range(count), desc=f"Generating {category}"):
        k = random.randint(*k_range)
        candidate_themes = [t for t, lst in graphs_by_theme.items() if len(lst) >= k]
        if not candidate_themes:
            continue
        theme = random.choice(candidate_themes)
        selected = random.sample(graphs_by_theme[theme], k)
        merged = merge_graphs(selected, theme)
        if merged:
            fname = f"aba_{category}_{theme}_{i+1}.json"
            with open(output_dir / fname, 'w') as f:
                json.dump(merged, f, indent=4)
            generated += 1

print(f"\n{generated} graphes générés avec cohérence thématique.")