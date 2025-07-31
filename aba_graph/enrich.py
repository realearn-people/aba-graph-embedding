import os
import json
import pandas as pd
import glob

output_dir = "./generated_graphs"
os.makedirs(output_dir, exist_ok=True)

json_files = glob.glob(os.path.join(output_dir, "graph_*.json"))

#enrich attacks from a json file
def enrich_json_with_attack_rules(json_data, undercut_dict):
    assumptions = set(json_data.get("assumptions", []))
    rules = json_data.get("rules", [])
    contraries = json_data.get("contraries", {})
    language = set(json_data.get("language", []))

    new_rules = []
    new_contraries = {}
    new_assumptions = set()

    for a, b in undercut_dict.items():
        #if an assumption a is in the graph
        if a in assumptions:
            #deduce contrary "no_evident_not_" from b
            contr_head = contraries.get(a, f"no_evident_not_{a}")

            rule = {
                "head": contr_head,
                "body": [b]
            }

            if not any(r["head"] == rule["head"] and r["body"] == rule["body"] for r in rules):
                new_rules.append(rule)
            new_contraries[b] = a
            new_assumptions.add(b)

            language.update([a, b, contr_head])

    json_data["rules"].extend(new_rules)
    json_data["assumptions"] = list(set(assumptions) | new_assumptions)
    json_data["contraries"].update(new_contraries)
    json_data["language"] = list(language)

    return json_data


verify_path = "./data/Verif/4. Verify - Task 3 - Staff (Silver).xlsx"
verify_df = pd.read_excel(verify_path)

#filter rows with a "Yes" vote 
valid_votes_df = verify_df[verify_df['Vote'].str.strip() == "Yes"]

#build dict where A is an assumption and B is its contrary
undercut_dict = {}
for _, row in valid_votes_df.iterrows():
    assumption = str(row['A']).strip().lower()
    contrary = str(row['B']).strip().lower()
    if assumption and contrary:
        undercut_dict[assumption] = contrary


json_files = glob.glob("./generated_graphs/graph_*.json")
updated_files = 0

for path in json_files:
    with open(path, 'r') as f:
        aba_json = json.load(f)

    enriched = enrich_json_with_attack_rules(aba_json, undercut_dict)

    with open(path, 'w') as f:
        json.dump(enriched, f, indent=4)

    updated_files += 1

print(f"{updated_files} fichiers enrichis avec règles d'attaque inversées.")


verify_files = {
    "check-in": "./data/Verif/2. Verify - Task 3 - Check-in (Silver).xlsx",
    "check-out": "./data/Verif/1. Verify - Task 3 - Check-out (Silver).xlsx",
    "price": "./data/Verif/3. Verify - Task 3 - Price (Silver).xlsx"
}

all_undercuts = {}
for topic, path in verify_files.items():
    df = pd.read_excel(path)
    valid_votes_df = df[df['Vote'].str.strip() == "Yes"]
    for _, row in valid_votes_df.iterrows():
        a = str(row['A']).strip().lower()
        b = str(row['B']).strip().lower()
        if a and b:
            all_undercuts[a] = b

#enrichment function with inverse attack rules
def enrich_json_with_attack_rules_2(json_data, undercut_dict):
    assumptions = set(json_data.get("assumptions", []))
    rules = json_data.get("rules", [])
    contraries = json_data.get("contraries", {})
    language = set(json_data.get("language", []))

    new_rules = []
    new_contraries = {}
    new_assumptions = set()

    for a, b in undercut_dict.items():
        if a in assumptions:
            contr_head = contraries.get(a, f"no_evident_not_{a}")
            rule = {"head": contr_head, "body": [b]}
            if not any(r["head"] == rule["head"] and r["body"] == rule["body"] for r in rules):
                new_rules.append(rule)
            new_contraries[b] = a
            new_assumptions.add(b)
            language.update([a, b, contr_head])

    json_data["rules"].extend(new_rules)
    json_data["assumptions"] = list(set(assumptions) | new_assumptions)
    json_data["contraries"].update(new_contraries)
    json_data["language"] = list(language)

    return json_data

#apply enrichment to all generated JSON files
output_dir = "./generated_graphs"
json_files = glob.glob(os.path.join(output_dir, "graph_*.json"))

updated_files = 0
for path in json_files:
    with open(path, 'r') as f:
        aba_json = json.load(f)

    enriched = enrich_json_with_attack_rules_2(aba_json, all_undercuts)

    with open(path, 'w') as f:
        json.dump(enriched, f, indent=4)

    updated_files += 1
    

updated_files

print("Available undercut examples :", list(all_undercuts.items())[:10])

