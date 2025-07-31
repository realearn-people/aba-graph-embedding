import os
import json
import pandas as pd
import glob

file_path = "./data/data_reviews.xlsx"
xls = pd.ExcelFile(file_path)

sheet_names = xls.sheet_names

df = xls.parse(sheet_names[0])
df.head()

output_dir = "./generated_graphs"
os.makedirs(output_dir, exist_ok=True)

#At least a Head and one Body + corresponding Contrary
def is_valid_row(row):
    if pd.isna(row["Head"]):
        return False
    body_cols = [f"Body {i}" for i in range(1, 16)]
    contr_cols = [f"Cont. Body {i}" for i in range(1, 16)]
    has_valid_body = any(not pd.isna(row[col]) for col in body_cols)
    has_valid_contrary = any(not pd.isna(row[col]) for col in contr_cols)
    return has_valid_body and has_valid_contrary

#clean and convert to literal format
def clean_literal(literal):
    return str(literal).strip().lower()

generated = 0
for idx, row in df.iterrows():
    if not is_valid_row(row):
        continue

    head = clean_literal(row["Head"])

    body_literals = [clean_literal(row[f"Body {i}"]) for i in range(1, 16) if not pd.isna(row[f"Body {i}"])]
    contraries_literals = [clean_literal(row[f"Cont. Body {i}"]) for i in range(1, 16) if not pd.isna(row[f"Cont. Body {i}"])]

    #build ABA 
    aba_json = {
        "language": list(set([head] + body_literals + contraries_literals)),
        "rules": [
            {
                "head": head,
                "body": body_literals
            }
        ],
        "assumptions": body_literals,
        "contraries": {
            assumption: contrary
            for assumption, contrary in zip(body_literals, contraries_literals)
        }
    }

    json_filename = os.path.join(output_dir, f"graph_{idx+1}.json")
    with open(json_filename, 'w') as f:
        json.dump(aba_json, f, indent=4)

    generated += 1

generated

