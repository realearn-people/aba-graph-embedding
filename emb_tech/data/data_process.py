def load_triplets_from_file(path):
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                triplets.append((h, r, t))
    return triplets

def filter_triplets(triplets, ent_set, rel_set):
    return [
        (h, r, t) for (h, r, t) in triplets
        if h in ent_set and t in ent_set and r in rel_set
    ]
