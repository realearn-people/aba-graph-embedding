import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './models')))

from data_process import load_triplets_from_file, filter_triplets
from transE import TransE
from rotatE import RotatE, eval
from r_gcn import RGCN

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/FB15k'))
    train_file = os.path.join(data_dir, 'train.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    valid_file = os.path.join(data_dir, "valid.txt")

    train_triplets = load_triplets_from_file(train_file)[:10000] 
    #model = TransE(train_triplets, emb_dim=100, lr=0.01, margin=1.0)
    #model = RotatE(train_triplets, emb_dim=100, lr=0.01, gamma=6.0)
    model = RGCN(train_triplets, emb_dim=32, nlayers=2)
    #model.train(epochs=50)

    test_triplets = load_triplets_from_file(test_file)[:1000]
    #filtered_test = filter_triplets(test_triplets, model.ent2vec.keys(), model.rel2vec.keys())
    #filtered_test = filter_triplets(test_triplets, model.ent2vec.keys(), model.rel2phase.keys())
    filtered_test = filter_triplets(test_triplets, model.ent2id.keys(), model.rel2id.keys())

    valid_triplets = load_triplets_from_file(valid_file)
    #valid_triplets = filter_triplets(valid_triplets, model.ent2vec.keys(), model.rel2vec.keys())
    valid_triplets = filter_triplets(valid_triplets, model.ent2vec.keys(), model.rel2phase.keys())

    eval_sub = filtered_test[:100]
    mrr, hits = eval(model, eval_sub)
    print(f"MRR: {mrr:.4f}")
    for k, v in hits.items():
        print(f"Hits@{k}: {v:.4f}")

    print("\\nScore examples on test triplets :")
    for h, r, t in filtered_test[:10]:
        print(f"({h}, {r}, {t}) -> Score: {model.score(h, r, t):.4f}")

if __name__ == "__main__":
    main()
