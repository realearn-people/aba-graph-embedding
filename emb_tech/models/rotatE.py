import numpy as np
import random

class RotatE:
    def __init__(self, triplets, emb_dim=50, gamma=6.0, lr=0.01):
        self.triplets = triplets 
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.lr = lr

        self.entities = set([h for h,_,t in triplets] + [t for h,_,t in triplets])
        self.rel = set([r for _,r,_ in triplets])

        self.ent2vec = {
            e: self.init_complex_emb() for e in self.entities
        }

        self.rel2phase = {
            r: self.init_phase_emb() for r in self.rel
        }


    def init_complex_emb(self):
        phase = np.random.uniform(0, 2 * np.pi, self.emb_dim)
        rad = np.random.uniform(0, 1, self.emb_dim)
        return rad * (np.cos(phase) + 1j * np.sin(phase))
    
    
    def init_phase_emb(self):
        return np.random.uniform(-np.pi, np.pi, self.emb_dim)


    def dist(self, h, r, t):
        rota = h * np.exp(1j * r)
        return np.linalg.norm(rota - t)
    

    def project_disk(self, vec):
        norm = np.abs(vec)
        return np.where(norm > 1, vec / norm, vec)


    def corrupt_triplets(self, h, r, t):
        corrupt_h = random.random() < 0.5
        if corrupt_h:
            h_corr = random.choice(list(self.entities - {h}))
            return (h_corr, r, t)
        else:
            t_corr = random.choice(list(self.entities - {t}))
            return (h, r, t_corr)
        

    def train(self, epochs=10):
        for e in range(epochs):
            loss_total = 0.0
            random.shuffle(self.triplets)

            for h, r, t in self.triplets:
                h_c, r_c, t_c = self.corrupt_triplets(h, r, t)

                h_vec = self.ent2vec[h]
                r_phase = self.rel2phase[r]
                t_vec = self.ent2vec[t]

                hc_vec = self.ent2vec[h_c]
                tc_vec = self.ent2vec[t_c]

                pos_dist = self.dist(h_vec, r_phase, t_vec)
                neg_dist = self.dist(hc_vec, r_phase, tc_vec)

                loss = max(0, self.gamma + pos_dist - neg_dist)
                loss_total += loss

                if loss > 0:
                    grad = (h_vec * np.exp(1j * r_phase) - t_vec)
                    self.ent2vec[h] -= self.lr * grad
                    self.ent2vec[t] += self.lr * grad

                    gneg = (hc_vec * np.exp(1j * r_phase) - t_vec)
                    self.ent2vec[h_c] += self.lr * gneg
                    self.ent2vec[t_c] -= self.lr * gneg

                    self.ent2vec[h] = self.project_disk(self.ent2vec[h])
                    self.ent2vec[t] = self.project_disk(self.ent2vec[t])
                    self.ent2vec[h_c] = self.project_disk(self.ent2vec[h_c])
                    self.ent2vec[t_c] = self.project_disk(self.ent2vec[t_c])

            print(f"Epoch {e+1}/{epochs}, Loss : {loss_total:.4f}")

    def score(self, h, r, t):
        h_vec = self.ent2vec[h]
        r_phase = self.rel2phase[r]
        t_vec = self.ent2vec[t]
        return self.dist(h_vec, r_phase, t_vec)


def eval(model, test_triplets, k_list=[1, 3, 10]):
    hits_at_k = {k: 0 for k in k_list}
    mrr = 0.0
    total = 0

    entities = list(model.ent2vec.keys())

    for h, r, t in test_triplets:
        if h not in model.ent2vec or r not in model.rel2phase or t not in model.ent2vec:
            continue

        # Tail prediction: (h, r, ?)
        tail_scores = []
        for candidate in entities:
            score = model.score(h, r, candidate)
            tail_scores.append((candidate, score))
        tail_scores = sorted(tail_scores, key=lambda x: x[1])
        tail_ranked = [ent for ent, _ in tail_scores]
        tail_rank = tail_ranked.index(t) + 1

        mrr += 1.0 / tail_rank
        for k in k_list:
            if tail_rank <= k:
                hits_at_k[k] += 1

        # Head prediction: (?, r, t)
        head_scores = []
        for candidate in entities:
            score = model.score(candidate, r, t)
            head_scores.append((candidate, score))
        head_scores = sorted(head_scores, key=lambda x: x[1])
        head_ranked = [ent for ent, _ in head_scores]
        head_rank = head_ranked.index(h) + 1

        mrr += 1.0 / head_rank
        for k in k_list:
            if head_rank <= k:
                hits_at_k[k] += 1

        total += 2  # two predictions per test triplet

    mrr /= total
    hits_at_k = {k: v / total for k, v in hits_at_k.items()}

    return mrr, hits_at_k
