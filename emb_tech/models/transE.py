import numpy as np
import random

class TransE:
    def __init__(self, triplets, emb_dim=50, lr=0.01, margin=1.0, norm="L1"):
        self.triplets = triplets
        self.emb_dim = emb_dim
        self.lr = lr
        self.margin = margin
        self.norm = norm

        self.entities = set([h for h,_,t in triplets] + [t for h,_,t in triplets])
        self.rel = set([r for _,r,_ in triplets])

        self.ent2vec = {
            e: self.init_emb() for e in self.entities
        }

        self.rel2vec = {
            r: self.init_emb() for r in self.rel
        }


    def init_emb(self):
        l = 6 / (np.sqrt(self.emb_dim))
        return np.random.uniform(-l, l, self.emb_dim)


    def dist(self, h, r, t):
        if self.norm == "L1":
            return np.sum(np.abs(h+r-t))
        elif self.norm == "L2":
            return np.linalg.norm(h+r-t)


    def normalize(self, vec):
        return vec / (np.linalg.norm(vec))
    

    def corrupt_triplets(self, h, r, t):
        corrupt_h = random.random() < 0.5
        if corrupt_h:
            h_corr = random.choice(list(self.entities - {h}))
            return (h_corr, r, t)
        else:
            t_corr = random.choice(list(self.entities - {t}))
            return (h, r, t_corr)
        

    def train(self, epochs=100):
        for e in range(epochs):
            random.shuffle(self.triplets)
            total_loss = 0

            for h, r, t in self.triplets:
                h_c, r_c, t_c = self.corrupt_triplets(h, r, t)

                h_vec = self.ent2vec[h]
                r_vec = self.rel2vec[r]
                t_vec = self.ent2vec[t]

                hc_vec = self.ent2vec[h_c]
                tc_vec = self.ent2vec[t_c]

                pos_dist = self.dist(h_vec, r_vec, t_vec)
                neg_dist = self.dist(hc_vec, r_vec, tc_vec)

                loss = max(0, self.margin + pos_dist - neg_dist)
                total_loss += loss

                if loss > 0:
                   grad = np.sign(h_vec + r_vec - t_vec)

                   self.ent2vec[h] -= self.lr * grad
                   self.rel2vec[r] -= self.lr * grad
                   self.ent2vec[t] += self.lr * grad

                   grad_c = np.sign(hc_vec + r_vec - tc_vec)
                   self.ent2vec[h_c] += self.lr * grad_c
                   self.ent2vec[t_c] -= self.lr * grad_c

                   for ent in [h, t, h_c, t_c]:
                       self.ent2vec[ent] = self.normalize(self.ent2vec[ent])
                   self.rel2vec[r] = self.normalize(self.rel2vec[r])

            print(f"Epoch {e+1}/{epochs}, Loss: {total_loss:.4f}")

    def get_ent_emb(self, ent):
        return self.ent2vec.get(ent)
    
    def get_rel_emb(self, rel):
        return self.rel2vec.get(rel)
    
    def score(self, h, r, t):
        h_vec = self.ent2vec[h]
        r_vec = self.rel2vec[r]
        t_vec = self.ent2vec[t]
        return self.dist(h_vec, r_vec, t_vec)




        
                 