from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237

result = pipeline(
    model='TransE',
    dataset=FB15k237,  
    model_kwargs=dict(embedding_dim=50, random_seed=42), 
    optimizer_kwargs=dict(lr=0.01),  
    training_kwargs=dict(num_epochs=50, use_tqdm_batch=False), 
    loss="softplus", 
    random_seed=42,
    device='cpu'
)

print(f"Mean Reciprocal Rank: {result.get_metric('mean_reciprocal_rank')}")
print(f"hits@10: {result.get_metric('hits@10')}")