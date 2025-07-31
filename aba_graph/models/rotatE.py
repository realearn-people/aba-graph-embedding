import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.model_selection import train_test_split

df = pd.read_csv("../aba_triples.tsv", sep='\t', header=None)
df.columns = ['head', 'relation', 'tail']

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

training_factory = TriplesFactory.from_labeled_triples(train_df.values)
testing_factory = TriplesFactory.from_labeled_triples(test_df.values)

result = pipeline(
    model="RotatE",
    training=training_factory,
    testing=testing_factory,
    model_kwargs=dict(embedding_dim=512),
    training_kwargs=dict(num_epochs=200),
    evaluator_kwargs=dict(filtered=True),
    random_seed=42,
    device="cuda"  
)

metrics = result.metric_results.to_dict()

print("\nEvaluation Results of RotatE :")
print(f"Mean Reciprocal Rank: {result.get_metric('mean_reciprocal_rank')}")
print(f"Hits@10{result.get_metric('hits@10')}")
