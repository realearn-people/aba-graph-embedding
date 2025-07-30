# aba-graph-embedding

aba_graph folder : 
⦁	aba_graph.py : allow user to create and visualize ABA graphs from a json file (the folder "json" contains some examples)

⦁	The data folder contains the main Dataset "data_reviews.xlsx" and a Verification folder containing the files used to do edge mining 

⦁	xlsx_data_to_json.py : allow generating simple ABA graphs from the Dataset "data_reviews"
⦁	enrich.py : using Verification files to draw some edges on the previous generated graphs 
⦁	gen_graph.py : create graph regroup by their topics from the graph created before
⦁	The models folder is containing : 
	- util.py : convert graphs into a .tsv file 
	- transE.py and rotatE.py : training these models on the .tsv file (with PyKeen)


emb_tech folder : 
⦁	data folder with some classical datasets for testing graph embedding techniques 
⦁	models folder containing implémentations from scratch of embedding techniques 
