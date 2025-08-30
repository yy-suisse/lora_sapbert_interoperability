# UMLS concept analysis
This repo does the following things, please use `embed_and_find_nearst_across_all_ontologies.ipynb`:  
1. Load the trained LoRA checkpoint
2. Get preprocessed dataframe of HUG-SNOMED and ULMS 
3. Get embeddings of concepts from the above mentioned dataframe, the exact columns to embed are: 
   1. HUG-SNOMED dataframe column: `label` (fully defined name for pre-coordination, cleaned expression for post-coordination)
   2. UMLS dataframe column: `STR` (All textual terms of a given concept)
4. Compute the cosine similarity for between 2 given tensors of embeddings, which are ``query tensor`` and ``candidate tensor``, following are implemented as examples, be sure to follow the logic of slice as in example when you want to evaluate some rows of dataframe:
   1. similarity between SNOMED and all UMLS (just an implemented example, this doesnt make sense, because SNOMED is inside of UMLS too)
   2. similarity between free text and all UMLS
   3. similarity between post and all UMLS
   4. similarity between mapped concepts and all UMLS
   5. similarity mapped concepts and all ICD-9 AND 10