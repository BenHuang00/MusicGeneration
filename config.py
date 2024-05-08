is_development = True

dataset_path = 'datasets/DadaGP-v1.1' if not is_development else 'datasets/DadaGP-v1.1-dev'

# preprocess settings
window_size = 200
window_step = 1

preprocess_batch_size = 4096   # The number of items within each file, recommend to be same as the batch size of the model

preprocess_path = 'preprocess'


# DO NOT MODIFY THE CODE BELOW #
dataset_all_metadata_name = '_DadaGP_all_metadata.json'
dataset_all_tokens_name = '_DadaGP_all_tokens.json'

tokens2ids_name = 'tokens2ids.pkl'
ids2tokens_name = 'ids2tokens.pkl'
