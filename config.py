is_development = False

dataset_path = 'datasets/DadaGP-v1.1' if not is_development else 'datasets/DadaGP-v1.1-dev'

# preprocess settings
window_size = 200
window_step = 1

exclude_file_head = False

preprocess_path = 'datasets/preprocess' if not is_development else 'datasets/preprocess-dev'


# DO NOT MODIFY THE CODE BELOW #
dataset_all_metadata_name = '_DadaGP_all_metadata.json'
dataset_all_tokens_name = '_DadaGP_all_tokens.json'

tokens2ids_name = 'tokens2ids.json'
ids2tokens_name = 'ids2tokens.json'

preprocess_dataset_name = 'preprocess_dataset.pkl'
