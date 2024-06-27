import requests
from collections import defaultdict
from datasets import load_dataset


USERNAME      = "ajaykarthick"
DATASET_NAME  = "codecfake-audio"
REPO_ID       = f"{USERNAME}/{DATASET_NAME}"
JSON_FILE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/audio_id_to_file_map.json"


def fetch_audio_id_list():
    """
    Fetch the list of audio IDs from the dataset.
    """
    response = requests.get(JSON_FILE_URL)
    response.raise_for_status()
    audio_id_to_file_map = response.json()
    
    return list(audio_id_to_file_map.keys())

def get_audio_dataset(audio_ids, cache_dir=None):
    """
    Fetch the dataset for given audio ID or list of audio IDs.
    """
    response = requests.get(JSON_FILE_URL)
    response.raise_for_status()
    audio_id_to_file_map = response.json()

    if isinstance(audio_ids, str):
        audio_ids = [audio_ids]

    # Create a dictionary to map parquet files to audio IDs
    parquet_to_audio_ids = defaultdict(list)
    for audio_id in audio_ids:
        parquet_file = audio_id_to_file_map[audio_id]['train']
        parquet_to_audio_ids[parquet_file].append(audio_id)

    # Create a generator to yield filtered examples from each parquet file
    def dataset_generator():
        for parquet_file, ids in parquet_to_audio_ids.items():
            if cache_dir:
                dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", cache_dir=cache_dir)
            else:
                dataset = load_dataset("parquet", data_files={'train': parquet_file}, split="train", streaming=True)
            
            filtered_ds = dataset.filter(lambda example: example['audio_id'] in ids)
            for example in filtered_ds:
                yield example

    return dataset_generator()
