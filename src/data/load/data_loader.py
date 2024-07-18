from .codecfake import fetch_audio_id_list, get_audio_dataset, get_dataset_from_single_parquet


def get_codecfake_audio_id_list():
    """
    Fetch the list of audio IDs in the codecfake dataset.
    """
    return fetch_audio_id_list()

def load_audio_data(audio_ids, dataset='codecfake', cache_dir=None):
    """
    Load audio data for the given audio IDs.
    """
    if dataset == 'codecfake':
        return get_audio_dataset(audio_ids, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid dataset: {dataset}") 
    
def load_parquet_data(partition_id, cache_dir=None):
    """
    Load audio data from a single parquet file.
    """
    return get_dataset_from_single_parquet(partition_id, cache_dir=cache_dir)