from .codecfake import fetch_audio_id_list, get_audio_dataset


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