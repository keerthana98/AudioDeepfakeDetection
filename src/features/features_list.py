DEFAULT_SPECTRAL_FEATURES = [
    'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 
    'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate', 
    'mfccs', 'chroma_stft', 'spectral_flux'
]
DEFAULT_PROSODIC_FEATURES = ['f0', 'energy', 'speaking_rate_and_pauses']

ALL_SPECTRAL_FEATURES = [
    'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 
    'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate', 
    'mfccs', 'chroma_stft', 'spectral_flux'
]

ALL_PROSODIC_FEATURES = ['f0', 'energy', 'speaking_rate_and_pauses']


DEFAULT_FEATURES = {
    'spectral': DEFAULT_SPECTRAL_FEATURES,
    'prosodic': DEFAULT_PROSODIC_FEATURES,
}

ALL_FEATURES = {
    'spectral': ALL_SPECTRAL_FEATURES,
    'prosodic': ALL_PROSODIC_FEATURES,
}

