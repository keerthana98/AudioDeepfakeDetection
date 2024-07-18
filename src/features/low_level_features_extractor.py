import librosa
import parselmouth
from parselmouth.praat import call

import numpy as np
from tqdm import tqdm

from .spectral_features import SpectralFeatureExtractor
from .prosodic_features import ProsodicFeatureExtractor
from .voice_quality_features import VoiceQualityFeatureExtractor
from .features_list import DEFAULT_FEATURES


class LowLevelFeatureExtractor:
    
    def __init__(self, target_sr=16000, include_only=None, spectral_features=None, prosodic_features=None, voice_quality_features=None):
        self.target_sr = target_sr
        self.include_spectral = True
        self.include_prosodic = True
        self.include_voice_quality = True
        self.spectral_features = spectral_features if spectral_features is not None else DEFAULT_FEATURES['spectral']
        self.prosodic_features = prosodic_features if prosodic_features is not None else DEFAULT_FEATURES['prosodic']
        self.voice_quality_features = voice_quality_features if voice_quality_features is not None else DEFAULT_FEATURES['voice_quality']

        if include_only is not None:
            self.include_spectral      = 'spectral' in include_only
            self.include_prosodic      = 'prosodic' in include_only
            self.include_voice_quality = 'voice_quality' in include_only
            
            if not self.include_spectral:
                self.spectral_features = []
            
            if not self.include_prosodic:
                self.prosodic_features = []
            
            if not self.include_voice_quality:
                self.voice_quality_features = []

    def resample_audio(self, audio_arr, orig_sr):
        return librosa.resample(audio_arr, orig_sr=orig_sr, target_sr=self.target_sr)

    def extract_features(self, row):
        audio_id = row['audio_id']
        audio_arr = row['audio_arr']
        orig_sr = row['srate']
        real_or_fake = row['real_or_fake']

        y = self.resample_audio(audio_arr, orig_sr)
        
        features = {}
        
        if self.include_spectral:
                spectral_extractor = SpectralFeatureExtractor(y, self.target_sr)
                features.update(spectral_extractor.extract(self.spectral_features))

        if self.include_prosodic:
            prosodic_extractor = ProsodicFeatureExtractor(y, self.target_sr, audio_arr, orig_sr)
            features.update(prosodic_extractor.extract(self.prosodic_features))
            
        if self.include_voice_quality:
            voice_quality_extractor = VoiceQualityFeatureExtractor(audio_arr, orig_sr)
            features.update(voice_quality_extractor.extract(self.voice_quality_features))

        features = {**{'audio_id': audio_id, 'real_or_fake': real_or_fake}, **features}

        return features

    def low_level_feature_generator(self, df):
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audios"):
            yield self.extract_features(row)
