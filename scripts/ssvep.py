# Script for EEG SSVEP prediction
from core.processing.features import CombinedFeatures, CCAFeatures, FFTFeatures, PSDFeatures
from core.processing.processor import Processor
from .utils import create_app

def get_processor():
    fs = 250
    window_size = 1
    step_size = 0.05
    target_frequencies = [12, 15]

    return Processor(
        fs, window_size, step_size,
        feature_extractor=CombinedFeatures(
            CCAFeatures(fs=fs, target_frequencies=target_frequencies),
            FFTFeatures(fs=fs, target_frequencies=target_frequencies),
            PSDFeatures(fs=fs, target_frequencies=target_frequencies),
        )
    )

if __name__ == '__main__':
    app = create_app(
        mode='scratch',
        signal_type='eeg',
        processor_getter=get_processor,
        data_path='./experiments/data_001',
        session_path='./experiments/session_001',
        automatic_save=False,
    )

    serial_buffer_size = 60
    serial_message_mapping = {
        'izq': 'cal',
        'der': 'cal',
    }

    app.setup_communicator(
        buffer_size=serial_buffer_size,
        message_mapping=serial_message_mapping
    )

    app.run('random')