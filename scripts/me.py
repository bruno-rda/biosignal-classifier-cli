# Script for EEG motor execution prediction
from core.processing.features import CSPFeatures
from core.processing.processor import Processor
from .utils import create_app

def get_processor():
    fs = 250
    window_size = 1
    step_size = 0.05

    return Processor(
        fs, window_size, step_size,
        feature_extractor=CSPFeatures()
    )

if __name__ == '__main__':
    app = create_app(
        mode='data',
        signal_type='eeg',
        processor_getter=get_processor,
        data_path='./experiments/data_009',
        session_path='./experiments/session_001',
        automatic_save=False,
    )

    serial_buffer_size = 60
    serial_message_mapping = {
        'move': 'unlock',
        'rest': 'lock',
    }

    app.setup_communicator(
        buffer_size=serial_buffer_size,
        message_mapping=serial_message_mapping
    )

    app.run('udp')