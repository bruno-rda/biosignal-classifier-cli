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
        mode='scratch',
        signal_type='eeg',
        processor_getter=get_processor,
        data_path='./experiments/data_001',
        session_path='./experiments/session_001',
        automatic_save=False,
    )

    serial_buffer_size = 300
    serial_message_mapping = {
        'izq': 'cal',
        'der': 'cal',
    }

    app.setup_communicator(
        buffer_size=serial_buffer_size,
        message_mapping=serial_message_mapping
    )

    app.run('random')