from core.processing.features import LinearFeatures
from core.processing.processor import Processor
from .utils import create_app

def get_processor():
    fs = 1200
    window_size = 1
    step_size = 0.05

    return Processor(
        fs, window_size, step_size,
        feature_extractor=LinearFeatures()
    )

if __name__ == '__main__':
    app = create_app(
        mode='session',
        signal_type='emg',
        processor_getter=get_processor,
        data_path='./experiments/data_001',
        session_path='./experiments/session_001',
        automatic_save=False,
    )

    serial_buffer_size = 300
    serial_message_mapping = {
        'puño': '33333',
        'uno': '30333',
        'paz': '30033',
        'tres': '30003',
        'cuatro': '30000',
        'cinco': '00000',
        'rock': '30330',
        'ok': '22000',
        'L': '00333',
        'thumbsup': '03333',
        'mentada': '33033',
    }

    app.setup_communicator(
        buffer_size=serial_buffer_size,
        message_mapping=serial_message_mapping
    )

    app.run('cycle', './experiments/data_001/data.csv')