from app import App
from core.processing.processor import Processor
from typing import Literal, Callable, Optional

def create_app(
    mode: Literal['scratch', 'data', 'session'], 
    signal_type: Optional[Literal['emg', 'eeg']] = None,
    processor_getter: Optional[Callable[[], Processor]] = None,
    data_path: Optional[str] = None,
    session_path: Optional[str] = None,
    automatic_save: bool = True,
) -> App:
    if mode == 'scratch':
        if signal_type == 'emg':
            columns = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'TIMESTAMP']
            metadata_columns = ['TIMESTAMP']

        else:
            columns = [
                'Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8', 
                'ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z', 
                'GYROSCOPE_X', 'GYROSCOPE_Y', 'GYROSCOPE_Z', 
                'BATTERY_LEVEL', 'COUNTER', 'VALIDATION_INDICATOR', 'TIMESTAMP'
            ]
            metadata_columns = ['BATTERY_LEVEL', 'COUNTER', 'VALIDATION_INDICATOR', 'TIMESTAMP']
        
        return App.from_scratch(
            columns=columns,
            metadata_columns=metadata_columns,
            processor=processor_getter(),
            automatic_save=automatic_save,
        )

    if mode == 'data':
        return App.from_data(
            data_path=data_path,
            processor=processor_getter(),
            automatic_save=automatic_save,
        )

    if mode == 'session':
        return App.from_session(
            session_path=session_path,
            automatic_save=automatic_save,
        )