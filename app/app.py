from .cli import CLI
from .controller import Controller
from core.data.manager import DataManager
from core.model.manager import ModelManager
from core.processing.processor import Processor
from core.io import SerialCommunicator, UDPReceiver, RandomReceiver, CycleReceiver
from multiprocessing.managers import BaseManager
from sklearn.pipeline import Pipeline
from typing import Optional, Literal
import logging

logging.basicConfig(
    level='INFO',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class App:
    def __init__(
        self, 
        data_manager: DataManager,
        model_manager: ModelManager,
        processor: Processor,
        base_dir: str,
        automatic_save: bool
    ):  
        # For data receiver
        self.columns = data_manager.columns

        # Controller and CLI
        controller = Controller(
            data_manager=data_manager,
            model_manager=model_manager,
            processor=processor,
            communicator=None,
            base_dir=base_dir,
            automatic_save=automatic_save
        )
        self.cli = CLI(controller)
    
    @classmethod
    def from_scratch(
        cls,
        columns: list[str],
        processor: Processor,
        metadata_columns: Optional[list[str]] = None,
        pipeline: Optional[Pipeline] = None,
        base_dir: str = './experiments',
        automatic_save: bool = True,
    ) -> 'App':
        data_manager = DataManager(
            columns=columns,
            metadata_columns=metadata_columns
        )
        model_manager = ModelManager(pipeline=pipeline)
        
        return cls(
            data_manager=data_manager,
            model_manager=model_manager,
            processor=processor,
            base_dir=base_dir,
            automatic_save=automatic_save
        )

    @classmethod
    def from_data(
        cls,
        data_path: str,
        processor: Processor,
        pipeline: Optional[Pipeline] = None,
        base_dir: str = './experiments',
        automatic_save: bool = True,
    ) -> 'App':
        data_manager = DataManager.load(data_path)
        model_manager = ModelManager(pipeline=pipeline)
        
        return cls(
            data_manager=data_manager,
            model_manager=model_manager,
            processor=processor,
            base_dir=base_dir,
            automatic_save=automatic_save
        )

    @classmethod
    def from_session(
        cls,
        session_path: str,
        base_dir: str = './experiments',
        automatic_save: bool = True,
    ) -> 'App':
        data_manager = DataManager.load(session_path)
        model_manager = ModelManager.load(session_path)
        processor = Processor.load(session_path)

        return cls(
            data_manager=data_manager,
            model_manager=model_manager,
            processor=processor,
            base_dir=base_dir,
            automatic_save=automatic_save
        )

    @staticmethod
    def _get_connection_manager():
        class MyManager(BaseManager): pass

        MyManager.register('get_connection_manager')
        m = MyManager(address=('localhost', 50000), authkey=b'secretkey')
        
        try:
            m.connect()
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                'Ensure you run the server with the SerialConnectionManager'
            )

        return m.get_connection_manager()

    def setup_communicator(
        self, 
        buffer_size: int,
        message_mapping: Optional[dict[str, str]]
    ) -> None:
        connection_manager = self._get_connection_manager()

        communicator = SerialCommunicator(
            connection_manager=connection_manager,
            buffer_size=buffer_size,
            message_mapping=message_mapping
        )

        self.cli.controller.communicator = communicator

    def run(
        self,
        receiver_type: Literal['udp', 'random', 'cycle'] = 'udp',
        path: Optional[str] = None,
    ):
        if receiver_type == 'udp':
            receiver = UDPReceiver(n_channels=len(self.columns))
        elif receiver_type == 'random':
            receiver = RandomReceiver(n_channels=len(self.columns))
        elif receiver_type == 'cycle':
            assert path is not None, 'Path is required for cycle receiver'
            receiver = CycleReceiver(path=path, columns=self.columns)

        receiver.start()
        self.cli.start()

        try:
            while self.cli.running:
                data = receiver.receive()

                if data is None:
                    continue

                self.cli.on_data(data)

        except Exception as e:
            logger.exception(e)
        
        finally:
            receiver.stop()
            self.cli.stop()
            logger.info('Program terminated')