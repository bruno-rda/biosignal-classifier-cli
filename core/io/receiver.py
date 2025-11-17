import socket
import numpy as np
import time
import logging
import pandas as pd
from typing import Optional
from core.processing.windowing import Windowing

logger = logging.getLogger(__name__)

class Receiver:
    def receive(self) -> np.ndarray: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class UDPReceiver(Receiver):
    def __init__(self, n_channels: int, port: int = 8000):
        self.n_channels = n_channels
        self.port = port

    def receive(self) -> Optional[np.ndarray]:
        try:
            pkt, _ = self.sock.recvfrom(65535)
            
            if len(pkt) % 8 != 0:
                pkt = pkt[:-len(pkt) % 8]

            data = np.frombuffer(pkt, dtype='<f8')
            n_package_samples = len(data) // self.n_channels
            data = np.array(data).reshape(n_package_samples, self.n_channels)
            return data

        except socket.timeout:
            return None

        except Exception as e:
            raise e

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.settimeout(1)
        logger.info(f'UDPReceiver started on port {self.port}')
    
    def stop(self) -> None:
        self.sock.close()


class RandomReceiver(Receiver):
    def __init__(self, n_channels: int):
        self.n_channels = n_channels

    def receive(self) -> np.ndarray:
        time.sleep(0.005)
        return np.random.rand(8, self.n_channels)
    
    def start(self) -> None:
        logger.info('Mocking data (random)')


class CycleReceiver(Receiver):
    def __init__(self, path: str, columns: list[str]):
        self.df = pd.read_csv(path)[columns]
        self.idx = 0
        self.windowing = Windowing(60, 30)
        self.windows = self.windowing.transform(self.df.values)

    def receive(self) -> np.ndarray:
        time.sleep(0.01)
        window = self.windows[self.idx]
        self.idx += 1
        self.idx %= len(self.windows)
        return window
    
    def start(self) -> None:
        logger.info('Mocking data (cycle)')