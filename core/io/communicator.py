import serial
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class SerialConnectionManager:
    def __init__(
        self, 
        port: str, 
        baudrate: int = 9600, 
        timeout: float = 1.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._conn: Optional[serial.Serial] = None
        self._client_ids: set[int] = set()

    def register(self, client_id: int):
        if client_id in self._client_ids:
            logger.warning(f'Client {client_id} already registered')
            return
        
        # Open connection on first client
        if len(self._client_ids) == 0:
            self._open()

        self._client_ids.add(client_id)
        logger.info(f'Client {client_id} registered. Total clients: {len(self._client_ids)}')

    def unregister(self, client_id: int):
        if client_id not in self._client_ids:
            logger.warning(f'Client {client_id} not registered')
            return
        
        # Close connection when last client leaves
        if len(self._client_ids) == 1:
            self._close()

        self._client_ids.remove(client_id)
        logger.info(f'Client {client_id} unregistered. Total clients: {len(self._client_ids)}')
    
    def _open(self):
        if self._conn is not None:
            logger.warning('Connection already open')
            return
            
        try:
            self._conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logger.info(f'Serial connection opened: {self.port} @ {self.baudrate}')
        except serial.SerialException as e:
            logger.error(f'Failed to open serial connection: {e}')
            raise
    
    def _close(self):
        if self._conn is None:
            return
            
        try:
            self._conn.close()
            logger.info('Serial connection closed')
        except Exception as e:
            logger.error(f'Error closing connection: {e}')
        finally:
            self._conn = None
    
    @property
    def is_open(self) -> bool:
        return self._conn is not None and self._conn.is_open
    
    def write(self, data: bytes):
        if not self.is_open:
            raise RuntimeError('Cannot write: connection not open')
        
        try:
            self._conn.write(data)
        except serial.SerialException as e:
            logger.error(f'Write failed: {e}')
            raise
    
    @property
    def client_count(self) -> int:
        """Number of registered clients"""
        return len(self._client_ids)


class SerialCommunicator:
    """
    Sends messages through a shared serial connection.
    Buffers messages and flushes the most repeated message when buffer is full.
    """
    
    def __init__(
        self,
        connection_manager: SerialConnectionManager,
        buffer_size: int = 5,
        message_mapping: Optional[dict[Any, str]] = None,
    ):
        self.connection_manager = connection_manager
        self.buffer_size = buffer_size
        self.message_mapping = message_mapping or {}
        
        self._buffer: list[Any] = []
        self._is_registered = False
        
    def enable(self):
        if self._is_registered:
            logger.warning('Communicator already enabled')
            return
            
        self.connection_manager.register(id(self))
        self._is_registered = True
        self._buffer.clear()
        logger.info('Communicator enabled')

    def disable(self):
        if not self._is_registered:
            logger.warning('Communicator already disabled')
            return

        self.connection_manager.unregister(id(self))
        self._is_registered = False
        self._buffer.clear()
        logger.info('Communicator disabled')
    
    @property
    def is_enabled(self) -> bool:
        return self._is_registered

    def send(self, message: Any):
        if not self._is_registered:
            logger.warning(f'Cannot send {message!r}: communicator not enabled')
            return
        
        if not self.connection_manager.is_open:
            logger.warning(f'Cannot send {message!r}: connection not open')
            return
        
        # Add to buffer
        self._buffer.append(message)
        logger.debug(f'Buffered: {message!r} (buffer size: {len(self._buffer)})')
        
        # Auto-flush when buffer full
        if len(self._buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        if not self._buffer:
            return
        
        # Send the most repeated message
        message = max(set(self._buffer), key=self._buffer.count)
        mapped = self.message_mapping.get(message, str(message))
        data = mapped + '\n'
        
        try:
            self.connection_manager.write(data.encode('utf-8'))
            logger.info(f'Flushed {message} {mapped}')
            self._buffer.clear()
        except Exception as e:
            logger.error(f'Flush failed: {e}')
    
    def clear_buffer(self):
        if self._buffer:
            logger.info(f'Cleared {len(self._buffer)} buffered messages')
            self._buffer.clear()
    
    def __del__(self):
        if self.is_enabled:
            self.disable()