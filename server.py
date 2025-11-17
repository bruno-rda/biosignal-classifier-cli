from multiprocessing.managers import BaseManager
from core.io import SerialConnectionManager
import logging

logging.basicConfig(
    level='INFO',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    connection_manager = SerialConnectionManager(
        '/dev/cu.usbserial-10', 
        baudrate=115200
    )

    class MyManager(BaseManager): ...
    MyManager.register(
        'get_connection_manager', 
        callable=lambda: connection_manager,
        exposed=['register', 'unregister', 'client_count', 'is_open', 'write']
    )

    m = MyManager(address=('', 50000), authkey=b'secretkey')
    server = m.get_server()
    logger.info('Manager server running on port 50000 ...')
    server.serve_forever()