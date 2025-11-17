from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
import numpy as np
import threading
from .controller import Controller
import logging

logger = logging.getLogger(__name__)

class CLI:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.running = False
        
        self.session = PromptSession(
            bottom_toolbar=self.controller.get_status,
            style=Style.from_dict({'bottom-toolbar': 'bg:default noreverse'}),
        )
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.lock = threading.Lock()

    def on_data(self, data: np.ndarray):
        with self.lock:
            self.controller.process_data(data)
            self.session.app.invalidate() # Refresh the UI

    def _input_loop(self):
        with patch_stdout(raw=True):
            while self.running:
                try:
                    user_input = self.session.prompt('> ')
                    user_input = user_input.strip().lower()
                    
                    if not user_input:
                        continue
                    
                    with self.lock:
                        params = self.controller.get_cmd_params(user_input)
                        args = [
                            self.session.prompt(f'>> {arg}: ')
                            for arg in params
                        ]
                        self.controller.handle_command(user_input, args)
                
                except KeyboardInterrupt:
                    logger.info('Keyboard interrupt detected. Exiting...')
                    self.stop()
                    break

                except Exception as e:
                    logger.exception(f'Unexpected error: {e}')
    
    def start(self):
        if self.running:
            return

        self.running = True
        self.input_thread.start()
        logger.info('CLI started - ready for command input')

    def stop(self):
        if not self.running:
            return
            
        self.running = False
        logger.info('CLI stopped')