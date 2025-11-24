import os
import json
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Any, Optional
from collections import deque
from core.data.manager import DataManager
from core.model.manager import ModelManager
from core.processing.processor import Processor
from core.io import SerialCommunicator, get_next_dir
import logging

logger = logging.getLogger(__name__)

class Mode(Enum):
    MAIN = auto()
    COLLECTION = auto()
    PREDICTION = auto()

@dataclass
class Command:
    description: str
    handler: Callable[..., None]
    params: Optional[list[str]] = None

class Controller:
    def __init__(
        self, 
        data_manager: DataManager,
        model_manager: ModelManager,
        processor: Processor,
        communicator: Optional[SerialCommunicator],
        base_dir: str,
        automatic_save: bool,
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.processor = processor
        self.communicator = communicator
        self.base_dir = base_dir
        self.automatic_save = automatic_save

        self.mode = Mode.PREDICTION if self.model_manager.is_trained else Mode.MAIN
        self.status_text = ''
        self._skipped_samples = 0
        
        # prediction buffers
        self.window_samples = self.processor.window_samples
        self.step_samples = self.processor.step_samples
        self.buffer = deque(maxlen=self.window_samples)
        self.current_steps = 0
        self.n_predictions = 0

        self.common_cmds = {
            'x': Command('Exit program', self._exit),
            'z': Command('Clear screen', self._clear_screen),
            'o': Command('Toggle automatic training save', self._toggle_auto_save),
        }
        self.commands_by_mode = {
            Mode.MAIN: {
                'c': Command('Start collection', self._start_collection, params=['label']),
                'd': Command('Delete group', self._delete_group, params=['group_id']),
                't': Command('Train model and start prediction', self._train_model),
                'i': Command('Show dataset info', self._show_dataset_info),
                's': Command('Save data', self._save_data),
                'r': Command('Reset data', self._reset_data, params=['[y/N]']),
                **self.common_cmds,
            },
            Mode.COLLECTION: {
                'q': Command('Quit collection', self._quit_collection),
                'n': Command('Next group', self._next_group),
                **self.common_cmds,
            },
            Mode.PREDICTION: {
                'r': Command('Reset model', self._reset_model, params=['[y/N]']),
                's': Command('Save all (data/model/processing)', self._save_all),
                'a': Command('Toggle serial communication', self._toggle_comm),
                **self.common_cmds,
            },
        }

        self.mode_menu = {
            mode: self._get_mode_menu(mode)
            for mode in Mode
        }
    
    def _get_mode_menu(self, mode: Mode) -> str:
        txt = '=' * 20 + f' {mode.name.capitalize()} ' + '=' * 20
        txt += '\n\nCommands:\n'

        for trigger, cmd in self.commands_by_mode[mode].items():
            txt += f'\n    {trigger}: {cmd.description}'
        
        return txt
    
    def _switch_mode(self, mode: Mode, summary: str = ''):
        logger.info(f'{self.mode.name} -> {mode.name}')
        summary and logger.info(summary)
        self.mode = mode

    def get_status(self) -> str:
        txt = self.mode_menu[self.mode]
        txt += f'\n\n{self.status_text}'
        return txt
    
    def get_cmd_params(self, trigger: str) -> list[str]:
        cmd = self.commands_by_mode[self.mode].get(trigger)
        return (cmd and cmd.params) or []

    def handle_command(
        self, 
        trigger: str, 
        args: Optional[list[Any]] = None
    ) -> None:
        cmd = self.commands_by_mode[self.mode].get(trigger)

        if not cmd:
            logger.error(f'Unknown command: {trigger!r}')
            return
        
        logger.info(
            f'Executing {trigger!r}: {cmd.description}' + 
            (f' | Args: {args!r}' if args else '')
        )
        cmd.handler(*(args or []))

    def process_data(self, data: np.ndarray) -> None:
        if self.mode == Mode.MAIN:
            self._main_process_data(data)
        elif self.mode == Mode.COLLECTION:
            self._collection_process_data(data)
        elif self.mode == Mode.PREDICTION:
            self._prediction_process_data(data)

    def _main_process_data(self, data: np.ndarray):
        self._skipped_samples += len(data)
        self.status_text = f' Skipped samples: {self._skipped_samples}'

    def _collection_process_data(self, data: np.ndarray):
        self._current_samples += len(data)
        self.data_manager.add_chunk(data, self._current_label, self._current_group)
        self.status_text = f' Collecting {self._current_label!r} | n_samples = {self._current_samples}'
    
    def _prediction_process_data(self, data: np.ndarray):
        for row in data:
            signal = self.data_manager.extract_signal(row)
            self.buffer.append(signal)
            self.current_steps += 1
            
            if (
                len(self.buffer) == self.window_samples 
                and self.current_steps >= self.step_samples
            ):
                self.current_steps = 0
                self._predict()

    def _predict(self):
        self.n_predictions += 1
        window = np.array(self.buffer)
        feats = self.processor.transform(window)
        pred, prob, probs = self.model_manager.predict(feats)[0]

        probs_fmt = '\n'.join([
            f'{p:.2%} {'->' if label == pred else ''} {str(label)}' 
            for label, p in probs.items()]
        )

        self.status_text = (
            f' Prediction {self.n_predictions}: ' 
            f' [{prob:.2%} {str(pred)!r}] \n\n{probs_fmt}'
        )

        if (
            self.communicator 
            and self.communicator.is_enabled 
            and prob > 0.8
        ):
            self.communicator.send(pred)

    # Main commands

    def _start_collection(self, label: str):
        assert label, 'Label cannot be empty'
        self._current_label = label
        self._current_group = self.data_manager.next_group_id()
        self._current_samples = 0
        self._switch_mode(
            Mode.COLLECTION, 
            f'Starting collection for {label!r} | Group: {self._current_group}'
        )

    def _delete_group(self, group_id: str):
        assert group_id.isdigit(), 'Group ID must be a number'
        assert int(group_id) >= 0, 'Group ID must be non-negative'
        self.data_manager.delete_group(int(group_id))

    def _train_model(self):
        X, y, groups = self.data_manager.get_dataset()
        X, y, groups = self.processor.fit_transform(X, y, groups)
        self.model_manager.train(X, y, groups)

        if self.automatic_save:
            self._save_all()

        self._switch_mode(Mode.PREDICTION)

    def _show_dataset_info(self):
        print(
            json.dumps(
                self.data_manager.details,
                indent=4,
                default=str
            )
        )

    def _save_data(self):
        d = get_next_dir(self.base_dir, 'data')
        self.data_manager.save(d)

    def _reset_data(self, confirm: str):
        assert confirm == 'y', 'Action cancelled'
        self.data_manager.reset()

    # Collection commands

    def _quit_collection(self):
        self._switch_mode(
            Mode.MAIN,
            f' Collection completed | {self._current_label!r} | {self._current_samples} samples'
        )
    
    def _next_group(self):
        label = self._current_label
        self._quit_collection()
        self._start_collection(label)

    # Prediction commands

    def _reset_model(self, confirm: str):
        assert confirm == 'y', 'Action cancelled'
        self.model_manager.reset()
        self._switch_mode(Mode.MAIN)

    def _save_all(self):
        d = get_next_dir(self.base_dir, 'session')
        self.data_manager.save(d)
        self.model_manager.save(d)
        self.processor.save(d)

    def _toggle_comm(self):
        com = self.communicator
        if not com:
            raise ValueError('No communicator available')
        com.disable() if com.is_enabled else com.enable()

    # Common commands
    def _exit(self):
        raise KeyboardInterrupt
    
    def _clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _toggle_auto_save(self):
        self.automatic_save = not self.automatic_save
        text = 'enabled' if self.automatic_save else 'disabled'
        logger.info(f'Automatic training save: {text}')