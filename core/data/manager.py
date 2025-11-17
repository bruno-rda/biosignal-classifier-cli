import pandas as pd
import numpy as np
from typing import Optional, Any
from core.io.utils import save_json, load_json
import logging

logger = logging.getLogger(__name__)

class DataManager:
    _data_filename = 'data.csv'
    _details_filename = 'details_data.json'

    def __init__(
        self,
        columns: list[str],
        metadata_columns: Optional[list[str]] = None
    ):
        self.columns = columns
        self.metadata_columns = metadata_columns or []
        
        assert set(self.metadata_columns).issubset(set(self.columns)), \
            'Metadata columns must be a subset of the columns'

        self.chunks: list[np.ndarray] = []        
        self.labels: list[Any] = []
        self.groups: list[int] = []
        self.signal_idxs = [
            i for i, col in enumerate(self.columns) 
            if col not in self.metadata_columns
        ]

    @property
    def details(self) -> dict[str, Any]:
        details = {
            'columns': self.columns,
            'metadata_columns': self.metadata_columns,
        }

        if self.has_data():
            df = pd.DataFrame({'label': self.labels, 'group': self.groups})
            groups_by_label = df.groupby('label')['group'].unique().to_dict()
            chunk_shape = self.chunks[0].shape

            details.update({
                'shape': (len(self.chunks) * chunk_shape[0], chunk_shape[1]),
                'unique_labels': list(set(self.labels)),
                'n_groups': len(set(self.groups)),
                'groups_by_label': groups_by_label
            })

        return details
    
    def has_data(self) -> bool:
        return bool(self.chunks and self.labels and self.groups)

    def _validate_shape(self, data: np.ndarray):
        data = np.atleast_2d(data)
        assert data.shape[1] == len(self.columns), \
            'Data must have the same number of columns as the dataset. ' \
            f'Expected {len(self.columns)}, got {data.shape[1]}'

    def add_chunk(
        self, 
        chunk: np.ndarray, 
        label: Any, 
        group: int = 0
    ):
        self._validate_shape(chunk)

        chunk = np.atleast_2d(chunk)
        self.chunks.append(chunk)
        self.labels.append(label)
        self.groups.append(group)
    
    def next_group_id(self) -> int:
        return max(self.groups) + 1 if self.groups else 0
    
    def delete_group(self, group_id: int):
        chunks, labels, groups = [], [], []
        for chunk, label, group in zip(self.chunks, self.labels, self.groups):
            if group == group_id:
                continue
            
            chunks.append(chunk)
            labels.append(label)
            groups.append(group)
        
        self.chunks = chunks
        self.labels = labels
        self.groups = groups
    
    def extract_signal(self, data: np.ndarray):
        """ Used before training or predicting. """
        self._validate_shape(data)

        if data.ndim == 1:
            return data[self.signal_idxs]

        return data[:, self.signal_idxs]
    
    def get_dataset(self, metadata: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Reshapes data into an ML ready dataset. """
        if not self.has_data():
            raise ValueError('No data to get')

        data = np.vstack(self.chunks)
        if not metadata: 
            data = self.extract_signal(data)

        labels = np.repeat(self.labels, [len(chunk) for chunk in self.chunks])
        groups = np.repeat(self.groups, [len(chunk) for chunk in self.chunks])
        return data, labels, groups

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        metadata_columns: Optional[list[str]] = None
    ):
        assert 'label' in df.columns, 'Label column is required'
        assert 'group' in df.columns, 'Group column is required'

        columns = [col for col in df.columns if col not in ['label', 'group']]
        manager = cls(columns, metadata_columns)
        
        for (group, label), chunk in df.groupby(['group', 'label']):
            manager.add_chunk(
                chunk=chunk[columns].values,
                label=label,
                group=group
            )
        return manager
    
    def save(
        self, 
        dir_path: str,
        data_filename: str = _data_filename,
        details_filename: str = _details_filename,
    ):
        if not self.has_data():
            raise ValueError('No data to save')

        data, labels, groups = self.get_dataset(metadata=True)
        df = pd.DataFrame(data, columns=self.columns)
        df['label'] = labels
        df['group'] = groups
        
        df.to_csv(f'{dir_path}/{data_filename}', index=False)
        save_json(self.details, f'{dir_path}/{details_filename}')
        logger.info(f'Data saved to {dir_path}')

    @classmethod
    def load(
        cls, 
        dir_path: str,
        data_filename: str = _data_filename,
        details_filename: str = _details_filename,
    ):
        df = pd.read_csv(f'{dir_path}/{data_filename}')
        details = load_json(f'{dir_path}/{details_filename}')

        return cls.from_dataframe(
            df=df,
            metadata_columns=details['metadata_columns']
        )
    
    def reset(self):
        self.chunks: list[np.ndarray] = []        
        self.labels: list[Any] = []
        self.groups: list[int] = []