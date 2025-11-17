import joblib
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.base import clone, BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectPercentile, f_classif
from core.io.utils import save_json, load_json
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _pipeline_filename = 'pipeline.joblib'
    _details_filename = 'details_pipeline.json'

    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline = pipeline or self._create_default_pipeline()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_metrics: dict[str, Any] = {}
    
    @property
    def details(self) -> dict[str, Any]:
        return {
            'classes': self.label_encoder.classes_.tolist()
            if hasattr(self.label_encoder, 'classes_')
            else [],
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'pipeline_params': self.pipeline.get_params(deep=True)
        }

    @staticmethod
    def _create_default_pipeline() -> Pipeline:
        pipeline = Pipeline([
            ('feature_selector', SelectPercentile(
                f_classif, percentile=90
            )),
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True))
        ])

        logger.info(f'Using default pipeline: {pipeline}')
        return pipeline
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        groups: np.ndarray,
        cross_validate: bool = True
    ):  
        if cross_validate:
            self.training_metrics = self.cross_validate(X, y, groups)
        
        logger.info('Training on full dataset...')
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline = clone(self.pipeline)
        self.pipeline.fit(X, y_encoded)
        
        # Store final metrics
        y_pred = self.pipeline.predict(X)
        self.training_metrics.update({
            'final_accuracy': accuracy_score(y_encoded, y_pred),
            'final_confusion_matrix': confusion_matrix(y_encoded, y_pred)
        })

        self.is_trained = True
        logger.info(f'Training accuracy: {self.training_metrics["final_accuracy"]:.3f}')
        
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        groups: np.ndarray, 
        random_state: Optional[int] = None
    ) -> dict[str, Any]:
        random_state = random_state or np.random.randint(0, 1e3)
        logger.info(f'Running cross-validation with random state: {random_state}')

        if pd.Series(groups).groupby(pd.Series(y)).nunique().min() == 1:
            logger.warning(
                'At least one label has only one unique group — this may cause uneven class '
                'distribution in cross-validation folds and reduce reliability of results.'
            )

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        y = self.label_encoder.fit_transform(y)
        
        metrics = {
            'accuracy': [],
            'f1': [],
            'confusion_matrices': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_pipeline = clone(self.pipeline)
            fold_pipeline.fit(X_train, y_train)
            y_pred = fold_pipeline.predict(X_val)
            
            # Compute metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            cm = confusion_matrix(y_val, y_pred)
            
            metrics['accuracy'].append(acc)
            metrics['f1'].append(f1)
            metrics['confusion_matrices'].append(cm)
            
            logger.info(f'Fold {fold}: Accuracy={acc:.3f}, F1={f1:.3f}')
        
        accuracy = metrics['accuracy']
        logger.info(f'Mean accuracy: {np.mean(accuracy):.3f} ± {np.std(accuracy):.3f}')
        return metrics
    
    def predict(self, X: np.ndarray) -> list[tuple[Any, float, dict[Any, float]]]:
        X = np.atleast_2d(X)
        probs = self.pipeline.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        preds_encoded = self.label_encoder.inverse_transform(preds)

        return [
            (
                pred_encoded, 
                prob.max(), 
                dict(zip(self.label_encoder.classes_, prob))
            )
            for pred_encoded, prob in zip(preds_encoded, probs)
        ]
    
    def save(
        self, 
        dir_path: str,
        pipeline_filename: str = _pipeline_filename,
        details_filename: str = _details_filename,
    ):
        if not self.is_trained:
            raise ValueError('No trained model to save')

        joblib.dump(self.pipeline, f'{dir_path}/{pipeline_filename}')
        save_json(self.details, f'{dir_path}/{details_filename}')
        logger.info(f'Model saved to {dir_path}')
        
    @classmethod
    def load(
        cls, 
        dir_path: str,
        pipeline_filename: str = _pipeline_filename,
        details_filename: str = _details_filename,
    ) -> 'ModelManager':
        pipeline = joblib.load(f'{dir_path}/{pipeline_filename}')
        details = load_json(f'{dir_path}/{details_filename}')

        manager = cls(pipeline)
        manager.label_encoder.classes_ = np.array(details['classes'])
        manager.training_metrics = details['training_metrics']
        manager.is_trained = details['is_trained']
        return manager
    
    def reset(self):
        self.pipeline = clone(self.pipeline)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_metrics = {}