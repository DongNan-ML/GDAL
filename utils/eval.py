import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ThreeDEvaluator:
    r"""
        Evaluator for the 3D datasets.
        Metric is Mean Absolute Error.
    """
    def __init__(self):
        pass 

    def eval(self, input_dict):
        r"""Run evaluation.

        Args:
            input_dict (dict): A python dict with the following items: :obj:`y_true` and :obj:`y_pred`. 
            :obj:`y_true` and :obj:`y_pred` need to be of the same type (either numpy.ndarray or torch.Tensor) and the same shape.

        :rtype: :class:`dict` (a python dict with items :obj:`mae`, :obj:`rmse`, and :obj:`r2`)
        """
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)

        # Convert to numpy arrays for sklearn
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy().flatten()
            y_pred_np = y_pred.detach().cpu().numpy().flatten()
        else:
            y_true_np = y_true.flatten()
            y_pred_np = y_pred.flatten()
        
        # Use sklearn metrics
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        r2 = r2_score(y_true_np, y_pred_np)
        
        return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}