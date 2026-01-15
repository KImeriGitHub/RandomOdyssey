import numpy as np

class RegimeChange:
    @staticmethod
    def to_time(x, inc_factor):
        """
        Convert tree-based values to time-based values.
        
        Parameters
        ----------
        x : np.ndarray
            Input array of tree-based values.
        inc_factor : float
            Increase factor for scaling.
            
        Returns
        -------
        np.ndarray
            Values between 0 and 1. Centered around 0.5.
            
        Notes
        -----
        - Applying to_tree after to_time with the same inc_factor returns roughly the original x.
        - Savety clipping to avoid numerical issues with log(0) or arctanh(1).
        """
        return np.clip(np.tanh(np.log(np.clip(x, 1e-8, None)) * inc_factor) / 2.0 + 0.5, 1e-8, 1 - 1e-8)
    
    @staticmethod
    def to_tree(y, inc_factor):
        """
        Convert time-based values back to tree-based values.
        
        Parameters
        ----------
        y : np.ndarray
            Input array of time-based values.
        inc_factor : float
            Increase factor for scaling.
            
        Returns
        -------
        np.ndarray
            Values between 0 and infinity. Centered around 1.0.
            
        Notes
        -----
        - Applying to_tree after to_time with the same inc_factor returns roughly the original y.
        """
        return np.exp(np.arctanh((y - 0.5) * 2.0)/inc_factor)  