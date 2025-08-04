import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

class SeriesExpansion():
    def __init__():
        pass
    
    @staticmethod
    def getChebychevNodes(N: int, mode:str = ""):
        if mode == "half":
            "Return nparray with N nodes in (0, 1)."
            "Denser at 1. Sorted"
            return np.flipud(np.cos(( 2*np.arange(1, N + 1) - 1) * np.pi / (4 * N)))
        
        "Return nparray with N+1 nodes in (-1, 1)"
        "Denser at -1, 1. Symmetrical. Sorted"
        return np.flipud(np.cos(( 2*np.arange(1, N + 1) - 1) * np.pi / (2 * N)))
        

    @staticmethod
    def getFourierConst(ft: np.ndarray):
        """
        PRE: ft is a matrix of size (M x N). It is the evaluation of a function
            f: [-pi, pi)^M -> C^M at the points linspace(-pi, pi, N+1) (without end).
            N must be even. N should have low prime divisors (opt: N=2^m).
            f should be 2pi periodic.
        POST: CosConst and SinConst are matrices of size M x N/2.
        Desc:
            If f(t) = sum_{n=0}^{K} (a_n cos(nt) + b_n sin(nt)),
            then this function returns a_n and b_n as long as K <= N/2.
            Function can be vectorized along one dimension.
        NOTE:
         - ft can be complex. Then this program divides real part and imaginary part
           and combines them again.
        """
        ft = np.atleast_2d(ft)
        M, N = ft.shape
        if ft.ndim > 2:
            raise ValueError(f"Input ft must be 2D, got {ft.ndim}D.")
    
        # 1) FFT on real and imag parts
        r_fft = np.fft.fft(ft.real, axis=1)
        i_fft = np.fft.fft(ft.imag, axis=1)
        
        # 2) Precompute sign and scale
        k = np.arange(N)
        sign = (-1) ** k
        sign_scale = np.tile(sign, (M, 1)) * (1.0 / N)
        
        # 3) Compute intermediate a/b components
        ar = r_fft.real * sign_scale
        ai = i_fft.real * sign_scale
        br = -r_fft.imag * sign_scale
        bi = -i_fft.imag * sign_scale
        
        # 4) Combine positive/negative frequencies
        K = N // 2
        CosConst = np.empty((M, K), dtype=complex)
        SinConst = np.empty((M, K), dtype=complex)
        
        # zero-frequency term
        CosConst[:, 0] = ar[:, 0] + 1j * ai[:, 0]
        SinConst[:, 0] = br[:, 0] + 1j * bi[:, 0]
        
        # k = 1..K-1
        idx = np.arange(1, K)
        oos = (-1) ** (N%2)
        CosConst[:, idx] = (ar[:, idx] + oos*ar[:, N - idx]) + 1j * (ai[:, idx] + oos*ai[:, N - idx])
        SinConst[:, idx] = (br[:, idx] - oos*br[:, N - idx]) + 1j * (bi[:, idx] - oos*bi[:, N - idx])
        
        return CosConst, SinConst
        
    @staticmethod
    def getFourierInterpCoeff(arr: np.ndarray, multFactor: int = 8, fouriercutoff: int = 5):
        """
        Compute Fourier-interpolated features for a batch of price series.

        Args:
            arr (np.ndarray): 2D array of past price series (shape: [m, n]).
            multFactor (int, optional): Number of interpolated points between samples. Defaults to 8.
            fouriercutoff (int, optional): How many Fourier coefficients to keep. Defaults to 5.

        Raises:
            ValueError: If `arr` is not a 2D numeric array.
            Warning: If `fouriercutoff` is larger than the available coefficient count.

        Returns:
            tuple:
                diffPerStep (np.ndarray): Per-series average step increase (length m).
                cos_coeffs (np.ndarray): Truncated cosine coefficients (shape: [m, fouriercutoff]).
                sin_coeffs (np.ndarray): Truncated sine coefficients (shape: [m, fouriercutoff]).
        """
        arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
        _, n = arr.shape
        x = np.arange(n)
        fx0: float = arr[:, 0]
        fxend: float = arr[:, n-1]
        yfit = fx0[:, None] + (fxend[:, None]-fx0[:, None])*(x[None, :] / (n-1))
        skewedPrices = arr - yfit
        
        fourierInput = np.concatenate((skewedPrices, np.fliplr(-skewedPrices[:, :(n-1)])), axis = -1)
        cs = CubicSpline(np.arange(fourierInput.shape[-1]), fourierInput, bc_type='periodic', axis = -1)
        fourierInputSpline = cs(np.linspace(0, fourierInput.shape[-1]-1, 1 + (fourierInput.shape[-1] - 1) * multFactor))
        fourierInputSmooth = gaussian_filter1d(fourierInputSpline, sigma=np.max([multFactor//4,1]), mode = "wrap", axis = -1)
        res_cos, res_sin = SeriesExpansion.getFourierConst(fourierInputSmooth)
        res_cos = res_cos.real
        res_sin = res_sin.real

        if res_cos.shape[1] < fouriercutoff:
            raise Warning("fouriercutoff is bigger than the array itself.")
    
        endIdx = np.min([res_cos.shape[1], fouriercutoff])
        diffPerStep = (fxend-fx0)/(n-1)

        return diffPerStep, res_cos[:, :endIdx], res_sin[:, :endIdx]
    
    @staticmethod
    def getFourierInterpFunct(res_cos, res_sin, resarr):
        """
        res_cos, res_sin: arrays of shape (K, M)
        resarr           : array of shape (K, N)
        returns
        f_lifted : (K, N)
        rmse     : (K, M)
        """
        resarr = resarr.reshape(1, -1) if resarr.ndim == 1 else resarr
        res_cos = res_cos.reshape(1, -1) if res_cos.ndim == 1 else res_cos
        res_sin = res_sin.reshape(1, -1) if res_sin.ndim == 1 else res_sin
        K, N = resarr.shape
        M = res_cos.shape[1]

        # time & linear lift
        t = np.linspace(-np.pi, 0.0, N)[None, None, :]      # → (1,1,N)
        x = np.arange(N)[None, :]                           # → (1,N)
        fx0   = resarr[:, 0][:, None]                        # → (K,1)
        fxend = resarr[:, -1][:, None]                       # → (K,1)
        lift  = fx0 + (fxend - fx0) * (x / (N-1))            # → (K, N)

        # trig matrix
        n = np.arange(M)[None, :, None]                     # → (1, M, 1)
        cos_term = res_cos[:, :, None] * np.cos(n * t)      # → (K, M, N)
        sin_term = res_sin[:, :, None] * np.sin(n * t)      # → (K, M, N)
        trig = cos_term + sin_term

        # cumulative recon + lift + error
        f_rec_cum     = np.cumsum(trig, axis=1)             # → (K, M, N)
        f_lifted_cum  = f_rec_cum + lift[:, None, :]        # → (K, M, N)
        diff          = f_lifted_cum - resarr[:, None, :]   # → (K, M, N)
        rmse          = np.sqrt(np.mean(diff**2, axis=2))   # → (K, M)
        f_lifted      = f_lifted_cum[:, -1, :]              # → (K, N)

        return f_lifted, rmse