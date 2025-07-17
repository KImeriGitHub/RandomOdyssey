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
    def getFourierConst(ft):
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
        # Handle both 1D and 2D cases
        if ft.ndim == 1 and ft.shape[0] == 1:
            M, N = 1, ft.shape[1]
        elif ft.ndim == 1 and ft.shape[0] > 1:
            M, N = 1, ft.shape[0]
            ft = ft.reshape(1,-1)
        elif ft.ndim == 2:
            M, N = ft.shape
        elif ft.ndim > 2:
            raise ValueError
    
        rFfft = np.fft.fft(np.real(ft), axis=1)
        iFfft = np.fft.fft(np.imag(ft), axis=1)
    
        cjrFfft = np.conj(rFfft)
        cjiFfft = np.conj(iFfft)
    
        signmat = (-1) ** np.arange(N)
        signmat = np.tile(signmat, (M, 1))
    
        ar = (rFfft + cjrFfft) * signmat / (N * 2)
        br = (rFfft - cjrFfft) * signmat * (1j) / (N * 2)
    
        ai = (iFfft + cjiFfft) * signmat / (N * 2)
        bi = (iFfft - cjiFfft) * signmat * (1j) / (N * 2)
    
        CosConst = (np.hstack([ar[:, :1], ar[:, 1:N//2] + (-1) ** (N%2)*np.fliplr(ar[:, N//2+N%2+1:])]) +
                    1j * np.hstack([ai[:, :1], ai[:, 1:N//2] + (-1) ** (N%2)*np.fliplr(ai[:, N//2+N%2+1:])]))
    
        SinConst = (np.hstack([br[:, :1], br[:, 1:N//2] - (-1) ** (N%2)*np.fliplr(br[:, N//2+N%2+1:])]) +
                    1j * np.hstack([bi[:, :1], bi[:, 1:N//2] - (-1) ** (N%2)*np.fliplr(bi[:, N//2+N%2+1:])]))
    
        return CosConst, SinConst
    
    @staticmethod
    def getFourierInterpCoeff(arr: np.array, multFactor: int = 8, fouriercutoff: int = 5):
        """
        Args:
            pastPrices (np.array): slice of prices
            multFactor (int, optional): adds that many points in between. Defaults to 8.
            fouriercutoff (int, optional): from all fourier coeffs cutoff that many. Defaults to 25.

        Raises:
            ValueError: if pastPrices invalid
            Warning: if fouriercutoff too high

        Returns:
            list[float]: features: first one is average increase, then cos coeffs, then sin coeffs
        """
        n = len(arr)
        if n==1:
            raise ValueError("Input to getFeatureFromPrice are invalid.")
        if n==2:
            return arr
        
        x = np.arange(n)
        fx0: float = arr[0]
        fxend: float = arr[n-1]
        yfit = fx0 + (fxend-fx0)*(x/(n-1))
        skewedPrices = arr-yfit
        
        fourierInput = np.concatenate((skewedPrices,np.flipud(-skewedPrices[:(n-1)])))
        cs = CubicSpline(np.arange(len(fourierInput)), fourierInput, bc_type='periodic')
        fourierInputSpline = cs(np.linspace(0, len(fourierInput)-1, 1 + (len(fourierInput) - 1) * multFactor))
        fourierInputSmooth = gaussian_filter1d(fourierInputSpline, sigma=np.max([multFactor//4,1]), mode = "wrap")
        res_cos, res_sin = SeriesExpansion.getFourierConst(fourierInputSmooth)
        res_cos=res_cos.T.real.flatten().tolist()
        res_sin=res_sin.T.real.flatten().tolist()

        if len(res_cos) < fouriercutoff:
            raise Warning("fouriercutoff is bigger than the array itself.")

        endIdx = np.min([len(res_cos), fouriercutoff])
        diffPerStep = (fxend-fx0)/(n-1)

        return diffPerStep, res_cos[:endIdx], res_sin[:endIdx]
        
    
    @staticmethod
    def getFourierInterpFunct(res_cos, res_sin, resarr: np.array):
        # Add fourier approximation error to the features
        N  = resarr.size
        M  = len(res_cos)                      # number of Fourier terms
        t  = np.linspace(-np.pi, 0.0, N)       # shape (N,)
        x  = np.arange(N)
        fx0, fxend = resarr[0], resarr[-1]
        lift = fx0 + (fxend - fx0) * (x / (N - 1))   # shape (N,)
    
        # ---- pre‑compute trig matrices ----------------------------------------
        n   = np.arange(M)[:, None]            # shape (M,1)
        trig = (np.asarray(res_cos)[:, None] * np.cos(n * t) +
                np.asarray(res_sin)[:, None] * np.sin(n * t))   # shape (M,N)
    
        # ---- cumulative reconstruction & error --------------------------------
        f_rec_cum = np.cumsum(trig, axis=0)           # shape (M,N)
        f_lifted_cum = f_rec_cum + lift               # broadcast
        diff = f_lifted_cum - resarr                  # broadcast
        rsme = np.sqrt(np.mean(diff**2, axis=1))      # shape (M,)
    
        f_lifted = f_lifted_cum[-1]                   # final reconstruction
        return f_lifted, rsme