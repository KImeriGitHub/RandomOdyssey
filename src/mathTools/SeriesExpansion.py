import numpy as np

class SeriesExpansion():
    def __init__():
        pass

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
        if ft.ndim == 1:
            M, N = 1, ft.shape[0]
            ft = ft.reshape(1, -1)
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
    
        CosConst = (np.hstack([ar[:, :1], ar[:, 1:N//2] + np.fliplr(ar[:, N//2+N%2+1:])]) +
                    1j * np.hstack([ai[:, :1], ai[:, 1:N//2] + np.fliplr(ai[:, N//2+N%2+1:])]))
    
        SinConst = (np.hstack([br[:, :1], br[:, 1:N//2] - np.fliplr(br[:, N//2+N%2+1:])]) +
                    1j * np.hstack([bi[:, :1], bi[:, 1:N//2] - np.fliplr(bi[:, N//2+N%2+1:])]))
    
        return CosConst, SinConst