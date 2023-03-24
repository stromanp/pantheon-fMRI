def two_gamma_hrf(length=32,
                  TR=2,
                  peak_delay=6,
                  under_delay=16,
                  peak_disp=1,
                  under_disp=1,
                  p_u_ratio=6,
                  normalize=True,
                  ):
    """ HRF function from sum of two gamma PDFs
    It is a *peak* gamma PDF (with location `peak_delay` and
    dispersion `peak_disp`), minus an *undershoot* gamma PDF
    (with location `under_delay` and dispersion `under_disp`,
    and divided by the `p_u_ratio`).
    Parameters
    ----------
    length: float
        length of HRF in seconds
    TR : float
        repetition (sampling) time in seconds
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.
    Returns
    -------
    hrf : array
        vector of samples from HRF at TR intervals
    """

    t = np.linspace(0, length, length/TR)
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)