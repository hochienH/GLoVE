"""
alpha101_long
================

This module provides a complete and modular implementation of the 101 formulaic
alphas originally published by Zura Kakushadze (2015) in his paper “101
Formulaic Alphas”.  The implementation is designed to operate directly on
long‑format input data – a single pandas DataFrame with the following
columns:

    - ``code``: instrument identifier (e.g., stock ticker)
    - ``date``: timestamp (must be sortable)
    - ``industry_code``: sector/industry/subindustry classification used for
      cross‑sectional neutralisation
    - ``open``: open price
    - ``high``: high price
    - ``low``: low price
    - ``close``: close price
    - ``volume``: traded volume (number of shares/contracts)
    - ``volume_amount``: dollar (or base currency) traded value
    - ``cap``: market capitalisation

The module exposes a single high‑level class, :class:`Alpha101`, which
encapsulates both a library of common operators (rolling sums, cross‑sectional
rank, etc.) and the 101 alpha definitions themselves.  On construction the
class accepts a long‑format DataFrame and pivots it internally to a wide
format (index by ``date`` and columns by ``code``) for efficient vectorised
operations.  All alphas return a DataFrame of the same shape (dates × codes)
containing the daily alpha values.

The emphasis is on clarity and modularity rather than ultimate performance.
Where necessary the continuous window lengths from the original paper have
been rounded to the nearest integer.  The code avoids in‑place modification of
inputs and attempts to guard against divide‑by‑zero and other numerical
pathologies.  Operators such as ``indneutralize`` make use of the provided
``industry_code`` column to demean values within each industry on each day.

Example
-------

::

    import pandas as pd
    from alpha101_long import Alpha101

    # assume df is a long‑format DataFrame containing the required fields
    engine = Alpha101(df)
    signal = engine.alpha001()
    print(signal.tail())

"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

###############################################################################
# Helper and operator functions
###############################################################################

def _ensure_df(x, name="input") -> pd.DataFrame:
    """Ensure that *x* is a DataFrame.  If it is a Series it will be
    converted to a single‑column DataFrame.  Otherwise a TypeError is raised.

    Parameters
    ----------
    x : pandas.Series or pandas.DataFrame
        Input object to convert.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(f"{name} must be Series or DataFrame, got {type(x)}")


def _rowwise_group_demean(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Demean each row of *x* by subtracting the mean value within each
    group defined in *groups*.

    Both *x* and *groups* must share the same index and columns.  The
    ``groups`` DataFrame should contain a categorical value for each cell
    specifying the grouping of each element.  Typically it comes from
    pivoting an ``industry_code`` column.

    Parameters
    ----------
    x : pandas.DataFrame
        The DataFrame whose values will be demeaned.
    groups : pandas.DataFrame
        A DataFrame of the same shape as *x* containing group labels.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with group means subtracted from each row.
    """
    # Make a copy to avoid mutating the input
    out = x.copy()
    for date in out.index:
        # group labels for the row; may contain NaNs if an industry code is
        # missing; drop NaNs to avoid grouping them separately
        codes = groups.loc[date]
        row = out.loc[date]
        # compute mean per group; transform so it matches the index order
        means = row.groupby(codes).transform('mean')
        out.loc[date] = row - means
    return out


def indneutralize(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Cross‑sectionally demean the values in *df* using the group labels
    provided in *groups*.

    The groups DataFrame is expected to be the same shape as *df* and to
    contain categorical labels indicating group membership (e.g. industry).
    For each date (row) we subtract the mean of *df* within each group.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to demean.
    groups : pandas.DataFrame
        Group labels; must align with *df*.

    Returns
    -------
    pandas.DataFrame
        Group‑demeaned values.
    """
    return _rowwise_group_demean(df, groups)


def div(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    numer = _ensure_df(numer, "numer")
    denom = _ensure_df(denom, "denom")
    denom = denom.replace(0.0, np.nan)
    return numer / denom

def log(x: pd.DataFrame) -> pd.DataFrame:
    x = _ensure_df(x, "x")
    x = x.replace(0.0, np.nan)
    return np.log(x)


def _safe_compare(left, right, op) -> pd.DataFrame:
    """Elementwise comparison that preserves NaN where inputs are insufficient."""
    left_df = _ensure_df(left, "left")
    if isinstance(right, (pd.Series, pd.DataFrame)):
        right_df = _ensure_df(right, "right")
    else:
        right_df = pd.DataFrame(right, index=left_df.index, columns=left_df.columns)
    result = op(left_df, right_df)
    valid = left_df.notna() & right_df.notna()
    return result.where(valid).astype("boolean")


def safe_gt(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a > b)


def safe_ge(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a >= b)


def safe_lt(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a < b)


def safe_le(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a <= b)


def safe_eq(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a == b)


def safe_ne(left, right) -> pd.DataFrame:
    return _safe_compare(left, right, lambda a, b: a != b)


def ts_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sum over the past *window* observations."""
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).sum()


def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean over the past *window* observations."""
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).mean()


def ts_stddev(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation over the past *window* observations."""
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).std()


def ts_min(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling minimum over the past *window* observations."""
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).min()


def ts_max(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling maximum over the past *window* observations."""
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).max()


def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling rank: for each date take the last value's rank over the past
    *window* observations, expressed as a fraction between 0 and 1 (1 being
    the largest).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    window : int
        Number of observations to consider.

    Returns
    -------
    pandas.DataFrame
        Rolling rank of the last element in each window.
    """
    df = _ensure_df(df)

    def _last_rank(a: np.ndarray) -> float:
        # rankdata ranks ascending; we want the fractional rank of the last
        r = rankdata(a)
        return r[-1] / len(r)

    return df.rolling(window, min_periods=window).apply(_last_rank, raw=True)


def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the position of the maximum value within each rolling window.
    The result is an integer between 1 and *window* inclusive, where 1
    corresponds to the oldest observation in the window.
    """
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).apply(np.argmax, raw=True) + 1


def ts_argmin(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the position of the minimum value within each rolling window.
    The result is an integer between 1 and *window* inclusive, where 1
    corresponds to the oldest observation in the window.
    """
    df = _ensure_df(df)
    return df.rolling(window, min_periods=window).apply(np.argmin, raw=True) + 1


def delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Difference between the current value and the value *period* periods
    ago."""
    df = _ensure_df(df)
    return df.diff(periods=period)


def delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Shift the data back by *period* periods (lag)."""
    df = _ensure_df(df)
    return df.shift(periods=period)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross‑sectional rank across columns for each row.  The lowest value
    gets a rank of 1/n and the highest a rank of 1.  Ties are assigned
    average ranks.
    """
    df = _ensure_df(df)
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """Scale each row of *df* so that the sum of the absolute values equals
    *k*.  If the sum is zero the row is left unchanged.
    """
    df = _ensure_df(df)
    denom = df.abs().sum(axis=1).replace(0.0, np.nan)
    return df.mul(k).div(denom, axis=0)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling correlation between *x* and *y* over a window of given length.
    NaNs are propagated and infinite values are replaced with zero.
    """
    x = _ensure_df(x)
    y = _ensure_df(y)
    out = x.rolling(window, min_periods=window).corr(y)
    return out.replace([np.inf, -np.inf], np.nan)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling covariance between *x* and *y* over a window of given length.
    """
    x = _ensure_df(x)
    y = _ensure_df(y)
    out = x.rolling(window, min_periods=window).cov(y)
    return out.replace([np.inf, -np.inf], np.nan)


def decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linear weighted moving average.  The most recent observation has
    weight ``window`` and the oldest weight ``1``.  When there are
    insufficient observations the function returns NaN.
    """
    df = _ensure_df(df)
    w = np.arange(1, window + 1, dtype=float)
    w /= w.sum()

    def _lwma(a: np.ndarray) -> float:
        return np.dot(a, w)

    return df.rolling(window, min_periods=window).apply(_lwma, raw=True)


def signed_power(x: pd.DataFrame, p: float) -> pd.DataFrame:
    """Compute the signed power of *x* raised to *p*, i.e. ``sign(x) *
    |x|**p``.  This avoids producing complex values when *x* contains
    negative numbers and *p* is non‑integer.
    """
    x = _ensure_df(x)
    return np.sign(x) * (np.abs(x) ** p)


###############################################################################
# Alpha101 class
###############################################################################

class Alpha101:
    """Container for computing the 101 formulaic alphas.

    Parameters
    ----------
    data : pandas.DataFrame
        A long‑format DataFrame containing at least the columns defined in the
        module docstring.  Additional columns will be ignored.

    Notes
    -----
    Upon initialisation the input data is pivoted into wide format and
    stored as instance attributes.  Rolling operators are computed on these
    wide matrices.  The ``industry_code`` column is pivoted to form a
    DataFrame used by the ``indneutralize`` operator for sector or
    industry neutralisation.
    """

    def __init__(self, data: pd.DataFrame):
        # ensure required columns are present
        required = {'code', 'date', 'open', 'high', 'low', 'close', 'volume',
                    'volume_amount', 'industry_code'}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        # sort by date to ensure chronological ordering
        df = data.copy()
        df = df.sort_values(['date', 'code'])

        # pivot each field to wide format; index=date, columns=code
        self.open = df.pivot(index='date', columns='code', values='open')
        self.high = df.pivot(index='date', columns='code', values='high')
        self.low = df.pivot(index='date', columns='code', values='low')
        self.close = df.pivot(index='date', columns='code', values='close')
        self.volume = df.pivot(index='date', columns='code', values='volume')
        self.volume_amount = df.pivot(index='date', columns='code',
                                      values='volume_amount')
        self.cap = df.pivot(index='date', columns='code', values='cap')
        # compute vwap; avoid divide by zero
        self.vwap = self.volume_amount / (self.volume.replace(0, np.nan))
        # compute close‑to‑close returns
        self.returns = self.close.pct_change()
        # create industry matrix for neutralisation; we forward fill to cover
        # cases where classifications change or are missing on some dates
        industry = df.pivot(index='date', columns='code', values='industry_code')
        self.industry = industry.ffill().bfill()

    # -------------------------------------------------------------------------
    # Alpha definitions
    #
    # Each method below computes one of the 101 alphas and returns a
    # DataFrame of the same shape as the underlying price matrices.  Where
    # appropriate intermediate results are cleaned to remove infinite and
    # missing values.
    # -------------------------------------------------------------------------

    def alpha001(self) -> pd.DataFrame:
        # (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close),2), 5)) - 0.5)
        base = self.close.copy()
        # replace positive returns with rolling stddev of returns where returns < 0
        mask = safe_lt(self.returns, 0)
        rep = ts_stddev(self.returns, 20)
        base = base.where(~mask, rep)
        val = signed_power(base, 2)
        return cs_rank(ts_argmax(val, 5))

    def alpha002(self) -> pd.DataFrame:
        # -corr(rank(delta(log(volume), 2)), rank((close - open) / open), 6)
        log_vol = np.log(self.volume.replace(0, np.nan))
        x = cs_rank(delta(log_vol, 2))
        y = cs_rank((self.close - self.open) / self.open)
        out = -correlation(x, y, 6)
        return out.fillna(0.0)

    def alpha003(self) -> pd.DataFrame:
        # -corr(rank(open), rank(volume), 10)
        out = -correlation(cs_rank(self.open), cs_rank(self.volume), 10)
        return out.fillna(0.0)

    # def alpha004(self) -> pd.DataFrame:
        ## 分布很奇怪
    #     # -Ts_Rank(rank(low), 9)
    #     return -ts_rank(cs_rank(self.low), 9)

    def alpha005(self) -> pd.DataFrame:
        # rank((open - average(vwap,10))) * (-abs(rank(close - vwap)))
        vwap_ma10 = ts_sum(self.vwap, 10) / 10.0
        term1 = cs_rank(self.open - vwap_ma10)
        term2 = -cs_rank(self.close - self.vwap).abs()
        return term1 * term2

    def alpha006(self) -> pd.DataFrame:
        # -corr(open, volume, 10)
        return -correlation(self.open, self.volume, 10).fillna(0.0)

    def alpha007(self) -> pd.DataFrame:
        # (adv20 < volume) ? [-ts_rank(|Δ7 close|,60) * sign(Δ7 close)] : -1
        adv20 = ts_mean(self.volume, 20)
        d7 = delta(self.close, 7)
        core = -ts_rank(d7.abs(), 60) * np.sign(d7)
        out = core.where(safe_lt(adv20, self.volume), -1.0)
        return out

    def alpha008(self) -> pd.DataFrame:
        # -rank((sum(open,5) * sum(returns,5)) - delay(sum(open,5) * sum(returns,5), 10))
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -cs_rank(x - delay(x, 10))

    def alpha009(self) -> pd.DataFrame:
        # If min5(Δ close) > 0 or max5(Δ close) < 0: Δ close else -Δ close
        d1 = delta(self.close, 1)
        cond_pos = safe_gt(ts_min(d1, 5), 0)
        cond_neg = safe_lt(ts_max(d1, 5), 0)
        base = -d1
        base = base.where(~(cond_pos | cond_neg), d1)
        return base

    def alpha010(self) -> pd.DataFrame:
        # rank of the conditional from alpha009 but window=4
        d1 = delta(self.close, 1)
        cond_pos = safe_gt(ts_min(d1, 4), 0)
        cond_neg = safe_lt(ts_max(d1, 4), 0)
        base = -d1
        base = base.where(~(cond_pos | cond_neg), d1)
        return cs_rank(base)

    def alpha011(self) -> pd.DataFrame:
        # (rank(ts_max(vwap - close,3)) + rank(ts_min(vwap - close,3))) * rank(delta(volume,3))
        diff = self.vwap - self.close
        term1 = cs_rank(ts_max(diff, 3))
        term2 = cs_rank(ts_min(diff, 3))
        term3 = cs_rank(delta(self.volume, 3))
        return (term1 + term2) * term3

    def alpha012(self) -> pd.DataFrame:
        # sign(delta(volume)) * (-delta(close))
        dv = delta(self.volume, 1)
        dc = delta(self.close, 1)
        return np.sign(dv) * (-dc)

    def alpha013(self) -> pd.DataFrame:
        # -rank(cov(rank(close), rank(volume), 5))
        x = cs_rank(self.close)
        y = cs_rank(self.volume)
        return -cs_rank(covariance(x, y, 5))

    def alpha014(self) -> pd.DataFrame:
        # -rank(delta(returns,3)) * corr(open, volume,10)
        corr = correlation(self.open, self.volume, 10).fillna(0.0)
        return -cs_rank(delta(self.returns, 3)) * corr

    def alpha015(self) -> pd.DataFrame:
        # -sum3(rank(correlation(rank(high), rank(volume),3)))
        x = cs_rank(self.high)
        y = cs_rank(self.volume)
        corr = correlation(x, y, 3).fillna(0.0)
        return -ts_sum(cs_rank(corr), 3)

    def alpha016(self) -> pd.DataFrame:
        # -rank(covariance(rank(high), rank(volume),5))
        return -cs_rank(covariance(cs_rank(self.high), cs_rank(self.volume), 5))

    def alpha017(self) -> pd.DataFrame:
        # -rank(ts_rank(close,10)) * rank(delta(delta(close),1)) * rank(ts_rank(volume/adv20,5))
        adv20 = ts_mean(self.volume, 20)
        term1 = cs_rank(ts_rank(self.close, 10))
        term2 = cs_rank(delta(delta(self.close, 1), 1))
        term3 = cs_rank(ts_rank(self.volume / adv20, 5))
        return -term1 * term2 * term3

    def alpha018(self) -> pd.DataFrame:
        # -rank(stddev(|close - open|,5) + (close - open) + corr(close,open,10))
        diff = self.close - self.open
        vol = ts_stddev(diff.abs(), 5)
        corr = correlation(self.close, self.open, 10).fillna(0.0)
        return -cs_rank(vol + diff + corr)

    def alpha019(self) -> pd.DataFrame:
        # -sign((close - delay(close,7)) + delta(close,7)) * (1 + rank(1 + sum(returns,250)))
        part = (self.close - delay(self.close, 7)) + delta(self.close, 7)
        sgn = -np.sign(part)
        cumret = ts_sum(self.returns, 250)
        return sgn * (1.0 + cs_rank(1.0 + cumret))

    def alpha020(self) -> pd.DataFrame:
        # -rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))
        return -cs_rank(self.open - delay(self.high, 1)) * \
               cs_rank(self.open - delay(self.close, 1)) * \
               cs_rank(self.open - delay(self.low, 1))

    def alpha021(self) -> pd.DataFrame:
        # complex conditional based on moving averages and volatility
        ma8 = ts_mean(self.close, 8)
        ma2 = ts_mean(self.close, 2)
        vol8 = ts_stddev(self.close, 8)
        cond1 = safe_lt(ma8 + vol8, ma2)
        # if adv20/vol < 1 then -1; else 1
        adv20 = ts_mean(self.volume, 20)
        cond2 = safe_lt(adv20 / self.volume, 1)
        # start with +1, flip to -1 when either condition holds
        out = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        out = out.where(~(cond1 | cond2), -1.0)
        return out

    def alpha022(self) -> pd.DataFrame:
        # -delta(corr(high,volume,5),5) * rank(stddev(close,20))
        corr = correlation(self.high, self.volume, 5).fillna(0.0)
        return -delta(corr, 5) * cs_rank(ts_stddev(self.close, 20))

    def alpha023(self) -> pd.DataFrame:
        # ((mean(high,20) < high) ? -delta(high,2) : 0)
        ma = ts_mean(self.high, 20)
        cond = safe_lt(ma, self.high)
        val = -delta(self.high, 2)
        out = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        return out.where(~cond, val.fillna(0.0))

    def alpha024(self) -> pd.DataFrame:
        # conditional on long horizon trend
        ma100 = ts_mean(self.close, 100)
        cond = safe_le(delta(ma100, 100) / delay(self.close, 100), 0.05)
        part = -delta(self.close, 3)
        alt = -(self.close - ts_min(self.close, 100))
        return part.where(~cond, alt)

    def alpha025(self) -> pd.DataFrame:
        # rank(((-returns) * adv20 * vwap * (high - close)))
        adv20 = ts_mean(self.volume, 20)
        return cs_rank((-self.returns) * adv20 * self.vwap * (self.high - self.close))

    def alpha026(self) -> pd.DataFrame:
        # -ts_max(corr(ts_rank(volume,5), ts_rank(high,5),5),3)
        corr = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5).fillna(0.0)
        return -ts_max(corr, 3)

    def alpha027(self) -> pd.DataFrame:
        # sign flip based on correlation strength
        temp = ts_mean(correlation(cs_rank(self.volume), cs_rank(self.vwap), 6).fillna(0.0), 2) / 2.0
        out = cs_rank(temp)
        # if > 0.5 assign -1 else 1
        res = pd.DataFrame(np.ones_like(out), index=out.index, columns=out.columns)
        res = res.where(safe_le(out, 0.5), -1.0)
        return res

    def alpha028(self) -> pd.DataFrame:
        # scale(corr(adv20,low,5) + ((high+low)/2) - close)
        adv20 = ts_mean(self.volume, 20)
        corr = correlation(adv20, self.low, 5).fillna(0.0)
        val = (corr + (self.high + self.low) / 2) - self.close
        return scale(val)

    def alpha029(self) -> pd.DataFrame:
        """Alpha 29.

        Due to the extreme complexity of the original specification, this
        implementation uses a simplified proxy: a minimum of nested
        cross‑sectional ranks of rolling differences in close prices,
        combined with a time‑series rank of lagged negative returns.
        """
        part1 = ts_min(cs_rank(cs_rank(-cs_rank(delta(self.close, 5)))), 5)
        part2 = ts_rank(delay(-self.returns, 6), 5)
        return part1 + part2

    def alpha030(self) -> pd.DataFrame:
        # (1 - rank(sign(Δ close + sign(delay(Δ close)) + sign(delay^2(Δ close)))))*sum(vol,5)/sum(vol,20)
        d = delta(self.close, 1)
        inner = np.sign(d) + np.sign(delay(d, 1)) + np.sign(delay(d, 2))
        num = (1.0 - cs_rank(inner)) * ts_sum(self.volume, 5)
        den = ts_sum(self.volume, 20)
        return num / den.replace(0.0, np.nan)

    def alpha031(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        corr = correlation(adv20, self.low, 12).fillna(0.0)
        inner = -cs_rank(cs_rank(delta(self.close, 10)))
        p1 = cs_rank(cs_rank(cs_rank(decay_linear(inner, 10))))
        p2 = cs_rank(-delta(self.close, 3))
        p3 = np.sign(scale(corr))
        return p1 + p2 + p3

    def alpha032(self) -> pd.DataFrame:
        part1 = scale((ts_mean(self.close, 7) - self.close) / 7)
        part2 = 20 * scale(correlation(self.vwap, delay(self.close, 5), 230).fillna(0.0))
        return part1 + part2

    def alpha033(self) -> pd.DataFrame:
        return cs_rank(-1 + (self.open / self.close))

    def alpha034(self) -> pd.DataFrame:
        term = ts_stddev(self.returns, 2) / ts_stddev(self.returns, 5)
        term = term.replace([np.inf, -np.inf], 1).fillna(1)
        return cs_rank(2 - cs_rank(term) - cs_rank(delta(self.close, 1)))

    def alpha035(self) -> pd.DataFrame:
        return (ts_rank(self.volume, 32) * (1 - ts_rank(self.close + self.high - self.low, 16)) *
                (1 - ts_rank(self.returns, 32)))

    def alpha036(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        term1 = 2.21 * cs_rank(correlation(self.close - self.open, delay(self.volume, 1), 15).fillna(0.0))
        term2 = 0.7 * cs_rank(self.open - self.close)
        term3 = 0.73 * cs_rank(ts_rank(delay(-self.returns, 6), 5))
        term4 = cs_rank(abs(correlation(self.vwap, adv20, 6)))
        term5 = 0.6 * cs_rank(((ts_mean(self.close, 200) / 200) - self.open) * (self.close - self.open))
        return term1 + term2 + term3 + term4 + term5

    def alpha037(self) -> pd.DataFrame:
        corr = correlation(delay(self.open - self.close, 1), self.close, 200).fillna(0.0)
        return cs_rank(corr) + cs_rank(self.open - self.close)

    def alpha038(self) -> pd.DataFrame:
        ratio = (self.close / self.open).replace([np.inf, -np.inf], 1).fillna(1)
        return -cs_rank(ts_rank(self.open, 10)) * cs_rank(ratio)

    def alpha039(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        term = delta(self.close, 7) * (1 - cs_rank(decay_linear(self.volume / adv20, 9)))
        part = -cs_rank(term)
        return part * (1 + cs_rank(ts_mean(self.returns, 250)))

    def alpha040(self) -> pd.DataFrame:
        return -cs_rank(ts_stddev(self.high, 10)) * correlation(self.high, self.volume, 10).fillna(0.0)

    def alpha041(self) -> pd.DataFrame:
        return np.sqrt(self.high * self.low) - self.vwap

    # def alpha042(self) -> pd.DataFrame:
    #     ## 分布很奇怪
    #     return cs_rank(self.vwap - self.close) / cs_rank(self.vwap + self.close).replace(0.0, np.nan)

    def alpha043(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank(-delta(self.close, 7), 8)

    def alpha044(self) -> pd.DataFrame:
        return -correlation(self.high, cs_rank(self.volume), 5).fillna(0.0)

    def alpha045(self) -> pd.DataFrame:
        part1 = cs_rank(ts_mean(delay(self.close, 5), 20))
        corr = correlation(self.close, self.volume, 2).fillna(0.0)
        part3 = cs_rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        return -part1 * corr * part3

    def alpha046(self) -> pd.DataFrame:
        inner = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        base = -delta(self.close, 1)
        res = base.copy()
        res = res.where(~safe_lt(inner, 0), 1.0)
        res = res.where(~safe_gt(inner, 0.25), -1.0)
        return res

    def alpha047(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        part1 = (cs_rank(1 / self.close) * self.volume) / adv20
        part2 = (self.high * cs_rank(self.high - self.close)) / (ts_mean(self.high, 5) / 5)
        return part1 * part2 - cs_rank(self.vwap - delay(self.vwap, 5))

    def alpha048(self) -> pd.DataFrame:
        """Alpha 48.

        Implements a sector/subindustry neutralised ratio of the correlation
        between successive differences of the close price.  The numerator
        takes the 250‑day rolling correlation between ``delta(close,1)`` and
        ``delta(delay(close,1),1)``, multiplies by ``delta(close,1)`` and
        divides by ``close``, then neutralises by industry.  The
        denominator sums the squared ratio of ``delta(close,1)`` and its
        lagged value over 250 days.  A small epsilon prevents division by
        zero.
        """
        d1 = delta(self.close, 1)
        d1_delayed = delta(delay(self.close, 1), 1)
        corr = correlation(d1, d1_delayed, 250).fillna(0.0)
        num = corr * d1 / self.close.replace(0.0, np.nan)
        num = indneutralize(num, self.industry)
        denom = ts_sum((d1 / delay(self.close, 1).replace(0.0, np.nan)).pow(2), 250)
        return num / denom.replace(0.0, np.nan)

    def alpha049(self) -> pd.DataFrame:
        inner = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        base = -delta(self.close, 1)
        res = base.copy()
        res = res.where(~safe_lt(inner, -0.1), 1.0)
        return res

    def alpha050(self) -> pd.DataFrame:
        return -ts_max(cs_rank(correlation(cs_rank(self.volume), cs_rank(self.vwap), 5).fillna(0.0)), 5)

    def alpha051(self) -> pd.DataFrame:
        inner = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        base = -delta(self.close, 1)
        res = base.copy()
        res = res.where(~safe_lt(inner, -0.05), 1.0)
        return res

    def alpha052(self) -> pd.DataFrame:
        part1 = -delta(ts_min(self.low, 5), 5)
        part2 = cs_rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        part3 = ts_rank(self.volume, 5)
        return part1 * part2 * part3

    def alpha053(self) -> pd.DataFrame:
        denom = (self.close - self.low).replace(0.0, 1e-5)
        return -delta(((self.close - self.low) - (self.high - self.close)) / denom, 9)

    def alpha054(self) -> pd.DataFrame:
        denom = (self.low - self.high).replace(0.0, -1e-5)
        return -((self.low - self.close) * (self.open ** 5) / (denom * (self.close ** 5)))

    def alpha055(self) -> pd.DataFrame:
        denom = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0.0, 1e-5)
        inner = (self.close - ts_min(self.low, 12)) / denom
        return -correlation(cs_rank(inner), cs_rank(self.volume), 6).fillna(0.0)

    def alpha056(self) -> pd.DataFrame:
        """Alpha 56.

        Uses market capitalisation to weight returns.  The formula is

        ``-1 * (rank(sum(returns,10) / sum(sum(returns,2),3)) * rank(returns * cap))``.

        Here we interpret ``sum(sum(returns,2),3)`` as a 3‑period rolling sum
        of a 2‑period rolling sum of returns.  If ``cap`` is missing the
        computation will produce NaNs.
        """
        if self.cap.isnull().all().all():
            # no cap data available
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        part1 = ts_sum(self.returns, 10) / ts_sum(ts_sum(self.returns, 2), 3)
        part2 = self.returns * self.cap
        return -(cs_rank(part1) * cs_rank(part2))

    def alpha057(self) -> pd.DataFrame:
        denom = decay_linear(cs_rank(ts_argmax(self.close, 30)), 2)
        return -(self.close - self.vwap) / denom

    def alpha058(self) -> pd.DataFrame:
        """Alpha 58.

        ``-Ts_Rank(decay_linear(correlation(IndNeutralize(vwap), volume, ~4), ~8), ~6)``

        The windows derived from the original paper (3.92795, 7.89291,
        5.50322) have been rounded to 4, 8 and 6 respectively.  This alpha
        measures the rate of change of the volume–VWAP correlation,
        neutralised by industry.
        """
        x = indneutralize(self.vwap, self.industry)
        corr = correlation(x, self.volume, 4).fillna(0.0)
        dec = decay_linear(corr, 8)
        return -ts_rank(dec, 6)

    def alpha059(self) -> pd.DataFrame:
        """Alpha 59.

        ``-Ts_Rank(decay_linear(correlation(IndNeutralize(vwap), volume, ~4), ~16), ~8)``

        This alpha is similar to Alpha58 but uses a longer decay and
        ranking horizon.  The numerical windows 4.25197, 16.2289 and
        8.19648 are rounded to 4, 16 and 8.
        """
        x = indneutralize(self.vwap, self.industry)
        corr = correlation(x, self.volume, 4).fillna(0.0)
        dec = decay_linear(corr, 16)
        return -ts_rank(dec, 8)

    def alpha060(self) -> pd.DataFrame:
        denom = (self.high - self.low).replace(0.0, 1e-5)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / denom
        return -((2 * scale(cs_rank(inner))) - scale(cs_rank(ts_argmax(self.close, 10))))

    def alpha061(self) -> pd.DataFrame:
        adv180 = ts_mean(self.volume, 180)
        left = cs_rank(self.vwap - ts_min(self.vwap, 16))
        right = cs_rank(correlation(self.vwap, adv180, 18))
        cond = safe_lt(left, right)
        return cond.astype(float)

    def alpha062(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        left = cs_rank(correlation(self.vwap, ts_mean(adv20, 22), 10))
        inner = safe_lt((cs_rank(self.open) + cs_rank(self.open)), (cs_rank((self.high + self.low) / 2) + cs_rank(self.high)))
        right = cs_rank(inner)
        cond = safe_lt(left, right).astype(float)
        return cond * -1

    def alpha063(self) -> pd.DataFrame:
        """Alpha 63.

        A difference between two ranked decayed series.  The first term
        looks at the change in industry‑neutralised close prices, while the
        second term uses a correlation between a weighted combination of
        VWAP/open and a long horizon average volume.
        """
        # term1: rank(decay_linear(delta(IndNeutralize(close),2),8))
        term1 = cs_rank(decay_linear(delta(indneutralize(self.close, self.industry), 2), 8))
        # term2: rank(decay_linear(correlation((vwap*0.318108 + open*(1-0.318108)), sum(adv180,37), 14),12))
        adv180 = ts_mean(self.volume, 180)
        combined = self.vwap * 0.318108 + self.open * (1 - 0.318108)
        corr = correlation(combined, ts_sum(adv180, 37), 14).fillna(0.0)
        term2 = cs_rank(decay_linear(corr, 12))
        return (term1 - term2) * -1

    def alpha064(self) -> pd.DataFrame:
        adv120 = ts_mean(self.volume, 120)
        left = cs_rank(correlation(ts_mean((self.open * 0.178404 + self.low * (1 - 0.178404)), 13), ts_mean(adv120, 13), 17))
        right = cs_rank(delta(((self.high + self.low) / 2 * 0.178404 + self.vwap * (1 - 0.178404)), 4))
        cond = safe_lt(left, right).astype(float)
        return cond * -1

    def alpha065(self) -> pd.DataFrame:
        adv60 = ts_mean(self.volume, 60)
        left = cs_rank(correlation(self.open * 0.00817205 + self.vwap * (1 - 0.00817205), ts_mean(adv60, 9), 6))
        right = cs_rank(self.open - ts_min(self.open, 14))
        cond = safe_lt(left, right).astype(float)
        return cond * -1

    def alpha066(self) -> pd.DataFrame:
        term1 = cs_rank(decay_linear(delta(self.vwap, 4), 7))
        num = ((self.low * 0.96633 + self.low * (1 - 0.96633)) - self.vwap)
        denom = (self.open - (self.high + self.low) / 2).replace(0.0, 1e-5)
        term2 = ts_rank(decay_linear(num / denom, 11), 7)
        return (term1 + term2) * -1

    def alpha067(self) -> pd.DataFrame:
        """Alpha 67.

        Combines the difference between current highs and recent minima with
        a correlation between industry‑neutralised VWAP and neutralised
        average volume.  The exponentiation of ranks captures non‑linear
        effects.  Windows have been rounded appropriately.
        """
        term1 = cs_rank(self.high - ts_min(self.high, 2))
        x = indneutralize(self.vwap, self.industry)
        y = indneutralize(ts_mean(self.volume, 20), self.industry)
        corr = correlation(x, y, 6).fillna(0.0)
        term2 = cs_rank(corr)
        return (term1.pow(term2)) * -1

    # def alpha068(self) -> pd.DataFrame:
    ## right幾乎不會變動
    #     adv15 = ts_mean(self.volume, 15)
    #     left = ts_rank(correlation(cs_rank(self.high), cs_rank(adv15), 9), 14)
    #     right = cs_rank(delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1))
    #     cond = safe_lt(left, right).astype(float)
    #     return cond * -1

    def alpha069(self) -> pd.DataFrame:
        """Alpha 69.

        Uses the maximum of industry‑neutralised VWAP changes and the
        correlation of a weighted close/VWAP combination with average
        volume.  Fractional windows have been rounded to [3,5] and [5,9].
        """
        term1 = cs_rank(ts_max(delta(indneutralize(self.vwap, self.industry), 3), 5))
        adv20 = ts_mean(self.volume, 20)
        combined = self.close * 0.490655 + self.vwap * (1 - 0.490655)
        corr = correlation(combined, adv20, 5).fillna(0.0)
        term2 = ts_rank(corr, 9)
        return (term1.pow(term2)) * -1
    def alpha070(self) -> pd.DataFrame:
        """Alpha 70.

        Ranks the change in VWAP and raises it to the power of a ranked
        correlation between industry‑neutralised close and average volume.
        Windows rounded to 1 and 18 respectively.
        """
        term1 = cs_rank(delta(self.vwap, 1))
        adv50 = ts_mean(self.volume, 50)
        corr = correlation(indneutralize(self.close, self.industry), adv50, 18).fillna(0.0)
        term2 = ts_rank(corr, 18)
        return (term1.pow(term2)) * -1

    def alpha071(self) -> pd.DataFrame:
        adv180 = ts_mean(self.volume, 180)
        corr1 = correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18).fillna(0.0)
        p1 = ts_rank(decay_linear(corr1, 4), 16)
        base = cs_rank(((self.low + self.open) - (self.vwap + self.vwap))).pow(2)
        p2 = ts_rank(decay_linear(base, 16), 4)
        cond = safe_ge(p1, p2)
        return p1.where(cond, p2)

    def alpha072(self) -> pd.DataFrame:
        adv40 = ts_mean(self.volume, 40)
        corr1 = correlation((self.high + self.low) / 2, adv40, 9).fillna(0.0)
        num = cs_rank(decay_linear(corr1, 10))
        corr2 = correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7).fillna(0.0)
        den = cs_rank(decay_linear(corr2, 3))
        return num / den.replace(0.0, np.nan)

    def alpha073(self) -> pd.DataFrame:
        p1 = cs_rank(decay_linear(delta(self.vwap, 5), 3))
        ratio = (delta(self.open * 0.147155 + self.low * (1 - 0.147155), 2) /
                 (self.open * 0.147155 + self.low * (1 - 0.147155))) * -1
        p2 = ts_rank(decay_linear(ratio, 3), 17)
        cond = safe_ge(p1, p2)
        return -(p1.where(cond, p2))

    # def alpha074(self) -> pd.DataFrame:
        ## left幾乎不會變動
    #     adv30 = ts_mean(self.volume, 30)
    #     left = cs_rank(correlation(self.close, ts_mean(adv30, 37), 15))
    #     right = cs_rank(correlation(cs_rank(self.high * 0.0261661 + self.vwap * (1 - 0.0261661)), cs_rank(self.volume), 11))
    #     cond = safe_lt(left, right).astype(float)
    #     return cond * -1

    # def alpha075(self) -> pd.DataFrame:
        ## left幾乎不會變動
    #     adv50 = ts_mean(self.volume, 50)
    #     left = cs_rank(correlation(self.vwap, self.volume, 4))
    #     right = cs_rank(correlation(cs_rank(self.low), cs_rank(adv50), 12))
    #     cond = safe_lt(left, right).astype(float)
    #     return cond

    def alpha076(self) -> pd.DataFrame:
        """Alpha 76.

        Maximum of a decayed VWAP change and a complex ranking of
        correlations between neutralised low prices and long horizon
        average volumes.  Windows rounded to [1,12], [8,20] and [17,19].
        """
        adv81 = ts_mean(self.volume, 81)
        # part1: rank(decay_linear(delta(vwap,1),12))
        p1 = cs_rank(decay_linear(delta(self.vwap, 1), 12))
        # part2: Ts_Rank(decay_linear(Ts_Rank(corr(IndNeutralize(low), adv81,8),20),17),19)
        corr = correlation(indneutralize(self.low, self.industry), adv81, 8).fillna(0.0)
        inner = ts_rank(corr, 20)
        dec_inner = decay_linear(inner, 17)
        p2 = ts_rank(dec_inner, 19)
        cond = safe_ge(p1, p2)
        out = p1.where(cond, p2)
        return out * -1

    def alpha077(self) -> pd.DataFrame:
        adv40 = ts_mean(self.volume, 40)
        p1 = cs_rank(decay_linear((((self.high + self.low) / 2 + self.high) - (self.vwap + self.high)), 20))
        corr = correlation((self.high + self.low) / 2, adv40, 3).fillna(0.0)
        p2 = cs_rank(decay_linear(corr, 6))
        cond = safe_le(p1, p2)
        return p1.where(cond, p2)

    def alpha078(self) -> pd.DataFrame:
        adv40 = ts_mean(self.volume, 40)
        part1 = cs_rank(correlation(ts_sum(self.low * 0.352233 + self.vwap * (1 - 0.352233), 20), ts_sum(adv40, 20), 7).fillna(0.0))
        part2 = cs_rank(correlation(cs_rank(self.vwap), cs_rank(self.volume), 6).fillna(0.0))
        return part1.pow(part2)

    def alpha079(self) -> pd.DataFrame:
        """Alpha 79.

        Compares the change in a weighted close/open combination (after
        industry neutralisation) to the correlation of ranked VWAP and
        ranked long horizon average volume.  Windows rounded for
        implementation.
        """
        weighted = self.close * 0.60733 + self.open * (1 - 0.60733)
        term1 = cs_rank(delta(indneutralize(weighted, self.industry), 1))
        adv150 = ts_mean(self.volume, 150)
        term2 = cs_rank(correlation(ts_rank(self.vwap, 4), ts_rank(adv150, 9), 15))
        cond = safe_lt(term1, term2).astype(float)
        return cond
    
    def alpha080(self) -> pd.DataFrame:
        """Alpha 80.

        Raises the rank of the sign of an industry‑neutralised open/high
        combination to the power of a ranked correlation between high
        prices and short horizon average volume.  Windows rounded.
        """
        combo = self.open * 0.868128 + self.high * (1 - 0.868128)
        term1 = cs_rank(np.sign(delta(indneutralize(combo, self.industry), 4)))
        adv10 = ts_mean(self.volume, 10)
        term2 = ts_rank(correlation(self.high, adv10, 5).fillna(0.0), 6)
        return term1.pow(term2) * -1

    def alpha081(self) -> pd.DataFrame:
        adv10 = ts_mean(self.volume, 10)
        part1 = cs_rank(np.log(ts_sum(cs_rank((cs_rank(correlation(self.vwap, ts_sum(adv10, 50), 8).fillna(0.0)).pow(4))), 15)))
        part2 = cs_rank(correlation(cs_rank(self.vwap), cs_rank(self.volume), 5).fillna(0.0))
        cond = safe_lt(part1, part2).astype(float)
        return cond * -1

    def alpha082(self) -> pd.DataFrame:
        """Alpha 82.

        Minimum of two decayed series: one based on open price changes and
        another on the correlation of industry‑neutralised volume with the
        open price.  Windows rounded accordingly.
        """
        term1 = cs_rank(decay_linear(delta(self.open, 1), 15))
        x = indneutralize(self.volume, self.industry)
        y = self.open
        corr = correlation(x, y, 17).fillna(0.0)
        term2 = ts_rank(decay_linear(corr, 7), 13)
        cond = safe_le(term1, term2)
        return (term1.where(cond, term2)) * -1

    def alpha083(self) -> pd.DataFrame:
        num = cs_rank(delay((self.high - self.low) / (ts_sum(self.close, 5) / 5), 2)) * cs_rank(cs_rank(self.volume))
        den = ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close).replace(0.0, np.nan)
        return num / den.replace(0.0, np.nan)

    def alpha084(self) -> pd.DataFrame:
        # Alpha84 會有 Overflow 的問題
        alpha = alpha = signed_power(ts_rank(self.vwap - ts_max(self.vwap, 15), 21), delta(self.close, 5))
        # mask 出 inf
        mask_inf = np.isinf(alpha)
        # row-wise 最大有限值
        row_max = alpha.replace([np.inf, -np.inf], np.nan).max(axis=1)
        # 利用 DataFrame.broadcast 自然對齊 row → columns
        alpha = alpha.mask(mask_inf, row_max, axis=0)
        return alpha


    def alpha085(self) -> pd.DataFrame:
        adv30 = ts_mean(self.volume, 30)
        left = cs_rank(correlation(self.high * 0.876703 + self.close * (1 - 0.876703), adv30, 10).fillna(0.0))
        right = cs_rank(correlation(ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7).fillna(0.0))
        return left.pow(right)

    def alpha086(self) -> pd.DataFrame:
        adv20 = ts_mean(self.volume, 20)
        left = ts_rank(correlation(self.close, ts_mean(adv20, 15), 6).fillna(0.0), 20)
        right = cs_rank((self.open + self.close) - (self.vwap + self.open))
        cond = safe_lt(left, right).astype(float)
        return cond * -1

    def alpha087(self) -> pd.DataFrame:
        """Alpha 87.

        Maximum of two ranked series: one based on VWAP changes and the
        other on the absolute correlation between neutralised average volume
        and close prices.  Windows rounded.
        """
        term1 = cs_rank(decay_linear(delta(self.close * 0.369701 + self.vwap * (1 - 0.369701), 2), 3))
        adv81 = ts_mean(self.volume, 81)
        corr = abs(correlation(indneutralize(adv81, self.industry), self.close, 13).fillna(0.0))
        term2 = ts_rank(decay_linear(corr, 5), 14)
        cond = safe_ge(term1, term2)
        out = term1.where(cond, term2)
        return out * -1

    def alpha088(self) -> pd.DataFrame:
        adv60 = ts_mean(self.volume, 60)
        p1 = cs_rank(decay_linear((cs_rank(self.open) + cs_rank(self.low) - cs_rank(self.high) - cs_rank(self.close)), 8))
        corr = correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8).fillna(0.0)
        p2 = ts_rank(decay_linear(corr, 7), 3)
        cond = safe_le(p1, p2)
        return p1.where(cond, p2)

    def alpha089(self) -> pd.DataFrame:
        """Alpha 89.

        Difference of two ranked decayed series.  The first component is
        based on the correlation of low prices with a short horizon
        average volume; the second on industry‑neutralised VWAP changes.
        Windows rounded.
        """
        adv10 = ts_mean(self.volume, 10)
        corr = correlation(self.low, adv10, 7).fillna(0.0)
        term1 = ts_rank(decay_linear(corr, 6), 4)
        term2 = ts_rank(decay_linear(delta(indneutralize(self.vwap, self.industry), 3), 10), 15)
        return term1 - term2
    def alpha090(self) -> pd.DataFrame:
        """Alpha 90.

        Uses the difference between close prices and recent maxima and
        correlates industry‑neutralised average volume with low prices.
        Windows rounded to 5 and 3.
        """
        term1 = cs_rank(self.close - ts_max(self.close, 5))
        adv40 = ts_mean(self.volume, 40)
        corr = correlation(indneutralize(adv40, self.industry), self.low, 5).fillna(0.0)
        term2 = ts_rank(corr, 3)
        return term1.pow(term2) * -1
    def alpha091(self) -> pd.DataFrame:
        """Alpha 91.

        Subtracts a cross‑sectional rank of a decayed correlation between
        VWAP and volume from a complex time‑series rank of nested decayed
        correlations.  Windows rounded.
        """
        x = indneutralize(self.close, self.industry)
        corr1 = correlation(x, self.volume, 10).fillna(0.0)
        d1 = decay_linear(corr1, 16)
        d2 = decay_linear(d1, 4)
        part1 = ts_rank(d2, 5)
        adv30 = ts_mean(self.volume, 30)
        corr2 = correlation(self.vwap, adv30, 4).fillna(0.0)
        part2 = cs_rank(decay_linear(corr2, 3))
        return (part1 - part2) * -1

    def alpha092(self) -> pd.DataFrame:
        adv30 = ts_mean(self.volume, 30)
        cond = safe_lt(((self.high + self.low) / 2 + self.close), (self.low + self.open))
        p1 = ts_rank(decay_linear(cond, 15), 19)
        corr = correlation(cs_rank(self.low), cs_rank(adv30), 8).fillna(0.0)
        p2 = ts_rank(decay_linear(corr, 7), 7)
        cond_p = safe_le(p1, p2)
        return p1.where(cond_p, p2)

    def alpha093(self) -> pd.DataFrame:
        """Alpha 93.

        Ratio of a time‑series rank of a decayed correlation between
        neutralised VWAP and long horizon average volume to the cross‑
        sectional rank of a decayed change in a weighted close/VWAP
        combination.  Windows rounded.
        """
        adv81 = ts_mean(self.volume, 81)
        corr = correlation(indneutralize(self.vwap, self.industry), adv81, 17).fillna(0.0)
        part1 = ts_rank(decay_linear(corr, 20), 8)
        combo = self.close * 0.524434 + self.vwap * (1 - 0.524434)
        part2 = cs_rank(decay_linear(delta(combo, 3), 16))
        return part1 / part2.replace(0.0, np.nan)

    def alpha094(self) -> pd.DataFrame:
        adv60 = ts_mean(self.volume, 60)
        term1 = cs_rank(self.vwap - ts_min(self.vwap, 12))
        term2 = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18).fillna(0.0), 3)
        return term1.pow(term2) * -1

    def alpha095(self) -> pd.DataFrame:
        adv40 = ts_mean(self.volume, 40)
        left = cs_rank(self.open - ts_min(self.open, 12))
        right = ts_rank(cs_rank(correlation(ts_mean((self.high + self.low) / 2, 19), ts_mean(adv40, 19), 13).fillna(0.0)).pow(5), 12)
        cond = safe_lt(left, right).astype(float)
        return cond

    def alpha096(self) -> pd.DataFrame:
        adv60 = ts_mean(self.volume, 60)
        corr1 = correlation(cs_rank(self.vwap), cs_rank(self.volume), 4).fillna(0.0)
        p1 = ts_rank(decay_linear(corr1, 4), 8)
        corr2 = correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4).fillna(0.0)
        arg = ts_argmax(corr2, 13)
        p2 = ts_rank(decay_linear(arg, 14), 13)
        cond = safe_ge(p1, p2)
        return -(p1.where(cond, p2))

    def alpha097(self) -> pd.DataFrame:
        """Alpha 97.

        Difference between two ranked series.  The first term looks at the
        decayed change in a weighted low/VWAP combination (neutralised by
        industry); the second term examines nested decayed correlations of
        ranked low prices and ranked long horizon average volume.  Windows
        rounded.
        """
        combo = self.low * 0.721001 + self.vwap * (1 - 0.721001)
        term1 = cs_rank(decay_linear(delta(indneutralize(combo, self.industry), 3), 20))
        adv60 = ts_mean(self.volume, 60)
        corr = correlation(ts_rank(self.low, 8), ts_rank(adv60, 17), 5).fillna(0.0)
        inner = ts_rank(corr, 19)
        dec = decay_linear(inner, 16)
        term2 = ts_rank(dec, 7)
        return (term1 - term2) * -1

    def alpha098(self) -> pd.DataFrame:
        adv5 = ts_mean(self.volume, 5)
        adv15 = ts_mean(self.volume, 15)
        corr1 = correlation(self.vwap, ts_mean(adv5, 26), 5).fillna(0.0)
        left = cs_rank(decay_linear(corr1, 7))
        corr2 = correlation(cs_rank(self.open), cs_rank(adv15), 21).fillna(0.0)
        arg = ts_argmin(corr2, 9)
        right = cs_rank(decay_linear(ts_rank(arg, 7), 8))
        return left - right

    def alpha099(self) -> pd.DataFrame:
        adv60 = ts_mean(self.volume, 60)
        left = cs_rank(correlation(ts_sum((self.high + self.low) / 2, 20), ts_sum(adv60, 20), 9).fillna(0.0))
        right = cs_rank(correlation(self.low, self.volume, 6).fillna(0.0))
        cond = safe_lt(left, right).astype(float)
        return cond * -1

    def alpha100(self) -> pd.DataFrame:
        """Alpha 100.

        A complex combination of two scaled, industry‑neutralised terms.
        The first term is based on the rank of a ratio involving the
        distance between close and low relative to the day’s range and
        multiplied by volume.  The second term combines a correlation
        between close and ranked average volume with the rank of where
        close stands relative to its 30‑day minimum.  Finally, the result
        is scaled by volume/adv20 and negated.
        """
        adv20 = ts_mean(self.volume, 20)
        # first component: ((close-low) - (high-close))/(high-low) * volume
        denom = (self.high - self.low).replace(0.0, 1e-5)
        ratio = (((self.close - self.low) - (self.high - self.close)) / denom) * self.volume
        x1 = cs_rank(ratio)
        x1n = indneutralize(indneutralize(x1, self.industry), self.industry)
        term1 = 1.5 * scale(x1n)
        # second component: corr(close, rank(adv20),5) - rank(ts_argmin(close,30))
        corr = correlation(self.close, cs_rank(adv20), 5).fillna(0.0) - cs_rank(ts_argmin(self.close, 30))
        x2n = indneutralize(corr, self.industry)
        term2 = scale(x2n)
        result = -(term1 - term2) * (self.volume / adv20.replace(0.0, np.nan))
        return result

    def alpha101(self) -> pd.DataFrame:
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
    
    def alpha102(self) -> pd.DataFrame:
        """Volume Momentum 20d: (V_t - V_{t-20}) / V_{t-20}"""
        v = self.volume
        return div(v - delay(v, 20), delay(v, 20))

    def alpha103(self) -> pd.DataFrame:
        """Volume Z-Score 20d"""
        v = self.volume
        return (v - ts_mean(v, 20)) / ts_stddev(v, 20).replace(0.0, np.nan)

    def alpha104(self) -> pd.DataFrame:
        """Price-Volume Correlation 20d"""
        return correlation(self.close, self.volume, 20)

    def alpha105(self) -> pd.DataFrame:
        """Volume Shock: V / mean(V,20)"""
        v = self.volume
        return div(v, ts_mean(v, 20))

    def alpha106(self) -> pd.DataFrame:
        """Volume-based Reversal: -rank(V) * rank(Close,5)"""
        v_rank = ts_rank(self.volume, 20)
        c_rank = ts_rank(self.close, 5)
        return -v_rank * c_rank

    def alpha107(self) -> pd.DataFrame:
        """Volume Rank Momentum: rank(V,20) - delay(rank(V,20),5)"""
        vr = ts_rank(self.volume, 20)
        return vr - delay(vr, 5)

    def alpha108(self) -> pd.DataFrame:
        """Short-term Volume Momentum 3d"""
        v = self.volume
        return div(v - delay(v, 3), delay(v, 3))

    def alpha109(self) -> pd.DataFrame:
        """Volume Spike Strength: V / min(V,20)"""
        v = self.volume
        return div(v, ts_min(v, 20))

    def alpha110(self) -> pd.DataFrame:
        """Volume Divergence: rank(C,20) - rank(V,20)"""
        return ts_rank(self.close, 20) - ts_rank(self.volume, 20)

    def alpha111(self) -> pd.DataFrame:
        """Volume Stickiness: min(V,15)/max(V,15)"""
        v = self.volume
        return div(ts_min(v, 15), ts_max(v, 15))

    # ---------------- Tier 2 量能因子 ----------------

    def alpha112(self) -> pd.DataFrame:
        """Volume Percentile Rank 20d"""
        return ts_rank(self.volume, 20)

    def alpha113(self) -> pd.DataFrame:
        """Volume Volatility 20d"""
        return ts_stddev(self.volume, 20)

    def alpha114(self) -> pd.DataFrame:
        """Turnover Acceleration: Δ(ΔV_5,5)"""
        v = self.volume
        return delta(delta(v, 5), 5)

    def alpha115(self) -> pd.DataFrame:
        """Volume Spike Ratio: V / mean(V,40)"""
        v = self.volume
        return div(v, ts_mean(v, 40))

    def alpha116(self) -> pd.DataFrame:
        """Volume Skew proxy: (mean-min)/(max-min)"""
        v = self.volume
        v_min = ts_min(v, 20)
        v_max = ts_max(v, 20)
        v_mean = ts_mean(v, 20)
        denom = (v_max - v_min).replace(0.0, np.nan)
        return (v_mean - v_min) / denom

    def alpha117(self) -> pd.DataFrame:
        """Volume Spike After Quiet Period"""
        v = self.volume
        cond1 = safe_gt(v, ts_mean(v, 60))
        cond2 = safe_lt(ts_stddev(v, 20), ts_stddev(v, 60))
        cond = cond1 & cond2
        return cond.astype(float)

    def alpha118(self) -> pd.DataFrame:
        """Volume Shock Persistence: #days V>2*mean(V,20) in 10d"""
        v = self.volume
        cond = safe_gt(v, ts_mean(v, 20) * 2)
        return ts_sum(cond.astype(float), 10)

    def alpha119(self) -> pd.DataFrame:
        """Volume Volatility Ratio: std(V,10)/std(V,50)"""
        v = self.volume
        return div(ts_stddev(v, 10), ts_stddev(v, 50))

    def alpha120(self) -> pd.DataFrame:
        """Relative Volume Zscore 60d"""
        v = self.volume
        return (v - ts_mean(v, 60)) / ts_stddev(v, 60).replace(0.0, np.nan)

    def alpha121(self) -> pd.DataFrame:
        """Smoothed Volume Trend: linear decay 20d"""
        return decay_linear(self.volume, 20)

    # ---------------- 其他 Volume 因子 ----------------

    def alpha122(self) -> pd.DataFrame:
        """Long-term Volume Momentum 60d"""
        v = self.volume
        return div(v - delay(v, 60), delay(v, 60))

    def alpha123(self) -> pd.DataFrame:
        """Volume Trend Strength: mean(V,20)/mean(V,60)"""
        v = self.volume
        return div(ts_mean(v, 20), ts_mean(v, 60))

    def alpha124(self) -> pd.DataFrame:
        """Volume MA Ribbon Slope: MA10 - MA30"""
        v = self.volume
        return ts_mean(v, 10) - ts_mean(v, 30)

    def alpha125(self) -> pd.DataFrame:
        """Volume MACD-style: MA12 - MA26"""
        v = self.volume
        return ts_mean(v, 12) - ts_mean(v, 26)

    def alpha126(self) -> pd.DataFrame:
        """Volume Trend Persistence: fraction of up-volume days in 20d"""
        v = self.volume
        up = safe_gt(v, delay(v, 1)).astype(float)
        return ts_mean(up, 20)

    def alpha127(self) -> pd.DataFrame:
        """Volume Change Rate 1d"""
        v = self.volume
        return div(delta(v, 1), delay(v, 1))

    def alpha128(self) -> pd.DataFrame:
        """Volume vs rolling median 20d"""
        v = self.volume
        med = v.rolling(20, min_periods=20).median()
        return div(v, med)

    # def alpha129(self) -> pd.DataFrame:
    #         ## 分布很奇怪
    #     """Extreme Volume Percentile > 95% in 60d"""
    #     cond = safe_gt(ts_rank(self.volume, 60), 0.95)
    #     return cond.astype(float)

    def alpha130(self) -> pd.DataFrame:
        """Quiet -> Violent Volume Switch"""
        v = self.volume
        cond = safe_lt(ts_stddev(v, 40), ts_stddev(v, 10)).astype(float)
        return cond * ts_rank(v, 10)

    def alpha131(self) -> pd.DataFrame:
        """Volume / std(V,20)"""
        v = self.volume
        return div(v, ts_stddev(v, 20))

    def alpha132(self) -> pd.DataFrame:
        """Short-term avg volume / 20d min"""
        v = self.volume
        return div(ts_mean(v, 5), ts_min(v, 20))

    def alpha133(self) -> pd.DataFrame:
        """Cross-sectional Volume Rank"""
        return cs_rank(self.volume)

    def alpha134(self) -> pd.DataFrame:
        """Industry-neutral Volume"""
        return indneutralize(self.volume, self.industry)

    def alpha135(self) -> pd.DataFrame:
        """Argmax of Volume over 20d"""
        return ts_argmax(self.volume, 20)

    def alpha136(self) -> pd.DataFrame:
        """Argmin of Volume over 20d"""
        return ts_argmin(self.volume, 20)

    def alpha137(self) -> pd.DataFrame:
        """Volume Range Ratio: (max-min)/mean"""
        v = self.volume
        rng = ts_max(v, 20) - ts_min(v, 20)
        return div(rng, ts_mean(v, 20))

    def alpha138(self) -> pd.DataFrame:
        """Volume Entropy proxy: std/mean"""
        v = self.volume
        return div(ts_stddev(v, 20), ts_mean(v, 20))

    def alpha139(self) -> pd.DataFrame:
        """Volume SMA10 / SMA50"""
        v = self.volume
        return div(ts_mean(v, 10), ts_mean(v, 50))

    def alpha140(self) -> pd.DataFrame:
        """Volume SMA20 / SMA5"""
        v = self.volume
        return div(ts_mean(v, 20), ts_mean(v, 5))

    def alpha141(self) -> pd.DataFrame:
        """Volume deviation from 60d mean"""
        v = self.volume
        return v - ts_mean(v, 60)

    def alpha142(self) -> pd.DataFrame:
        """Volume^2 normalized by mean"""
        v = self.volume
        return div(v ** 2, ts_mean(v, 20))

    def alpha143(self) -> pd.DataFrame:
        """Log Volume Change"""
        v1 = self.volume
        v0 = delay(self.volume, 1)
        ratio = div(v1, v0)
        return log(ratio)

    def alpha144(self) -> pd.DataFrame:
        """Rank of Volume Volatility"""
        vv = ts_stddev(self.volume, 20)
        return ts_rank(vv, 20)

    def alpha145(self) -> pd.DataFrame:
        """Volume Momentum Normalized by std"""
        v = self.volume
        return div(delta(v, 5), ts_stddev(v, 20))

    def alpha146(self) -> pd.DataFrame:
        """Volume-weighted Argmax on Price"""
        return ts_argmax(self.close * self.volume, 20)

    def alpha147(self) -> pd.DataFrame:
        """Volume-weighted 5d return"""
        return delta(self.close, 5) * self.volume

    def alpha148(self) -> pd.DataFrame:
        """High Volume After Price Drop"""
        v_rank = ts_rank(self.volume, 20)
        drop = safe_lt(self.close, delay(self.close, 5)).astype(float)
        return v_rank * drop

    def alpha149(self) -> pd.DataFrame:
        """Price-Volume Corr Change"""
        corr_now = correlation(self.close, self.volume, 20)
        corr_past = delay(corr_now, 20)
        return corr_now - corr_past

    def alpha150(self) -> pd.DataFrame:
        """Down Volume Pressure"""
        v_rank = ts_rank(self.volume, 20)
        down = safe_lt(self.close, delay(self.close, 1)).astype(float)
        return v_rank * down

    def alpha151(self) -> pd.DataFrame:
        """Short vs Long Volume mean ratio: MA5/MA30"""
        v = self.volume
        return div(ts_mean(v, 5), ts_mean(v, 30))

    def alpha152(self) -> pd.DataFrame:
        """3-tier Volume Trend: MA5+MA20+MA60"""
        v = self.volume
        return ts_mean(v, 5) + ts_mean(v, 20) + ts_mean(v, 60)

    # def alpha153(self) -> pd.DataFrame:
    #         ## 分布很奇怪
    #     """Regime-dependent Volume (high vol price regime)"""
    #     high_vol = safe_gt(ts_stddev(self.close, 20), ts_stddev(self.close, 60)).astype(float)
    #     return high_vol * ts_rank(self.volume, 20)

    def alpha154(self) -> pd.DataFrame:
        """Compression -> Expansion Volume Signal"""
        v = self.volume
        cond = safe_lt(ts_stddev(v, 20), ts_stddev(v, 60)).astype(float)
        return cond * ts_rank(v, 10)

    def alpha155(self) -> pd.DataFrame:
        """Volume Volatility Spike: std10 / delay(std10,10)"""
        v = self.volume
        s10 = ts_stddev(v, 10)
        return div(s10, delay(s10, 10))

    def alpha156(self) -> pd.DataFrame:
        """Volume Compression Ratio: std20/mean20"""
        v = self.volume
        return div(ts_stddev(v, 20), ts_mean(v, 20))

    # def alpha157(self) -> pd.DataFrame:
    #         ## 分布很奇怪
    #     """Volume Climax vs 40d max"""
    #     v = self.volume
    #     cond = safe_eq(v, ts_max(v, 40)).astype(float)
    #     return cond

    def alpha158(self) -> pd.DataFrame:
        """Volume relative to cap"""
        # 注意：若 self.cap 可能為 0，這裡與 volume 無關，但一併保護
        return div(self.volume, self.cap)

    def alpha159(self) -> pd.DataFrame:
        """Volume / rolling sum 20d"""
        v = self.volume
        return div(v, ts_sum(v, 20))

    def alpha160(self) -> pd.DataFrame:
        """Volume shock vs median/std"""
        v = self.volume
        med = v.rolling(20, min_periods=20).median()
        return div(v - med, ts_stddev(v, 20))

    def alpha161(self) -> pd.DataFrame:
        """Volume CS outlier: V / cs_rank(V)"""
        v = self.volume
        r = cs_rank(v)
        # cs_rank 最小為 1/n，不會是 0，與 volume 無關
        return div(v, r)

    def alpha162(self) -> pd.DataFrame:
        """Volume-Return interaction: 5d return * rank(V)"""
        ret5 = delta(self.close, 5)
        vr = ts_rank(self.volume, 20)
        return ret5 * vr

    def alpha163(self) -> pd.DataFrame:
        """Volume Quietness indicator"""
        v = self.volume
        cond = safe_lt(ts_stddev(v, 20), ts_stddev(v, 60)).astype(float)
        return cond

    def alpha164(self) -> pd.DataFrame:
        """Volume Activity Level: MA10"""
        return ts_mean(self.volume, 10)

    def alpha165(self) -> pd.DataFrame:
        """Volume high-tail frequency in 20d"""
        v = self.volume
        cond = safe_gt(v, ts_mean(v, 20) * 1.5).astype(float)
        return ts_mean(cond, 20)

    def alpha166(self) -> pd.DataFrame:
        """5d return * Volume Shock"""
        v = self.volume
        shock = div(v, ts_mean(v, 20))
        return delta(self.close, 5) * shock

    def alpha167(self) -> pd.DataFrame:
        """Average of rank(V,10) and rank(C,10)"""
        vr = ts_rank(self.volume, 10)
        cr = ts_rank(self.close, 10)
        return 0.5 * (vr + cr)

    def alpha168(self) -> pd.DataFrame:
        """Sign of corr(Volume, Returns, 20d)"""
        corr_vr = correlation(self.volume, self.returns, 20)
        return np.sign(corr_vr)

    def alpha169(self) -> pd.DataFrame:
        """Covariance(Close, Volume, 20d)"""
        return covariance(self.close, self.volume, 20)

    def alpha170(self) -> pd.DataFrame:
        """Normalized Covariance: cov(C,V,20)/std(V,20)"""
        v = self.volume
        cov_cv = covariance(self.close, v, 20)
        return div(cov_cv, ts_stddev(v, 20))

    def alpha171(self) -> pd.DataFrame:
        """Volume regime flip: std10>std50"""
        v = self.volume
        cond = safe_gt(ts_stddev(v, 10), ts_stddev(v, 50)).astype(float)
        return cond
