import numpy as np

from scipy.optimize import brentq

from typing import Tuple, Any, List


def price(yld: float, maturity: int, cpn: float, cpn_per_year: int = 2, maturity_value: int = 100) -> float:
    payments:int = maturity * cpn_per_year
    n:List[float] = [(1 / cpn_per_year) * x for x in range(1, payments + 1)]
    cf:List[float] = [(cpn / cpn_per_year) * maturity_value for x in n]
    cf[-1] += maturity_value
    dcf:List[float] = [cf[i] / ((1 + yld) ** x) for i, x in enumerate(n)]
    return sum(dcf)


def price_dist(yld: float, maturity: int, cpn: float, target_price: float, cpn_per_year: int = 2,
               maturity_value: int = 100) -> float:
    return target_price - price(yld,maturity,cpn,cpn_per_year,maturity_value)


def ytm(price: float, maturity: int, cpn: float, cpn_per_year: int = 2, maturity_value: int = 100) -> Tuple[float,Any]:
    return brentq(price_dist,a=-cpn,b=2*cpn,args=(maturity,cpn,price))

