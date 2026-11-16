"""A tiny stats module with three bugs, used to demo the coding agent."""


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / (len(values) - 1)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty list")
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def is_prime(n: int) -> bool:
    if n < 1:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True
