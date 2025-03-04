import sys


def scorer(new: list[float, int], old: list[float, int]) -> float:
    """
    Calculate the updated score based on the old score and the new score.
    """
    return (new[0] * new[1] + old[0] * old[1]) / (new[1] + old[1])


if __name__ != "__main__":
    __all__ = ['scorer']
