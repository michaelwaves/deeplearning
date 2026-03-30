import math

from fractions import Fraction


def shannon_encode(probs: dict[str, float]) -> dict[str, str]:
    symbols = sorted(probs.items(), key=lambda x: -x[1])
    codes = {}
    cumulative = 0.0
    for symbol, prob in symbols:
        length = math.ceil(math.log2(1/prob)) if prob < 1 else 1
        codeword = _float_to_binary(cumulative, length)
        codes[symbol] = codeword
        cumulative += prob
    return codes


def _float_to_binary(value: float, bits: int) -> str:
    result = []
    for _ in range(bits):
        value *= 2
        bit = int(value)
        result.append(str(bit))
        value -= bit

    return "".join(result)


if __name__ == "__main__":
    probs = {"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.120, "E": 0.005}
    print(shannon_encode(probs))
