from typing import List


def rle_pack(s: str) -> str:
    if not s:
        return ""

    result: List[str] = []
    current_char = s[0]
    count = 1

    for ch in s[1:]:
        if ch == current_char:
            count += 1
        else:
            result.append(current_char + str(count))
            current_char = ch
            count = 1

    result.append(current_char + str(count))

    return "".join(result)


def main() -> None:
    """
    Читает строку с stdin, выводит RLE-представление.
    """
    s = input().rstrip("\n")
    print(rle_pack(s))


if __name__ == "__main__":
    main()
