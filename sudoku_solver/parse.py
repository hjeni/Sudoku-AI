import itertools


def convert_safe(value, t: type = int):
    """
    Tries to convert value to specified type
    :param value: Value to be converted
    :param t: type of conversion (int is default)
    :return: converted value when successful, None otherwise
    """
    try:
        return t(value)
    except (ValueError, TypeError):
        return None


def _try_convert_seq(text: str, t: type = int, dml: str = None) -> list:
    """
    Tries to convert values separated by delimiter to one common type
    :param text: string of values separated by the delimiter
    :param t: type of conversion (int is default)
    :param dml: delimiter (any whitespace is default)
    :return: list of converted elements (each element either converted or None when the conversion was not successful)
    """
    # split single elements
    if dml is None:
        split = [x for x in text.split() if x != '']
    else:
        split = [x for x in text.split(dml) if x != '']
    # convert each value separately
    return list(map(convert_safe, split, itertools.repeat(t)))


