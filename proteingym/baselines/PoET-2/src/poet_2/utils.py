import hashlib


def hash_of_string_list(lst: list[str]) -> str:
    m = hashlib.sha1()
    for elt in lst:
        m.update(elt.encode("utf-8"))
    return m.hexdigest()
