def merge_dicts(dict_list):
    merged = {}
    for d in dict_list:
        for key, value in d.items():
            merged.setdefault(key, []).append(value)  # Collect values as lists
    return merged
