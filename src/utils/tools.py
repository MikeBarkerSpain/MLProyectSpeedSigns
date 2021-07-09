import json
from collections import Counter

def read_json_to_dict(json_fullpath):
    """
    Read a json and return a object created from it.
    Args:
        json_fullpath: json fullpath

    Returns: json object.
    """
    try:
        with open(json_fullpath, 'r+') as outfile:
            json_readed = json.load(outfile)
        return json_readed
    except Exception as error:
        raise ValueError(error)

def most_frequent(List):
    if List[0] != List [1] and List[0] != List[2] and List[1] != List[2]:
        return List[1]
    else:
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]