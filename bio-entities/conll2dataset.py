import os
from datasets import Dataset, Features, Value, Sequence, ClassLabel, DatasetDict
from typing import List, Dict, Tuple, Union


def parse_conll_file(dirpath: str, fname: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """ Read .conll file and parse its content into 3 lists containing, ids, tokens and labels
    The file will be split between each sentence (where there is a carriage return) 

    Args:
        dirpath (str): path to directory containing .conll file
        fname (str): file name in directory

    Returns:
        Tuple[List[str], List[List[str]], List[List[str]]]: list of ids (format : filename_sentencenumber), list of list of tokens, list of list of labels
    """
    # read file
    assert fname.split(".")[1] == "conll"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        fcontent = f.read()
    fid = fname.split(".")[0]
    # parse file content
    sents_ids = []
    sents_tokens = []
    sents_labels = []
    sentences = fcontent.split('\n\n')
    for scount, s in enumerate(sentences):
        if s: # line can be empty (line jump)
            stokens = []
            slabels = []
            for token_line in s.split("\n"):
                token, label = token_line.split("\t")
                stokens.append(token)
                slabels.append(label)
            sid = fid + "_" + str(scount)
            sents_ids.append(sid)
            sents_tokens.append(stokens)
            sents_labels.append(slabels)
            assert len(sents_tokens) == len(sents_labels)
    assert len(sents_ids) == len(sents_tokens) == len(sents_labels)
    return sents_ids, sents_tokens, sents_labels


def parse_conll_dir(dirpath: str) -> Dict[str, Union[str, List[List[str]]]]:
    """Parse all conll files in a directory into a dictionary

    Args:
        dirpath (str): path to directory containing all conll files

    Returns:
        Dict[str, Union[str, List[List[str]]]]: dictionary containing list of ids, list of list tokens and list of list labels (for every files in a directory)
    """
    out_dict = {
        "id": [],
        "tokens": [],
        "labels": [],
    }
    for fname in os.listdir(dirpath):
        if fname.endswith(".conll"):
            sents_ids, sents_tokens, sents_labels = parse_conll_file(dirpath, fname)
            out_dict["id"] += sents_ids
            out_dict["tokens"] += sents_tokens
            out_dict["labels"] += sents_labels
    return out_dict


def get_unique_mapping(list_of_list: List[List[str]]) -> List[str]:
    """Get unique elements in list of list of string 
    and return a sorted list mapping (list indice = key) of unique elements

    Args:
        list_of_list (List[List[str]]): list of list of string containing all possible values

    Returns:
        List[str]: list ordered alphabetically containing all unique values 
    """
    flat_list = sum(list_of_list, [])
    unique = set(flat_list)
    id2label = []
    for u in sorted(unique):
        id2label.append(u)
    return id2label


def create_mapping_file(list_of_list: List[List[str]], mapping_filepath: str) -> None:
    """Create mapping file in working dir (for each line one unique label ordered ) 
    if it does not already exists

    Args:
        list_of_list (List[List[str]]): list of list of labels that you want to get the mapping from
        mapping_filepath (str): output file path where the unique mapping will be stored
    """
    if not os.path.exists(mapping_filepath):
        id2label = get_unique_mapping(list_of_list)
        id2label = [l + '\n' for l in id2label]
        with open(mapping_filepath, "w") as f:
            f.writelines(id2label)


def conlldict2dataset(conll_dict: Dict[str, Union[str, List[List[str]]]], mapping_filepath: str) -> Dataset:
    """Convert a dictionary of parsed conll files to a HuggingFace `datasets.Dataset` object

    Args:
        conll_dict (Dict[str, Union[str, List[List[str]]]]): dictionary containing ids, tokens and labels
        mapping_filepath (str): output file path for labels mapping

    Returns:
        Dataset: `datasets.Dataset` object containing ids, labels and tokens
    """
    create_mapping_file(conll_dict["labels"], mapping_filepath)
    features = Features({
        "id": Value('string'),
        "labels": Sequence(ClassLabel(names_file=mapping_filepath)),
        "tokens": Sequence(Value('string')),
    })
    return Dataset.from_dict(conll_dict, features=features)


def conll2dataset(conll_dir: str, mapping_filepath: str, save: bool=False, save_path: str=None) -> Dataset:
    """Convert conll files in specified directory to a `datasets.Dataset` object

    Args:
        conll_dir (str): directory containing .conll files
        mapping_filepath (str): file path to the labels mapping (one label per line, in the desired order)

    Returns:
        Dataset: object containing all the converted files
    """
    dataset = conlldict2dataset(parse_conll_dir(conll_dir), mapping_filepath)
    if save and save_path:
        dataset.save_to_disk(save_path)
    elif save and not save_path:
        dataset.save_to_disk()
    return dataset


def conll2datasetdict(split_mapping: Dict[str, str], mapping_filepath: str="labels.txt", save: bool=False, save_path: str=None) -> DatasetDict:
    """Given a mapping of split names -> split conll directory path, convert
    the files in each directory to a `datasets.DatasetDict` object

    Args:
        split_mapping (Dict[str, str]): mapping of split names (keys) to their associated directory (values) containing .conll files
        mapping_filepath (str, optional):  mapping of labels file path (will be created if does not exist). Defaults to "labels.txt".

    Returns:
        DatasetDict: object containing the datasets from each split
    """
    dataset_dict = {}
    for split_name, split_conll_dir in split_mapping.items():
        dataset_dict[split_name] = conll2dataset(split_conll_dir, mapping_filepath)
    return DatasetDict(dataset_dict)

