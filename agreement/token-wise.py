from brat_parser import get_entities_relations_attributes_groups, Entity
from sklearn.metrics import cohen_kappa_score
from os.path import join, isfile
from typing import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from annotated_token import TextToken
from argparse import ArgumentParser


def get_entities(ann_filepath) -> List[Entity]:
    """Parse entities from brat .ann file into list of `brat_parser.Entity` objects"""
    return [v for v in get_entities_relations_attributes_groups(ann_filepath)[0].values()]


def create_token_annotations(entities: List[Entity], text: str) -> List[TextToken]:
    """Iterate through each character in text to fill the annotation list
    will split tokens on every non alphanumeric character
    if token is included in one of brat entities in the list,
    then it is assigned its label
    """
    # Initializations
    annot_tokens = []
    current_token = ""
    char_count = 0
    start_ind, end_ind = 0, 0
    # Iterating through each character in text
    for c in text:
        charNotAlnum = not c.isalnum()
        if charNotAlnum and current_token:  # Splitting here when special character and current token not empty
            end_ind = char_count
            text_token = TextToken(
                text=current_token, span=(start_ind, end_ind))
            text_token.set_label(entities)
            annot_tokens.append(text_token)
            current_token = ""
        elif charNotAlnum and not current_token:  # multiple adjacent spaces or carriage returns
            char_count += 1
            continue  # to not count 2 times
        elif not charNotAlnum and not current_token:  # first character of a token
            start_ind = char_count
            current_token += c
        else:  # middle characters of token
            current_token += c
        char_count += 1
    return annot_tokens


def parse_pair_files(filename: str, ann1_dir: str, ann2_dir: str) -> Tuple[List[TextToken], List[TextToken]]:
    """Parse brat files across 2 annotator (they must have the same id)

    Args:
        filename (int): name of the file (ann_dir/{name}.txt // ann_dir/{name}.ann)
        ann1_dir (str): brat directory of first annotator
        ann2_dir (str): brat directory of second annotator

    Returns:
        Tuple[str, List[TextToken], List[TextToken]]: list of text tokens for each annotator
    """
    # check that all files exists
    ann1_txt, ann1_ann = join(
        ann1_dir, filename + ".txt"), join(ann1_dir, filename + ".ann")
    ann2_txt, ann2_ann = join(
        ann2_dir, filename + ".txt"), join(ann2_dir, filename + ".txt")
    assert isfile(ann1_txt) and isfile(
        ann1_ann) and isfile(ann2_txt) and isfile(ann2_ann)
    # text files should be the same across both annotators
    with open(ann1_txt, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(ann2_txt, "r", encoding="utf-8") as f:
        text2 = f.read()
    assert text1 == text2
    # parsing
    entities1 = get_entities(ann1_ann)
    entities2 = get_entities(ann2_ann)
    text_tokens_1 = create_token_annotations(entities1, text1)
    text_tokens_2 = create_token_annotations(entities2, text2)
    return text_tokens_1, text_tokens_2


def get_strat(t1: TextToken, t2: TextToken, strategy: str) -> bool:
    """Get strategy boolean for current token pair  

    Returns:
        boolean : are these token taken in account according to strategy
    """
    if strategy == "all":
        return True
    elif strategy == "both-annot":
        return t1.label != "Unlabeled" and t2.label != "Unlabeled"
    elif strategy == "one-annot":
        return t1.label != "Unlabeled" or t2.label != "Unlabeled"
    else:
        raise ValueError(
            "Unknown strategy : valid values are 'all', 'both-annot', 'one-annot'")


def filter_annotations(text_tokens_1: List[TextToken], text_tokens_2: List[TextToken], label2id: Dict[str, int], strategy: str) -> Tuple[List[int], List[int]]:
    """Filter annotations according to strategy :
    - `all` : all tokens (containing tokens that are unannotated by both annotators)
    - `both-annot` : tokens that are annotated by both annotators (intersection of both annot)
    - `one-annot` : tokens that are annotated by at least one annotator (union of both annot)
    """
    ann1, ann2 = [], []
    for t1, t2 in zip(text_tokens_1, text_tokens_2):
        if get_strat(t1, t2, strategy):
            ann1.append(label2id[t1.label])
            ann2.append(label2id[t2.label])
    return ann1, ann2


def class_wise_kappa(all_ann1: str, all_ann2: str, label: str):
    pass


def brat_pair_cohen_kappa(annotator1_dir: str, annotator2_dir: str, label2id: Dict[str, int],
                          common_studies: List[Union[str, int]], strategy: str = "one-annot") -> Dict[str, float]:
    kappa = {}
    all_ann1, all_ann2 = [], []
    # TODO : behaviour if label2id is None // common_studies is None or empty
    # TODO : common studies check ?
    common_studies = [str(id) for id in common_studies]
    for document_id in common_studies:
        entities1, entities2 = parse_pair_files(
            document_id, annotator1_dir, annotator2_dir)
        ann1, ann2 = filter_annotations(
            entities1, entities2, label2id, strategy)
        assert len(ann1) == len(ann2)
        if ann1 == ann2:
            kappa[str(document_id)] = 1.0
        else:
            kappa[str(document_id)] = cohen_kappa_score(ann1, ann2)
        all_ann1.extend(ann1)
        all_ann2.extend(ann2)
    assert len(all_ann1) == len(all_ann2)
    kappa["all"] = cohen_kappa_score(all_ann1, all_ann2)
    return kappa


def get_labels(label2id: Dict[str, int]) -> List[str]:
    labels = [""] * len(label2id)
    for k, v in label2id.items():
        labels[v] = k
    return labels


def brat_pair_confusion_matrix(annotator1_dir: str, annotator2_dir: str, label2id: Dict[str, int],
                                  common_studies: List[Union[str, int]] = None, strategy: str = "one-annot") -> None:
    """Plot confusion matrix for a pair of annotators with brat entity annotations"""
    all_ann1, all_ann2 = [], []
    for document_id in common_studies:
        entities1, entities2 = parse_pair_files(
            document_id, annotator1_dir, annotator2_dir)
        ann1, ann2 = filter_annotations(
            entities1, entities2, label2id, strategy)
        assert len(ann1) == len(ann2)
        all_ann1.extend(ann1)
        all_ann2.extend(ann2)
    ConfusionMatrixDisplay.from_predictions(all_ann1, all_ann2)
    plt.ylabel("annotator1")
    plt.xlabel("annotator2")
    plt.show()


def main():
    parser = ArgumentParser()
    # TODO :  add main function parsing (which argument is optional ?)
    parser.add_argument(

    )

if __name__ == "__main__":
    ann1_dir = "brat_annotations/annotator1"
    ann2_dir = "brat_annotations/annotator2"
    common_studies = [2, 3, 4, 11, 12, 13, 14, 15, 16,
                      19, 23, 24, 26, 34, 37, 43, 45, 116, 117, 126]
    # TODO : load label2id/id2label from file (json ? or txt ? or both ?)
    label2id = {
        "PrimaryOutcome": 0,
        "SecondaryOutcome": 1,
        "OtherOutcome": 2,
        "TimeFrame": 3,
        "OutcomeDefinition": 4,
        "Unlabeled": 5
    }
    strategy = "all"
    kappa = brat_pair_cohen_kappa(
        ann1_dir, ann2_dir, label2id, common_studies, strategy)
    print(kappa)
    brat_pair_confusion_matrix(
        ann1_dir, ann2_dir, label2id, common_studies, strategy)
