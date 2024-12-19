import os
from brat_parser import get_entities_relations_attributes_groups


def fix_jumps(txt_filepath: str) -> str:
    """ Replace content of given txt file fixing \n characters at end of in-sentence line"""
    # Read file lines in list
    with open(txt_filepath, "r") as f:
        textlines = f.readlines()

    # diagnostic flags
    hasHyphenJump = False
    hasAbbrvLineStart = False
    hasComposedWord = False

    # Modify each line
    for i in range(len(textlines)-1):
        # boolean variables
        nextLineStartsWithLower = textlines[i+1][0].islower()
        lineEndsWithJump = textlines[i].endswith("\n")
        lineLinkedToNext = textlines[i].endswith("-\n")
        nextLineStartsWithNumber = textlines[i+1][0].isdigit()
        nextLineStartsWithAbbrv = textlines[i+1][0:2].isupper()
        nextLineStartsWithHyph = textlines[i+1][0] == '-'
        nextLineStartsWithSpecial = textlines[i+1][0] == ("(" or "," or ".")

        if lineLinkedToNext and nextLineStartsWithLower:
            # remove -\n in this line so that the linked last word is made
            # \n counts for 1 character in python
            textlines[i] = textlines[i][:-2]
            hasHyphenJump = True
        elif lineEndsWithJump and (nextLineStartsWithLower or nextLineStartsWithNumber or nextLineStartsWithAbbrv or nextLineStartsWithSpecial):
            # remove line jump and add space at the end
            textlines[i] = textlines[i][:-1] + " "
            hasAbbrvLineStart = nextLineStartsWithAbbrv
        elif lineEndsWithJump and nextLineStartsWithHyph:
            textlines[i] = textlines[i][-1]
            hasComposedWord = True

    fixed_text = "".join(textlines)

    return fixed_text, hasHyphenJump, hasAbbrvLineStart, hasComposedWord


def fix_directory(txt_dir):
    for file in os.listdir(txt_dir):
        if file.endswith(".txt"):
            filepath = f'{txt_dir}/{file}'
            fixed_text, _, _, _ = fix_jumps(filepath)
            with open(filepath, 'w') as f:
                f.write(fixed_text)


def dir_diagnostic(txt_dir):
    diag_dict = {
        "hyphen_jump": [],
        "abbrv_line_start": [],
        "composed_words": [],
        "discontinuous_ent": []
    }
    for file in os.listdir(txt_dir):
        filepath = f'{txt_dir}/{file}'
        if file.endswith(".txt"):
            _, hasHyphenJump, hasAbbrvLineStart, hasComposedWord = fix_jumps(
                filepath)
            if hasHyphenJump:
                diag_dict["hyphen_jump"].append(file)
            if hasAbbrvLineStart:
                diag_dict["abbrv_line_start"].append(file)
            if hasComposedWord:
                diag_dict["composed_words"].append(file)
        elif file.endswith(".ann"):
            # check for discontinuous entities
            entities, _, _, _ = get_entities_relations_attributes_groups(
                filepath)
            for ent in entities.values():
                spans = ent.span
                if len(spans) > 1:
                    diag_dict["discontinuous_ent"].append(file)
                    break
    print(diag_dict)


if __name__ == "__main__":
    ann_dir = "ann1_fixed"
    mode = "diag"

    if mode == "diag":
        dir_diagnostic(ann_dir)

    elif mode == "fix":
        fix_directory(ann_dir)
