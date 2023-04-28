from nltk.tree import *

nonterminals = {
    "S": [
        ["NP", "VP"],
        ["X1", "VP"],
        ["VB", "NP"],
        ["X2", "PP"],
        ["VB", "VP"],
        ["VP", "PP"],
    ],
    "X1": [["Aux", "NP"]],
    "NP": [["Det", "Nom"]],
    "Nom": [["Nom", "NN"], ["Nom", "PP"]],
    "VP": [["VB", "NP"], ["X2", "PP"], ["VB", "PP"], ["VP", "PP"]],
    "X2": [["VB", "PP"]],
    "PP": [["Prep", "NP"]],
}
terminals = {
    # terminals from conversion
    "S": ["book", "include", "prefer", "read", "left"],
    "NP": ["I", "she", "me", "Houston", "United"],
    "Nom": ["book", "flight", "meal", "money", "morning"],
    "VP": ["book", "include", "prefer", "read", "left"],
    # terminals from original lexical entries
    "Det": ["that", "this", "the", "a"],
    "NN": ["book", "flight", "meal", "money", "morning"],
    "NNP": ["Houston", "United"],
    "Prep": ["from", "to", "on", "near", "through"],
    "Pron": ["I", "she", "me"],
    "VB": ["book", "include", "prefer", "read", "left"],
    "Aux": ["does"],
}


def parse(
    words: list[str],
    grammar_nonterminals: dict,
    grammar_terminals: dict,
    verbose: bool = False,
) -> list[list[str]]:
    # CKY parse table
    table = [[list() for _ in words] for _ in words]
    # pointer to build the parse tree
    pointers = [[list() for _ in words] for _ in words]

    # log progress if verbose is True
    if verbose:
        log = lambda x: print(x)
    log = lambda x: x

    # iterate through all words in the sentence
    for j, word in enumerate(words):
        # check whether the word is in the grammar
        for t in grammar_terminals:
            # only checking terminal grammars
            if word in grammar_terminals[t] or word.lower() in grammar_terminals[t]:
                table[j][j].append(t)
                pointers[j][j] = word  # nonterminal node

        # fill the upper diagonal
        for i in range(j - 1, 0 - 1, -1):
            log(f"filling table[{i}][{j}]")
            # fill table[i,j] with comparing table[i,k] and table[k+1,j]
            # where k is from i from j-1
            for k in range(i, j):
                log(f"  comparing table[{i}][{k}] and table[{k+1}][{j}]")
                for nt in grammar_nonterminals:
                    for rm in grammar_nonterminals[nt]:
                        log(
                            f"    checking {rm} table[{i}][{k}]={table[i][k]} \
                                and table[{k+1}][{j}]={table[k + 1][j]}"
                        )
                        # check whether nt -> table[i,k] table[k+1,j] exists
                        if rm[0] in table[i][k] and rm[1] in table[k + 1][j]:
                            log(f"      added '{nt}' to table[{i}][{j}]")
                            table[i][j].append(nt)
                            pointers[i][j].append(((i, k), (k + 1, j)))  # terminal node

    return table, pointers


def trace(table, pointers, row, col):
    if "S" not in table[0][-1]:
        raise ValueError("Table is not parseable")

    pointer = pointers[row][col]

    if isinstance(pointer, str):
        # nonterminal node
        return Tree(table[row][col][0], [pointer])

    return Tree(
        table[row][col][0],
        [
            trace(table, pointers, pointer[0][0][0], pointer[0][0][1]),
            trace(table, pointers, pointer[0][1][0], pointer[0][1][1]),
        ],
    )


for s in [
    "I read a book",
    "Book the flight through Houston",
    "Does she prefer the morning flight",
]:
    t, p = parse(s.split(), nonterminals, terminals)
    parseable = "S" in t[0][-1]

    print(f"sentence='{s}'")
    print("table=")
    print("\t".join(s.split()))

    # recognizing
    print("\n".join(["\t".join([",".join(c) for c in r]) for r in t]))
    print(f"parseable = {parseable}")
    print()

    # parsing
    tree = trace(t, p, 0, -1)
    print(f"tree={str(tree).replace('(', '[').replace(')', ']')}")
    print("tree=")
    tree.pretty_print(unicodelines=True)
    print()
