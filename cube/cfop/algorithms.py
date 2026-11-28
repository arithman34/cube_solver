OLL: dict[str, str] = {
    "OLL 1": "R U2 R2 F R F' U2 R' F R F'",
    "OLL 2": "r U r' U2 r U2 R' U2 R U' r'",
    "OLL 3": "r' R2 U R' U r U2 r' U M'",
    "OLL 4": "M U' r U2 r' U' R U' R' M'",
    "OLL 5": "l' U2 L U L' U l",
    "OLL 6": "r U2 R' U' R U' r'",
    "OLL 7": "r U R' U R U2 r'",
    "OLL 8": "l' U' L U' L' U2 l",
    "OLL 9": "R U R' U' R' F R2 U R' U' F'",
    "OLL 10": "R U R' U R' F R F' R U2 R'",
    "OLL 11": "r U R' U R' F R F' R U2 r'",
    "OLL 12": "M' R' U' R U' R' U2 R U' R r'",
    "OLL 13": "F U R U' R2 F' R U R U' R'",
    "OLL 14": "R' F R U R' F' R F U' F'",
    "OLL 15": "l' U' l L' U' L U l' U l",
    "OLL 16": "r U r' R U R' U' r U' r'",
    "OLL 17": "F R' F' R2 r' U R U' R' U' M'",
    "OLL 18": "r U R' U R U2 r2 U' R U' R' U2 r",
    "OLL 19": "r' R U R U R' U' M' R' F R F'",
    "OLL 20": "r U R' U' M2 U R U' R' U' M'",
    "OLL 21": "R U2 R' U' R U R' U' R U' R'",
    "OLL 22": "R U2 R2 U' R2 U' R2 U2 R",
    "OLL 23": "R2 D' R U2 R' D R U2 R",
    "OLL 24": "r U R' U' r' F R F'",
    "OLL 25": "F' r U R' U' r' F R",
    "OLL 26": "R U2 R' U' R U' R'",
    "OLL 27": "R U R' U R U2 R'",
    "OLL 28": "r U R' U' r' R U R U' R'",
    "OLL 29": "R U R' U' R U' R' F' U' F R U R'",
    "OLL 30": "F R' F R2 U' R' U' R U R' F2",
    "OLL 31": "R' U' F U R U' R' F' R",
    "OLL 32": "L U F' U' L' U L F L'",
    "OLL 33": "R U R' U' R' F R F'",
    "OLL 34": "R U R2 U' R' F R U R U' F'",
    "OLL 35": "R U2 R2 F R F' R U2 R'",
    "OLL 36": "L' U' L U' L' U L U L F' L' F",
    "OLL 37": "F R' F' R U R U' R'",
    "OLL 38": "R U R' U R U' R' U' R' F R F'",
    "OLL 39": "L F' L' U' L U F U' L'",
    "OLL 40": "R' F R U R' U' F' U R",
    "OLL 41": "R U R' U R U2 R' F R U R' U' F'",
    "OLL 42": "R' U' R U' R' U2 R F R U R' U' F'",
    "OLL 43": "F' U' L' U L F",
    "OLL 44": "F U R U' R' F'",
    "OLL 45": "F R U R' U' F'",
    "OLL 46": "R' U' R' F R F' U R",
    "OLL 47": "R' U' R' F R F' R' F R F' U R",
    "OLL 48": "F R U R' U' R U R' U' F'",
    "OLL 49": "r U' r2 U r2 U r2 U' r",
    "OLL 50": "r' U r2 U' r2 U' r2 U r'",
    "OLL 51": "F U R U' R' U R U' R' F'",
    "OLL 52": "R U R' U R U' B U' B' R'",
    "OLL 53": "l' U2 L U L' U' L U L' U l",
    "OLL 54": "r U2 R' U' R U R' U' R U' r'",
    "OLL 55": "R' F R U R U' R2 F' R2 U' R' U R U R'",
    "OLL 56": "r' U' r U' R' U R U' R' U R r' U r",
    "OLL 57": "R U R' U' M' U R U' r'",
}

PLL: dict[str, str] = {
    "Aa": "x L2 D2 L' U' L D2 L' U L'",
    "Ab": "x' L2 D2 L U L' D2 L U' L",
    "E": "x' L' U L D' L' U' L D L' U' L D' L' U L D",
    "F": "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "Ga": "R2 U R' U R' U' R U' R2 U' D R' U R D'",
    "Gb": "R' U' R U D' R2 U R' U R U' R U' R2 D",
    "Gc": "R2 U' R U' R U R' U R2 U D' R U' R' D",
    "Gd": "R U R' U' D R2 U' R U' R' U R' U R2 D'",
    "H": "M2 U M2 U2 M2 U M2",
    "Ja": "x R2 F R F' R U2 r' U r U2",
    "Jb": "R U R' F' R U R' U' R' F R2 U' R'",
    "Na": "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "Nb": "R' U R U' R' F' U' F R U R' F R' F' R U' R",
    "Ra": "R U' R' U' R U R D R' U' R D' R' U2 R'",
    "Rb": "R2 F R U R U' R' F' R U2 R' U2 R",
    "T": "R U R' U' R' F R2 U' R' U' R U R' F'",
    "Ua": "M2 U M U2 M' U M2",
    "Ub": "M2 U' M U2 M' U' M2",
    "V": "R' U R' U' y R' F' R2 U' R' U R' F R F",
    "Y": "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Z": "M' U M2 U M2 U M' U2 M2",
}

# Each wide turn, slice turn or rotation as outer face turns plus a rotation.
_EXTRA: dict[str, tuple[str, str]] = {
    "r": ("L", "x"),
    "l": ("R", "x'"),
    "f": ("B", "z"),
    "b": ("F", "z'"),
    "u": ("D", "y"),
    "d": ("U", "y'"),
    "M": ("R L'", "x'"),
    "x": ("", "x"),
    "y": ("", "y"),
    "z": ("", "z"),
}

# Where each face's name points after one quarter rotation: new name -> old name.
_ROTATIONS: dict[str, dict[str, str]] = {
    "x": {"U": "F", "F": "D", "D": "B", "B": "U", "R": "R", "L": "L"},
    "y": {"F": "R", "R": "B", "B": "L", "L": "F", "U": "U", "D": "D"},
    "z": {"R": "U", "D": "R", "L": "D", "U": "L", "F": "F", "B": "B"},
}

_OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}
_QUARTERS = {"": 1, "2": 2, "'": 3}
_SUFFIX = {1: "", 2: "2", 3: "'"}


def expand(alg: str) -> list[str]:
    """The algorithm as outer face turns only, with no rotations."""
    frame = {face: face for face in "URFDLB"}
    moves = []

    for token in alg.split():
        name, quarters = token[0], _QUARTERS[token[1:]]
        for _ in range(quarters):
            faces, rotation = _EXTRA.get(name, (name, ""))
            moves += [frame[move[0]] + move[1:] for move in faces.split()]
            if rotation:
                frame = _rotate(frame, rotation)

    return simplify(moves)


def simplify(moves: list[str]) -> list[str]:
    """Merge turns of the same face that are next to each other."""
    out: list[str] = []
    for move in moves:
        face, quarters = move[0], _QUARTERS[move[1:]]
        at = len(out) - 1
        if out and out[-1][0] == _OPPOSITE[face]:
            at -= 1
        if at >= 0 and out[at][0] == face:
            quarters = (quarters + _QUARTERS[out[at][1:]]) % 4
            del out[at]
            if quarters:
                out.insert(at, face + _SUFFIX[quarters])
        else:
            out.append(move)
    return out


def _rotate(frame: dict[str, str], rotation: str) -> dict[str, str]:
    """The frame after turning the whole cube."""
    turns = _QUARTERS[rotation[1:]]
    for _ in range(turns):
        frame = {new: frame[old] for new, old in _ROTATIONS[rotation[0]].items()}
    return frame
