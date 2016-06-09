from collections import OrderedDict

CHORD_TYPES = {
    '':                 [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    '7b9#11b13':        [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '7sus':             [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    '7b13':             [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    'Bass':             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'aug':              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    '13sus4':           [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    'm9b5':             [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    '7b9b13':           [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    'm11#5':            [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    '7b6':              [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    '7b9sus':           [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    '7aug':             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    '9no5':             [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    '7#11b13':          [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    'm#5':              [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'M#5':              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    '9#5#11':           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    '7b5b13':           [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '7b5':              [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    'mM9':              [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'm7b5':             [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'susb9':            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'mM7':              [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    '13sus':            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    '7b9':              [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '7b5b9':            [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    '9sus':             [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'mb6b9':            [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    '9b13':             [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    'M7':               [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'M9#5':             [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    '7no5':             [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    'M9':               [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    '+':                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    '9b5b13':           [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '7':                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '13b9#11':          [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    'o7':               [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    '7b9b13#11':        [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '9sus4':            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    '7#5#9':            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    'M':                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    '13#9#11':          [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    '13#11':            [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    '7susb9':           [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    '9#11b13':          [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '7#5':              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    '7#9':              [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    '13#9':             [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    '7b5b9b13':         [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    '7sus4b9b13':       [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    'h7':               [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'm11':              [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    'm13':              [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    '7b9#11':           [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    '7#5b9#11':         [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'NC':               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'm+':               [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'm7':               [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'm6':               [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '6':                [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'm9':               [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7#9b13':           [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    '7sus4b9':          [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    '9+':               [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    'M7#5':             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'Msus4':            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'Msus2':            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7#5b9':            [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    'mb6':              [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    '7#9#11b13':        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    '7#11':             [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    '11':               [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    '13':               [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    '9b5':              [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    '13b5':             [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    'Blues':            [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    '13b9':             [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    '+add9':            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'M7#11':            [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    'madd9':            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'mM7b6':            [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    'M7+':              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'aug7':             [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    'mb6M7':            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    'm11b5':            [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    '7b9b13sus4':       [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    '9':                [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '+7':               [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    'm9#5':             [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    'M#5add9':          [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    '7+':               [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    '9#11':             [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    '7alt':             [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    'sus2':             [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'sus4':             [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'm7#5':             [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    'm69':              [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'm':                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    '9#5':              [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    '7#9#11':           [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    '7sus4':            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    '7b5#9':            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    '7b9sus4':          [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
}

NOTE_OFFSETS = OrderedDict([
    ("c",    0),
    ("db",   1),
    ("d",    2),
    ("eb",   3),
    ("e",    4),
    ("f",    5),
    ("gb",   6),
    ("g",    7),
    ("ab",   8),
    ("a",    9),
    ("bb",   10),
    ("b",    11),

    ("c#",   1),
    ("d#",   3),
    ("f#",   6),
    ("g#",   8),
    ("a#",   10),

    ("cb",   11),
    ("fb",   4),
    ("e#",   5),
    ("b#",   0),
])

CHORD_NOTE_OFFSETS = OrderedDict((k[:1].upper()+k[1:],v%12) for k,v in NOTE_OFFSETS.items())
 
WHOLE                = 480; # slots in a whole note
HALF                 = WHOLE/2;              # 240
QUARTER              = WHOLE/4;              # 120
EIGHTH               = WHOLE/8;              #  60
SIXTEENTH            = WHOLE/16;             #  30
THIRTYSECOND         = WHOLE/32;             #  15
    
HALF_TRIPLET         = 2*HALF/3;             # 160
QUARTER_TRIPLET      = 2*QUARTER/3;          #  80
EIGHTH_TRIPLET       = 2*EIGHTH/3;           #  40
SIXTEENTH_TRIPLET    = 2*SIXTEENTH/3;        #  20
THIRTYSECOND_TRIPLET = 2*THIRTYSECOND/3;     #  10

QUARTER_QUINTUPLET      = 4*QUARTER/5;       #  96
EIGHTH_QUINTUPLET       = 4*EIGHTH/5;        #  48
SIXTEENTH_QUINTUPLET    = 4*SIXTEENTH/5;     #  24
THIRTYSECOND_QUINTUPLET = 4*THIRTYSECOND/5;  #  12

DOTTED_HALF           = 3*HALF/2;            # 360
DOTTED_QUARTER        = 3*QUARTER/2;         # 180
DOTTED_EIGHTH         = 3*EIGHTH/2;          #  90
DOTTED_SIXTEENTH      = 3*SIXTEENTH/2;       #  45

FOUREIGHTIETH         = 1; # WHOLE/WHOLE    #   1
TWOFORTIETH           = 2;                   #   2
ONETWENTIETH          = 4;                   #   4
SIXTIETH              = 8;                   #   8

RESOLUTION_SCALAR = 10;

MIDDLE_C_MIDI = 60
OCTAVE = 12