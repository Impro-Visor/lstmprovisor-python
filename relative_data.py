
import constants
import leadsheet
import numpy as np
import random

LOW_BOUND = 48
HIGH_BOUND = 84

CHORD_SIZE = 12

WINDOW_RADIUS = 12
WINDOW_SIZE = WINDOW_RADIUS*2+1

STARTING_POSITION = 72
INPUT_SIZE = CHORD_SIZE + 1 + 1 + 1 + WINDOW_SIZE
OUTPUT_SIZE = 1 + 1 + WINDOW_SIZE

"""
The relative-jump memory-shift LSTM network takes in input of the form

[
    chord_note x 12, (where chord_note[0] corresponds to current pitchclass)
    vague_position, (a float on -1 to 1 scaled on LOW_BOUND to HIGH_BOUND)
    did_rest,
    did_continue,
    did_jump x WINDOW_SIZE, (where last_jump[i] == 1 iff that was the jump chosen last round)
]

For memory shift processing, we also produce

    shift_size, (an integer in [-WINDOW_RADIUS,WINDOW_RADIUS], for how to shift)

The LSTM should produce as output

[

    rest,                   \
    continue,                } (a softmax set of excl probs) 
    play x WINDOW_SIZE,     /

]

The network starts on middle C (?)
"""

def rotate(li, x):
    """
    Rotate list li by x spaces to the right, i.e.
        rotate([1,2,3,4],1) -> [4,1,2,3]
    """
    if len(li) == 0: return []
    return li[-x % len(li):] + li[:-x % len(li)]

def repeat_print_sparse(li):

    last = None
    lastct = 0
    for c in li+[None]:
        if c == last:
            lastct += 1
        else:
            if last is not None:
                print("[" + " ".join(('.' if x == 0 else str(x)) for x in last) + "]", "*", lastct)
            last = c
            lastct = 1

def vague_position(cur_pos):
    return (2*(cur_pos-LOW_BOUND))/(HIGH_BOUND-LOW_BOUND) - 1

class NoteOutOfRangeException(Exception):
    pass

def melody_to_network_form(chords, melody):
    """
    Given chords and melody, produce the input, memshift, and output arrays
    """

    idx = 0
    input_form = []
    mem_shifts = []
    output_form = []

    cur_pos = next((n for n,d in melody if n is not None)) + random.randrange(-WINDOW_RADIUS, WINDOW_RADIUS+1)

    input_form.append(
        chords[0] +
        [(2.0*(cur_pos-LOW_BOUND))/(HIGH_BOUND-LOW_BOUND) - 1] +
        [1] +
        [0] +
        [0]*WINDOW_SIZE)
    mem_shifts.append(0)

    for note, dur in melody:
        if note is None:
            delta = 0
        else:
            delta = note - cur_pos
            cur_pos = note
        if not (-WINDOW_RADIUS <= delta <= WINDOW_RADIUS):
            raise NoteOutOfRangeException("Jump of size {} from {} to {} not allowed.".format(delta, note-delta, note), delta, note-delta, note)

        rcp = ([1 if note is None else 0] +
            [0] +
            [(1 if i==delta and note is not None else 0) for i in range(-WINDOW_RADIUS, WINDOW_RADIUS+1)])
        mem_shifts.append(delta)
        output_form.append(rcp)
        idx += 1
        input_form.append( rotate(chords[idx%len(chords)], -cur_pos) + [vague_position(cur_pos)] + rcp)

        for _ in range(dur-1):
            rcp = [1 if note is None else 0] + [0 if note is None else 1] + [0]*WINDOW_SIZE
            mem_shifts.append(0)
            output_form.append(rcp)
            idx += 1
            input_form.append( rotate(chords[idx%len(chords)], -cur_pos) + [vague_position(cur_pos)] + rcp)

    # Remove last input and mem shift, since they are unused (and have useless chords, anyway)
    input_form = input_form[:-1]
    mem_shifts = mem_shifts[:-1]

    return input_form, mem_shifts, output_form

def output_form_to_melody(output):
    """
    From an output form, create the corresponding melody form
    """

    melody = []
    cur_pos = STARTING_POSITION

    for timestep in output:
        should_rest = timestep[0]
        should_cont = timestep[1]
        should_jump = timestep[2:]

        if should_cont:
            # Continue a note
            if len(melody) == 0:
                print("ERROR: Can't continue from nothing! Inserting rest")
                melody.append([None, 0])
            melody[-1][1] += 1
        elif should_rest:
            # Rest
            if len(melody)>0 and melody[-1][0] is None:
                # More rest
                melody[-1][1] += 1
            else:
                melody.append([None, 1])
        else:
            # We have a jump!
            jump_amt = should_jump.index(1) - WINDOW_RADIUS
            cur_pos += jump_amt
            melody.append([cur_pos, 1])

    return [tuple(x) for x in melody]







