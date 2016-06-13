import sys
import leadsheet

if __name__ == '__main__':
    melody = []
    chords = []
    for arg in sys.argv[1:]:
        nc,nm = leadsheet.parse_leadsheet(arg)
        melody.extend(nm)
        chords.extend(nc)
    leadsheet.write_leadsheet(chords, melody, sys.argv[1] + "-cat.ls")