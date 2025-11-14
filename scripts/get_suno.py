import suno
from pathlib import Path

OUTPATH = Path("data/suno/sample")


with Path("data/suno/sample.txt").open("r") as f:
    for line in f:
        line = line.rstrip()
        if (OUTPATH/".suno"/("suno-"+line+".mp3")).exists():
            print(f"{line} exists, skip")
        else:
            try:
                print("downloading",OUTPATH/".suno"/("suno-"+line+".mp3"))
                suno.download(line,root=str(OUTPATH))
            except KeyboardInterrupt:
                break
            except Exception: # the library literally throws an "Exception" rahhhhh
                continue 