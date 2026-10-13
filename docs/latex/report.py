"""report.py -- summarise the build logs of both editions."""
import pathlib
import re

for log in sorted(pathlib.Path(".").glob("*/*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    name = log.name

    pages = re.findall(r"Output written on .*?\((\d+) pages?\)", text)
    overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", text)
    errors = [l for l in text.splitlines() if l.startswith("! ")]
    undefined = re.findall(r"LaTeX Warning: (Reference|Citation) .*? undefined", text)
    fontwarn = len(re.findall(r"Font shape .* undefined", text))
    missing = len(re.findall(r"LaTeX Warning: There were undefined references", text))

    print(f"--- {log.parent.name}/{name}")
    print(f"    pages            : {pages}")
    print(f"    errors (!)       : {len(errors)}")
    for e in errors[:8]:
        print(f"        {e}")
    print(f"    undefined refs   : {len(undefined)}")
    print(f"    rerun warnings   : {missing}")
    print(f"    font-shape warns : {fontwarn}")
    if overfull:
        widths = sorted((float(w) for w in overfull), reverse=True)
        print(f"    overfull hboxes  : {len(widths)}"
              f"  max={widths[0]}pt  >20pt={sum(1 for w in widths if w > 20)}"
              f"  >10pt={sum(1 for w in widths if w > 10)}")
    else:
        print("    overfull hboxes  : 0")
    print()
