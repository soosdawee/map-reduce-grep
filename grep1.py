#!/usr/bin/env python3
"""
    pip install rich datasets
    Firs run: [python] grep1.py -i "love|death" --mappers [number] --partitions [number]  --reducers [number] --books [number]
    After: [python] grep1.py -i "love|death" --mappers [number] --partitions [number]  --reducers [number]
"""

import argparse
import json
import os
import queue
import re
import shutil
import sys
import time
import zlib
from collections import defaultdict
from multiprocessing import Manager, Pool
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("This demo needs the 'rich' library:  pip install rich")


GUTENBERG_DIR = "gutenberg"
GUTENBERG_MAX_CHARS = 50000000


def download_gutenberg(outdir=GUTENBERG_DIR, max_chars=GUTENBERG_MAX_CHARS, max_books=None):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Downloading the corpus needs the 'datasets' library:  pip install datasets")

    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    total = 0
    seen = set()
    for i, example in enumerate(ds):
        name = example.get("id") or f"book-{i:05d}"
        if name in seen:
            continue
        seen.add(name)
        with open(Path(outdir) / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write(example["text"] + "\n")
        total += len(example["text"])
        if max_books is not None:
            if len(seen) >= max_books:
                break
        elif total > max_chars:
            break

    print(f"Size of gutenberg text: {total / 1024 / 1024:.1f} MB in {len(seen)} books")


def collect_files(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.txt")))
        elif p.is_file():
            paths.append(p)
    if not paths:
        sys.exit("No input files found (directories are searched for *.txt).")
    return paths


def book_title(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(300), f):
                line = line.lstrip("\ufeff")
                if line.startswith("Title:"):
                    title = line[len("Title:"):].strip()
                    if title:
                        return title
    except OSError:
        pass
    return path.stem


def make_splits(paths, split_bytes): #tbs
    splits, seen = [], set()
    for path in paths:
        book = book_title(path)
        if book in seen:
            book = f"{book} ({path.stem})"
        seen.add(book)

        start = offset = 0
        first_line, line_no = 1, 0
        with open(path, "rb") as f:
            for line_no, raw in enumerate(f, 1):
                offset += len(raw)
                if offset - start >= split_bytes:
                    splits.append(_split(len(splits), path, book, start, offset, first_line, line_no))
                    start, first_line = offset, line_no + 1
        if offset > start:
            splits.append(_split(len(splits), path, book, start, offset, first_line, line_no))
    return splits


def _split(sid, path, book, start, end, first_line, last_line): #tbs
    return dict(id=sid, path=str(path), book=book, offset=start, length=end - start,
                first_line=first_line, last_line=last_line,
                n_lines=last_line - first_line + 1)


def partition(key, n_partitions): #tbs
    return zlib.crc32(key.encode("utf-8")) % n_partitions


def map_task(split, pattern, flags, n_partitions, workdir, events, delay): #tbs
    pid = os.getpid()
    regex = re.compile(pattern, flags)
    events.put(("map_start", pid, split["id"]))

    with open(split["path"], "rb") as f:
        f.seek(split["offset"])
        data = f.read(split["length"])
    raw_lines = data.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()

    buckets = defaultdict(list)
    matches = 0
    for i, raw in enumerate(raw_lines):
        line = raw.decode("utf-8", errors="replace").rstrip("\r")
        if regex.search(line):
            key = split["book"]
            record = [key, split["first_line"] + i, line]
            buckets[partition(key, n_partitions)].append(json.dumps(record, ensure_ascii=False))
            matches += 1
        if (i + 1) % 100 == 0:
            if delay:
                time.sleep(delay)
            if (i + 1) % 500 == 0:
                events.put(("map_progress", pid, split["id"], i + 1, matches))

    counts = {}
    for p, records in buckets.items():
        out = Path(workdir) / f"map-{split['id']:05d}-part-{p}.jsonl"
        out.write_text("\n".join(records) + "\n", encoding="utf-8")
        counts[p] = len(records)

    events.put(("map_done", pid, split["id"], len(raw_lines), matches, counts))
    return counts


def reduce_task(p, workdir, outdir, events, delay): #tbs
    pid = os.getpid()
    files = sorted(Path(workdir).glob(f"map-*-part-{p}.jsonl"))
    events.put(("reduce_start", pid, p, len(files)))

    groups = defaultdict(list)
    records = 0
    for i, path in enumerate(files, 1):
        with open(path, encoding="utf-8") as f:
            for rec in f:
                key, line_no, line = json.loads(rec)
                groups[key].append((line_no, line))
                records += 1
        if delay:
            time.sleep(delay * 2)
        events.put(("reduce_fetch", pid, p, i, records))

    events.put(("reduce_sort", pid, p, len(groups)))
    summary = {}
    with open(Path(outdir) / f"part-{p:05d}.txt", "w", encoding="utf-8") as out:
        for key in sorted(groups):
            values = sorted(groups[key])
            out.write(f"=== {key} ({len(values)} matches) ===\n")
            for line_no, line in values:
                out.write(f"{line_no:>7}: {line}\n")
            out.write("\n")
            summary[key] = len(values)

    events.put(("reduce_done", pid, p, summary))
    return summary


def trunc(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


class Dashboard:
    def __init__(self, args, paths, splits):
        self.pattern = args.pattern
        self.ignore_case = args.ignore_case
        self.n_books = len(paths)
        self.n_bytes = sum(p.stat().st_size for p in paths)
        self.splits = {s["id"]: s for s in splits}
        self.n_mappers, self.n_reducers = args.mappers, args.reducers
        self.n_partitions = args.partitions
        self.phase = "map"
        self.t0 = time.time()
        self.t_map_end = None

        self.map_slots = {}
        self.map_state = {}
        self.maps_done = 0
        self.lines_scanned = 0
        self.total_matches = 0
        self.part_records = [0] * self.n_partitions
        self.part_files = [0] * self.n_partitions

        self.part_state = {p: dict(status="queued", reducer=None, files=0, total=0, records=0, keys=0)
                           for p in range(self.n_partitions)}
        self.reduce_slots = {}
        self.reducer_state = {r: dict(part=None, done_parts=[], records=0)
                              for r in range(self.n_reducers)}
        self.parts_done = 0

    @staticmethod
    def _slot(slots, pid):
        if pid not in slots:
            slots[pid] = len(slots)
        return slots[pid]

    def handle(self, ev):
        kind = ev[0]
        if kind == "map_start":
            _, pid, sid = ev
            self.map_state[self._slot(self.map_slots, pid)] = dict(split=sid, done=0, matches=0, finished=False)
        elif kind == "map_progress":
            _, pid, sid, done, matches = ev
            self.map_state[self._slot(self.map_slots, pid)].update(done=done, matches=matches)
        elif kind == "map_done":
            _, pid, sid, n_lines, matches, counts = ev
            self.map_state[self._slot(self.map_slots, pid)].update(done=n_lines, matches=matches, finished=True)
            self.maps_done += 1
            self.lines_scanned += n_lines
            self.total_matches += matches
            for p, c in counts.items():
                self.part_records[p] += c
                self.part_files[p] += 1
        elif kind == "reduce_start":
            _, pid, p, total = ev
            r = self._slot(self.reduce_slots, pid)
            self.part_state[p].update(status="fetching", reducer=r, total=total)
            self.reducer_state[r]["part"] = p
        elif kind == "reduce_fetch":
            _, pid, p, i, records = ev
            self.part_state[p].update(files=i, records=records)
        elif kind == "reduce_sort":
            _, pid, p, keys = ev
            self.part_state[p].update(status="sorting + writing", keys=keys)
        elif kind == "reduce_done":
            _, pid, p, summary = ev
            self.part_state[p].update(status="done", keys=len(summary))
            r = self.part_state[p]["reducer"]
            self.reducer_state[r]["done_parts"].append(p)
            self.reducer_state[r]["records"] += self.part_state[p]["records"]
            self.parts_done += 1

    def render(self):
        return Group(self._header(), self._map_panel(), self._shuffle_panel(), self._reduce_panel())

    def _header(self):
        flags = " (case-insensitive)" if self.ignore_case else ""
        return Text.assemble(
            ("grep ", "bold"), (f"/{self.pattern}/", "bold yellow"), (flags, "dim"),
            ("   ", ""),
            (f"{self.n_books} books, {self.n_bytes / 1e6:.1f} MB → {len(self.splits)} splits   ", "dim"),
            (f"{self.n_mappers} mappers, {self.n_reducers} reducers, "
             f"{self.n_partitions} partitions   ", "dim"),
            (f"{time.time() - self.t0:5.1f}s", "bold"),
        )

    def _map_panel(self):
        overall = Table.grid(padding=(0, 1))
        overall.add_row(
            ProgressBar(total=len(self.splits), completed=self.maps_done, width=40),
            Text(f"{self.maps_done}/{len(self.splits)} map tasks   "
                 f"{self.lines_scanned:,} lines scanned   {self.total_matches:,} matches"),
        )
        workers = Table.grid(padding=(0, 2))
        for slot in range(self.n_mappers):
            st = self.map_state.get(slot)
            name = Text(f"mapper {slot + 1}", style="cyan")
            if st is None:
                workers.add_row(name, Text("idle", style="dim"), Text(""), Text(""))
                continue
            s = self.splits[st["split"]]
            task = Text(f"split {s['id']:>3}  {trunc(s['book'], 30):<30}  "
                        f"lines {s['first_line']:>6}–{s['last_line']:<6}",
                        style="dim" if st["finished"] else "")
            workers.add_row(name, task,
                            ProgressBar(total=s["n_lines"], completed=st["done"], width=20),
                            Text(f"{st['matches']} hits", style="green" if st["matches"] else "dim"))
        style = "cyan" if self.phase == "map" else "dim"
        return Panel(Group(overall, Text(""), workers), border_style=style,
                     title="1 · map: scan each split, emit matching lines keyed by book")

    def _shuffle_panel(self):
        grid = Table.grid(padding=(0, 2))
        top = max(1, max(self.part_records))
        for p in range(self.n_partitions):
            st = self.part_state[p]
            if st["reducer"] is None:
                target = Text(f"partition {p:>2} → queued   ", style="dim")
            else:
                colour = "green" if st["status"] == "done" else "yellow"
                target = Text.assemble((f"partition {p:>2} → ", "magenta"),
                                       (f"reducer {st['reducer'] + 1}", colour))
            grid.add_row(
                target,
                ProgressBar(total=top, completed=self.part_records[p], width=40,
                            complete_style="magenta", finished_style="magenta"),
                Text(f"{self.part_records[p]:,} records from {self.part_files[p]} map outputs"),
            )
        style = "magenta" if self.phase == "map" else "dim"
        return Panel(grid, border_style=style,
                     title="2 · shuffle: route each record to partition crc32(book) mod P")

    def _reduce_panel(self):
        overall = Table.grid(padding=(0, 1))
        overall.add_row(
            ProgressBar(total=self.n_partitions, completed=self.parts_done, width=40,
                        complete_style="green"),
            Text(f"{self.parts_done}/{self.n_partitions} partitions reduced"),
        )
        grid = Table.grid(padding=(0, 2))
        for r in range(self.n_reducers):
            rs = self.reducer_state[r]
            name = Text(f"reducer {r + 1}", style="green")
            history = Text(f"done: {', '.join(map(str, rs['done_parts'])) or '–'}   "
                           f"{rs['records']:,} records total", style="dim")
            if rs["part"] is None:
                grid.add_row(name, Text("waiting", style="dim"), Text(""), Text(""), history)
                continue
            st = self.part_state[rs["part"]]
            colour = "green" if st["status"] == "done" else "yellow"
            grid.add_row(
                name,
                Text(f"partition {rs['part']:>2}  {st['status']:<17}", style=colour),
                ProgressBar(total=max(1, st["total"]), completed=st["files"], width=20,
                            complete_style="green"),
                Text(f"fetched {st['files']}/{st['total']} files   "
                     f"{st['records']:,} records   {st['keys']} books"),
                history,
            )
        style = "green" if self.phase == "reduce" else "dim"
        return Panel(Group(overall, Text(""), grid), border_style=style,
                     title="3 · reduce: each free reducer takes the next partition, "
                           "groups by book, sorts by line number")


def run_phase(results, events, dash, live):
    while True:
        drain(events, dash)
        live.update(dash.render())
        if all(r.ready() for r in results):
            break
        time.sleep(0.05)
    drain(events, dash)
    live.update(dash.render())
    return [r.get() for r in results]


def drain(events, dash):
    while True:
        try:
            dash.handle(events.get_nowait())
        except queue.Empty:
            return


def main():
    ap = argparse.ArgumentParser(description="Distributed grep as a visual MapReduce demo.")
    ap.add_argument("pattern", help="regular expression to search for")
    ap.add_argument("inputs", nargs="*",
                    help="text files and/or directories of *.txt files "
                         f"(default: ./{GUTENBERG_DIR}/, downloaded from Hugging Face if missing)")
    ap.add_argument("--books", type=int, default=None,
                    help=f"re-download ./{GUTENBERG_DIR}/ with this many books instead of stopping at "
                         f"{GUTENBERG_MAX_CHARS:,} characters (ignored when inputs are given)")
    ap.add_argument("-i", "--ignore-case", action="store_true")
    ap.add_argument("--mappers", type=int, default=min(4, os.cpu_count() or 4))
    ap.add_argument("--reducers", type=int, default=3)
    ap.add_argument("--partitions", type=int, default=None,
                    help="number of partitions (= reduce tasks) the keys are hashed into; "
                         "default: same as --reducers")
    ap.add_argument("--split-kb", type=int, default=256, help="target split size in KB (default 256)")
    ap.add_argument("--delay", type=float, default=0.01,
                    help="seconds to pause per 100 lines so the demo is watchable; 0 = full speed")
    ap.add_argument("--workdir", default="mr_work", help="intermediate map output (wiped each run)")
    ap.add_argument("--outdir", default="mr_output", help="final reducer output (wiped each run)")
    args = ap.parse_args()
    if args.partitions is None:
        args.partitions = args.reducers
    if min(args.mappers, args.reducers, args.partitions) < 1:
        sys.exit("--mappers, --reducers and --partitions must be at least 1.")

    flags =re.IGNORECASE if args.ignore_case else 0
    try:
        re.compile(args.pattern, flags)
    except re.error as e:
        sys.exit(f"Invalid pattern: {e}")

    if not args.inputs:
        if args.books is not None or not any(Path(GUTENBERG_DIR).glob("*.txt")):
            shutil.rmtree(GUTENBERG_DIR, ignore_errors=True)
            download_gutenberg(max_books=args.books)
        args.inputs = [GUTENBERG_DIR]

    paths = collect_files(args.inputs)
    for d in (args.workdir, args.outdir):
        shutil.rmtree(d, ignore_errors=True)
        Path(d).mkdir(parents=True)

    splits = make_splits(paths, args.split_kb * 1024)
    console = Console()
    manager = Manager()
    events = manager.Queue()
    dash = Dashboard(args, paths, splits)

    with Live(dash.render(), console=console, refresh_per_second=12) as live:
        with Pool(args.mappers) as pool:
            jobs = [pool.apply_async(map_task, (s, args.pattern, flags, args.partitions,
                                                args.workdir, events, args.delay))
                    for s in splits]
            run_phase(jobs, events, dash, live)
        dash.t_map_end = time.time()

        dash.phase = "reduce"
        with Pool(args.reducers) as pool:
            jobs = [pool.apply_async(reduce_task, (p, args.workdir, args.outdir, events, args.delay))
                    for p in range(args.partitions)]
            summaries = run_phase(jobs, events, dash, live)

    t_end = time.time()
    totals = {k: v for s in summaries for k, v in s.items()}
    table = Table(title="Matches per book", title_justify="left")
    table.add_column("book")
    table.add_column("matches", justify="right")
    table.add_column("partition", justify="right")
    table.add_column("reducer", justify="right")
    for book, n in sorted(totals.items(), key=lambda kv: -kv[1])[:15]:
        p = partition(book, args.partitions)
        table.add_row(trunc(book, 50), f"{n:,}", str(p), str(dash.part_state[p]["reducer"] + 1))
    if len(totals) > 15:
        table.add_row(f"… and {len(totals) - 15} more", "", "", "")
    console.print(table)
    console.print(
        f"{sum(totals.values()):,} matching lines in {len(totals)} of {len(paths)} books.   "
        f"map {dash.t_map_end - dash.t0:.1f}s, reduce {t_end - dash.t_map_end:.1f}s\n"
        f"Output: {Path(args.outdir).resolve()}/part-*.txt   "
        f"Intermediate map output: {Path(args.workdir).resolve()}/"
    )
    manager.shutdown()


if __name__ == "__main__":
    main()