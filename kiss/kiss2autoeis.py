#!/usr/bin/env python3
"""Convert kiss EISF spectra to AutoEIS input format.

Reads single-spectrum files in the kiss EISF 1.1.0 format (as written by
libkisstype, see /home/philipp/Programming/kiss/kisstype) or tar archives of
such files (as produced by KissDatasetGenerator, e.g.
TorchKissAnn/results/SimulationGap/fullReal_test.tar) and converts them to
the format AutoEIS consumes: frequency in Hz plus complex impedance, as
3-column CSVs (freq, Zreal, Zimag) readable by ``autoeis.io.load_eis_data``.

EISF 1.1.0 layout (one spectra per file):

    EISF, 1.1.0
    "<model string, e.g. r{10}p{1e-05,0.9}-p{1e-06,0.8}>"
    <header description or None>
    <header or None>
    [labelsNames
     "name1", "name2"
    labels
     v1, v2]
    omega, real, im
    <omega>, <real>, <im>
    ...

``omega`` is the angular frequency in rad/s; AutoEIS expects Hz, so the
converter divides by 2*pi. The class label of a spectra (as used by the
TorchKissAnn loaders) is the model string with all ``{...}`` parameter
blocks removed, e.g. ``r{10}p{1e-05,0.9}-p{1e-06,0.8}`` -> ``rp-p``.

Usage:
    kiss2autoeis.py FILE.csv -o OUTDIR
    kiss2autoeis.py ARCHIVE.tar -o OUTDIR
    kiss2autoeis.py ARCHIVE.tar -o OUTDIR --npy        # also write .npy pairs

The output directory receives one CSV per spectra (same base name as the
input) plus a ``manifest.csv`` with the ground-truth class per spectra.
"""

import argparse
import csv
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = "EISF"
_PARAM_BLOCK = re.compile(r"\{[^}]*\}")


@dataclass
class EISFSpectra:
    """A single spectra parsed from an EISF 1.1.0 file."""

    model: str = ""
    header_description: str = ""
    header: str = ""
    label_names: list[str] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    omega: np.ndarray = field(default_factory=lambda: np.array([]))
    Z: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))

    @property
    def freq(self) -> np.ndarray:
        """Frequency in Hz (EISF stores angular frequency in rad/s)."""
        return self.omega / (2.0 * np.pi)

    @property
    def class_string(self) -> str:
        """Class label: model string with parameter blocks removed."""
        return _PARAM_BLOCK.sub("", self.model)

    def to_autoeis(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (freq [Hz], Z [Ohm]) as consumed by AutoEIS."""
        return self.freq, self.Z


def _strip_quotes(line: str) -> str:
    line = line.strip()
    if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
        return line[1:-1]
    return line


def parse_eisf(text: str) -> EISFSpectra:
    """Parses the content of an EISF 1.1.0 file."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty file")

    first = lines[0].split(",")
    if len(first) != 2 or first[0].strip() != MAGIC:
        raise ValueError(f"not an EISF file: {lines[0]!r}")
    version = tuple(int(x) for x in first[1].strip().split("."))
    if version[0] > 1 or (version[0] == 1 and version[1] > 1):
        raise ValueError(f"unsupported EISF version: {first[1].strip()}")

    out = EISFSpectra()
    i = 1
    out.model = _strip_quotes(lines[i]) if lines[i].strip() != "None" else ""
    out.header_description = (
        _strip_quotes(lines[i + 1]) if lines[i + 1].strip() != "None" else ""
    )
    out.header = _strip_quotes(lines[i + 2]) if lines[i + 2].strip() != "None" else ""
    i += 3

    data_lines = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("labelsNames"):
            i += 1
            out.label_names = [
                _strip_quotes(t) for t in lines[i].split(",") if t.strip()
            ]
        elif line.startswith("labels"):
            i += 1
            out.labels = [float(t) for t in lines[i].split(",") if t.strip()]
        elif not line.strip() or line.startswith("#") or line.startswith("omega"):
            pass
        else:
            data_lines.append(line)
        i += 1

    if not data_lines:
        raise ValueError("no data points found")
    data = np.array(
        [[float(x) for x in line.split(",")] for line in data_lines], dtype=float
    )
    out.omega = data[:, 0]
    out.Z = data[:, 1] + 1j * data[:, 2]
    return out


def parse_eisf_file(path: str | Path) -> EISFSpectra:
    """Parses a single EISF file from disk."""
    return parse_eisf(Path(path).read_text())


def parse_eisf_tar(path: str | Path) -> dict[str, EISFSpectra]:
    """Parses all EISF files in a tar archive.

    Returns a dict mapping member name to spectra.
    """
    out: dict[str, EISFSpectra] = {}
    with tarfile.open(path, "r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                out[member.name] = parse_eisf(f.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                print(f"warning: skipping {member.name}: {e}", file=sys.stderr)
    return out


def write_autoeis_csv(spectra: EISFSpectra, path: str | Path) -> None:
    """Writes a spectra as a 3-column CSV (freq, Zreal, Zimag) in Hz/Ohm.

    The file is readable by ``autoeis.io.load_eis_data``.
    """
    freq, Z = spectra.to_autoeis()
    np.savetxt(
        path,
        np.column_stack([freq, Z.real, Z.imag]),
        delimiter=",",
        header="freq, Zreal, Zimag",
        comments="",
    )


def write_manifest(
    entries: list[tuple[str, EISFSpectra]], path: str | Path
) -> None:
    """Writes manifest.csv with the ground-truth class per spectra."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["name", "class", "model", "n_points", "fmin_Hz", "fmax_Hz", "source"]
        )
        for name, s in entries:
            w.writerow(
                [
                    name,
                    s.class_string,
                    s.model,
                    len(s.omega),
                    f"{s.freq.min():.6g}",
                    f"{s.freq.max():.6g}",
                    s.header,
                ]
            )


def convert(
    source: str | Path,
    outdir: str | Path,
    write_npy: bool = False,
) -> list[tuple[str, EISFSpectra]]:
    """Converts a single EISF file or a tar archive into ``outdir``.

    Returns the list of (name, spectra) entries written.
    """
    source = Path(source)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if source.suffix == ".tar":
        parsed = parse_eisf_tar(source)
        entries = list(parsed.items())
    else:
        entries = [(source.stem, parse_eisf_file(source))]

    for name, s in entries:
        base = Path(name).stem
        write_autoeis_csv(s, outdir / f"{base}.csv")
        if write_npy:
            freq, Z = s.to_autoeis()
            np.save(outdir / f"{base}_freq.npy", freq)
            np.save(outdir / f"{base}_Z.npy", Z)

    write_manifest(entries, outdir / "manifest.csv")
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="EISF file or tar archive of EISF files")
    ap.add_argument("-o", "--outdir", required=True, help="output directory")
    ap.add_argument(
        "--npy", action="store_true", help="also write freq/Z as .npy files"
    )
    args = ap.parse_args(argv)

    entries = convert(args.source, args.outdir, write_npy=args.npy)
    n_classes = len({s.class_string for _, s in entries})
    print(f"converted {len(entries)} spectra ({n_classes} classes) -> {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
