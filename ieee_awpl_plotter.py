#!/usr/bin/env python3
"""Plot the three existing antenna CSVs with the requested AWPL formatting.

Dependency: matplotlib (python -m pip install matplotlib).
Run: python ieee_awpl_plotter.py

Inputs, beside this script: ant1_hfss.csv, ant2_hfss.csv, ant3_hfss.csv.
Required columns: Freq_GHz, S11_dB, RealizedGain_dBi.
Outputs, beside this script: Evolution_Plot.pdf and Evolution_Plot.png.
The input values are plotted directly, without smoothing or normalization.
"""

import csv
import math
from pathlib import Path

import matplotlib

# Save files without requiring a desktop window or an interactive backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


BASE_DIRECTORY = Path(__file__).resolve().parent
ANTENNAS = (
    ("ant1_hfss.csv", "Ant. I", "k"),
    ("ant2_hfss.csv", "Ant. II", "b"),
    ("ant3_hfss.csv", "Ant. III", "r"),
)
COLUMNS = ("Freq_GHz", "S11_dB", "RealizedGain_dBi")


def configure_style() -> None:
    """Apply the requested font sizes and embed TrueType fonts in the PDF."""
    # Fail explicitly if the required font is absent instead of substituting it.
    try:
        font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman"),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is required. Install this font before plotting."
        ) from exc

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "savefig.facecolor": "white",
    })


def read_antenna_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Read GHz/dB/dBi columns and reject missing or invalid measurements."""
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                values = tuple(float(row[column]) for column in COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}, line {reader.line_num}: all measurements must be numeric."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path.name}, line {reader.line_num}: nonfinite measurement.")
            rows.append(values)

    if len(rows) < 2:
        raise ValueError(f"{path.name}: at least two data rows are required.")
    # Keep each frequency paired with its original S11 and gain values.
    rows.sort(key=lambda row: row[0])
    frequency, s11, gain = map(list, zip(*rows))
    if any(right <= left for left, right in zip(frequency, frequency[1:])):
        raise ValueError(f"{path.name}: duplicate frequencies; export a single sweep per CSV.")
    if frequency[-1] < 2.6 or frequency[0] > 4.4:
        raise ValueError(f"{path.name}: no data in the requested 2.6-4.4 GHz interval.")
    return frequency, s11, gain


def create_figure(input_directory: Path = BASE_DIRECTORY) -> plt.Figure:
    """Build the 3.5 by 2.8 inch figure from the three existing CSV files."""
    configure_style()
    datasets = [
        (label, color, read_antenna_csv(Path(input_directory) / filename))
        for filename, label, color in ANTENNAS
    ]
    fig, ax_s11 = plt.subplots(figsize=(3.5, 2.8))
    ax_gain = ax_s11.twinx()

    antenna_handles = []
    for label, color, (frequency, s11, gain) in datasets:
        line, = ax_s11.plot(frequency, s11, color=color, linestyle="-", label=label)
        ax_gain.plot(frequency, gain, color=color, linestyle="--", label=label)
        antenna_handles.append(line)

    ax_s11.set(xlabel="Frequency (GHz)", ylabel="S11 (dB)",
               xlim=(2.6, 4.4), ylim=(-35, 0))
    ax_gain.set(ylabel="Realized Gain (dBi)", ylim=(-30, 10))
    ax_s11.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_s11.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_s11.yaxis.set_major_locator(MultipleLocator(5))
    ax_gain.yaxis.set_major_locator(MultipleLocator(10))
    ax_s11.set_axisbelow(True)
    ax_s11.grid(True, which="major", axis="both", linestyle="--", alpha=0.3)
    ax_gain.grid(False)
    for axis in (ax_s11, ax_gain):
        axis.tick_params(axis="both", which="major", direction="in", labelsize=8)

    # Two compact rows distinguish antenna color and the measured quantity.
    # Both legends use the requested 10 pt font and occupy the lower clear area.
    antenna_legend = ax_gain.legend(
        handles=antenna_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=3, frameon=False, handlelength=1.1, handletextpad=0.35,
        columnspacing=0.65, borderaxespad=0.2,
    )
    ax_gain.add_artist(antenna_legend)
    ax_gain.legend(
        handles=[
            Line2D([], [], color="k", linestyle="-", label="S11"),
            Line2D([], [], color="k", linestyle="--", label="Realized Gain"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=2,
        frameon=False, handlelength=1.8, handletextpad=0.4,
        columnspacing=1.0, borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    """Export a vector PDF and a 600 dpi PNG next to the input CSV files."""
    fig = create_figure()
    try:
        for extension in ("pdf", "png"):
            output_path = BASE_DIRECTORY / f"Evolution_Plot.{extension}"
            # tight crops the export boundary; figsize remains exactly 3.5 x 2.8.
            fig.savefig(output_path, dpi=600, bbox_inches="tight")
            print(f"Saved: {output_path}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
