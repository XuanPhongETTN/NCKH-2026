#!/usr/bin/env python3
"""Generate three synthetic antenna CSV files with prescribed response shapes.

Dependencies: numpy and pandas.
Run: python generate_sample_hfss_data.py

The files ant1_hfss.csv, ant2_hfss.csv and ant3_hfss.csv are written to the
current working directory. They are illustrative mathematical data, not
HFSS simulation results. Each file has exactly 401 rows and these columns:
Freq_GHz, S11_dB, RealizedGain_dBi.

A strictly uniform 401-point grid over 2.6--4.4 GHz has a 0.0045 GHz step
and does not contain 3.10 or 3.90 GHz. The nearest samples are therefore
replaced by the required characteristic frequencies. The shared grid
still has 401 distinct, increasing points and the same endpoints.

A parabola with an interior minimum opens upward. Ant. I follows the
specified minimum of -20 dB at 3.5 GHz rather than a downward parabola.
"""

from pathlib import Path

import numpy as np
import pandas as pd


FREQUENCY_MIN_GHZ = 2.6
FREQUENCY_MAX_GHZ = 4.4
NUMBER_OF_POINTS = 401
COLUMN_NAMES = ("Freq_GHz", "S11_dB", "RealizedGain_dBi")
OUTPUT_FILENAMES = ("ant1_hfss.csv", "ant2_hfss.csv", "ant3_hfss.csv")


def make_frequency_grid() -> np.ndarray:
    """Return 401 frequencies including exact resonance, null and band edges."""
    frequencies = np.linspace(FREQUENCY_MIN_GHZ, FREQUENCY_MAX_GHZ, NUMBER_OF_POINTS)
    for landmark in (3.10, 3.30, 3.50, 3.55, 3.80, 3.90):
        nearest_index = int(np.argmin(np.abs(frequencies - landmark)))
        frequencies[nearest_index] = landmark
    if frequencies.size != NUMBER_OF_POINTS or not np.all(np.diff(frequencies) > 0):
        raise RuntimeError("Frequency grid must contain 401 distinct, increasing samples.")
    return frequencies


def gaussian(frequencies: np.ndarray, center_ghz: float, sigma_ghz: float) -> np.ndarray:
    """Unit-height Gaussian, with sigma expressed in GHz."""
    return np.exp(-0.5 * ((frequencies - center_ghz) / sigma_ghz) ** 2)


def gain_with_nulls(
    frequencies: np.ndarray,
    baseline_dbi: np.ndarray,
    null_frequencies_ghz: tuple[float, ...],
    null_depth_dbi: float,
    width_ghz: float,
) -> np.ndarray:
    """Apply smooth finite-depth nulls without overwriting individual samples.

    Each factor 1-exp(-((f-f_null)/width)**2) is nonnegative, is exactly
    zero at its null, and has zero first derivative there. If the baseline
    exceeds the null depth, the resulting curve has its exact minimum
    at each requested null. expm1 avoids cancellation close to a null.
    """
    notch_product = np.ones_like(frequencies, dtype=float)
    for null_frequency in null_frequencies_ghz:
        notch_product *= -np.expm1(-((frequencies - null_frequency) / width_ghz) ** 2)
    return null_depth_dbi + (baseline_dbi - null_depth_dbi) * notch_product


def antenna_one_response(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single patch resonance and a strictly positive Gaussian gain curve."""
    s11_db = -20.0 + 18.0 * ((frequencies - 3.50) / 0.90) ** 2
    realized_gain_dbi = 6.6 * gaussian(frequencies, center_ghz=3.50, sigma_ghz=0.45)
    return s11_db, realized_gain_dbi


def antenna_two_response(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two matched S11 resonances and one gain null at 3.10 GHz, -15 dBi."""
    s11_db = (
        -2.0
        - 20.0 * gaussian(frequencies, center_ghz=3.40, sigma_ghz=0.11)
        - 18.0 * gaussian(frequencies, center_ghz=3.70, sigma_ghz=0.11)
    )
    baseline_dbi = -3.0 + 9.4 * gaussian(frequencies, center_ghz=3.50, sigma_ghz=0.48)
    realized_gain_dbi = gain_with_nulls(
        frequencies, baseline_dbi, null_frequencies_ghz=(3.10,),
        null_depth_dbi=-15.0, width_ghz=0.035,
    )
    return s11_db, realized_gain_dbi


def antenna_three_response(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broad matching band and two gain nulls at 3.10/3.90 GHz, -20 dBi.

    Throughout 3.3 <= f <= 3.8 GHz, abs(f-3.55) <= 0.25. Consequently
    S11 <= -3 - 17*exp(-(0.25/0.32)**4), approximately -14.71 dB.
    This guarantees the entire continuous band is below -10 dB.
    """
    s11_db = -3.0 - 17.0 * np.exp(-((frequencies - 3.55) / 0.32) ** 4)
    baseline_dbi = -8.0 + 14.0 * gaussian(frequencies, center_ghz=3.55, sigma_ghz=0.50)
    realized_gain_dbi = gain_with_nulls(
        frequencies, baseline_dbi, null_frequencies_ghz=(3.10, 3.90),
        null_depth_dbi=-20.0, width_ghz=0.035,
    )
    return s11_db, realized_gain_dbi


def generate_sample_data(output_directory: str | Path | None = None) -> list[Path]:
    """Generate and write the three CSV files; default destination is cwd.

    Existing files with the same three names are replaced. No DataFrame
    index or metadata column is written to the CSV files.
    """
    destination = Path.cwd() if output_directory is None else Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    frequencies = make_frequency_grid()
    response_functions = (antenna_one_response, antenna_two_response, antenna_three_response)
    output_paths = []

    for filename, response_function in zip(OUTPUT_FILENAMES, response_functions):
        s11_db, realized_gain_dbi = response_function(frequencies)
        dataframe = pd.DataFrame({
            "Freq_GHz": frequencies,
            "S11_dB": s11_db,
            "RealizedGain_dBi": realized_gain_dbi,
        }, columns=list(COLUMN_NAMES))
        if dataframe.shape != (NUMBER_OF_POINTS, 3) or not np.isfinite(dataframe.to_numpy()).all():
            raise RuntimeError(f"Invalid data generated for {filename}.")
        output_path = destination / filename
        dataframe.to_csv(output_path, index=False, float_format="%.10f", encoding="utf-8")
        output_paths.append(output_path.resolve())

    return output_paths


def main() -> None:
    """Generate the CSV files in the directory from which the command is run."""
    paths = generate_sample_data()
    print("Generated synthetic antenna data: 401 samples from 2.6 to 4.4 GHz.")
    print("Columns: " + ", ".join(COLUMN_NAMES))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
