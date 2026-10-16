#!/usr/bin/env python3
"""Fit the specified coupled, parallel RLC filtenna model to complex S11.

Python >= 3.10; dependencies: numpy >= 1.24, scipy >= 1.10.
Install with: python -m pip install numpy scipy

All public numerical arguments use SI units: Hz, ohm, H, F, and dimensionless
coupling coefficients. R_lp, R_ls, R_lt are *parallel loss resistances*, not
series resistances and not the radiation resistance Rr.

The intrinsic modal admittance is exactly

    Y(w) = diag(1 / R_li) + 1j*w*diag(C_i) + inv(L)/(1j*w),
    L_ii = L_i,  L_ij = k_ij*sqrt(L_i*L_j).

Ant. I uses p; Ant. II uses p,s; Ant. III uses p,s,t, including all three
magnetic couplings. Modes p, s and t represent the patch, U-slot and
parasitic strips, respectively. Diagonal and off-diagonal admittances
are taken from the inverse of the *full* inductance matrix.

Let A = [n_f*e_p, d]. The reciprocal two-port impedance is A.T @ inv(Y) @ A,
with Z_f = R_f + 1j*w*L_f added only to Z11. The second port is terminated
by Rr, and the fitted quantity is exactly

    S11 = (Z11*Rr + Z11*Z22 - Z12*Z21 - R0*(Rr+Z22)) /
          (Z11*Rr + Z11*Z22 - Z12*Z21 + R0*(Rr+Z22)).

For Ant. I, d_p=1, so this reduces to the specified expression

    Zin = Zf + n_f**2 / (1/Rr + 1/R_lp + 1j*(w*C_p - 1/(w*L_p))).

n_f, R_f, L_f, Rr and d are known, fixed calibration/model inputs; they
are never silently fitted. For II/III, d defaults to ones, which is an
explicit convention, not a value inferred from S11. Supply the d vector
from the circuit analysis when using measured or simulated antenna data.

Identifiability: Ant. III has 12 RLC/coupling parameters but at most 11
independent real coefficients in its one-port rational response. A good
S11 fit alone therefore cannot identify a unique physical set. Use
fixed_parameters, for example {"L_t": 2.0e-9}, with independently known
values. Additional degeneracies can occur for uncoupled/unobservable modes.
No hidden-mode inductance is silently fixed. The result reports numerical
rank and every explicitly fixed element.

CSV inputs contain frequency and two S11 columns (RI, magnitude/phase, or
dB/phase). Touchstone 1.x .s1p and .s2p inputs are also supported. A .s2p
file contributes S11 only. Phase is required for magnitude input.

Examples:
    python rlc_circuit_fitter.py
    python rlc_circuit_fitter.py --demo --output-dir rlc_fit_results
    python rlc_circuit_fitter.py --input antenna_iii.csv --antenna III \
        --representation ri --frequency-unit GHz --d 1 0.35 0.25 \
        --nf 1.1 --rr 50 --rf 0.5 --lf-nh 0.12 --fix L_t=2e-9
    python rlc_circuit_fitter.py --input antenna_ii.s1p --antenna II \
        --d 1 0.35 --rr 50

SciPy reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.signal import find_peaks


MODES = {"I": ("p",), "II": ("p", "s"), "III": ("p", "s", "t")}
FREQUENCY_SCALES = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _configuration(antenna: str | int) -> str:
    value = re.sub(r"[\s._-]", "", str(antenna)).upper()
    for prefix in ("ANTENNA", "ANT"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = {"1": "I", "2": "II", "3": "III"}.get(value, value)
    if value not in MODES:
        raise ValueError("antenna must be I, II or III.")
    return value


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        condition = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {condition}.")
    return value


def _frequencies(frequency_hz: ArrayLike) -> FloatArray:
    values = np.asarray(frequency_hz, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("frequency_hz must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All frequencies must be finite and strictly positive.")
    return values


def complex_s11(
    *,
    real: ArrayLike | None = None,
    imag: ArrayLike | None = None,
    magnitude: ArrayLike | None = None,
    phase: ArrayLike | None = None,
    magnitude_unit: str = "linear",
    phase_unit: str = "deg",
) -> ComplexArray:
    """Convert either Re/Im or magnitude/phase to complex S11 without unwrapping.

    ``magnitude_unit='db'`` means 20*log10(abs(S11)), usually negative;
    it does not mean positive return loss. Phase may be degrees or radians.
    """
    cartesian = real is not None or imag is not None
    polar = magnitude is not None or phase is not None
    if cartesian == polar:
        raise ValueError("Supply exactly one complete pair: real/imag or magnitude/phase.")
    if cartesian:
        if real is None or imag is None:
            raise ValueError("Both real and imag are required.")
        first, second = np.asarray(real, float), np.asarray(imag, float)
    else:
        if magnitude is None or phase is None:
            raise ValueError("Both magnitude and phase are required; magnitude alone is insufficient.")
        first, second = np.asarray(magnitude, float), np.asarray(phase, float)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("The two S11 arrays must be nonempty, one-dimensional and equal in shape.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("S11 input contains NaN or infinity.")
    if cartesian:
        return np.asarray(first + 1j * second, dtype=complex)
    if magnitude_unit.lower() == "db":
        with np.errstate(over="ignore"):
            amplitude = np.power(10.0, first / 20.0)
    elif magnitude_unit.lower() == "linear":
        if np.any(first < 0):
            raise ValueError("Linear magnitude cannot be negative.")
        amplitude = first
    else:
        raise ValueError("magnitude_unit must be 'linear' or 'db'.")
    if phase_unit.lower() in ("deg", "degrees"):
        angle = np.deg2rad(second)
    elif phase_unit.lower() in ("rad", "radians"):
        angle = second
    else:
        raise ValueError("phase_unit must be 'deg' or 'rad'.")
    result = amplitude * np.exp(1j * angle)
    if not np.all(np.isfinite(result)):
        raise ValueError("Magnitude conversion overflowed; check the data and units.")
    return np.asarray(result, dtype=complex)


@dataclass
class S11Data:
    frequency_hz: FloatArray
    s11: ComplexArray
    z0_ohm: float = 50.0
    source: str = "array"

    def __post_init__(self) -> None:
        self.frequency_hz = _frequencies(self.frequency_hz).copy()
        self.s11 = np.asarray(self.s11, dtype=complex).copy()
        if self.s11.shape != self.frequency_hz.shape or not np.all(np.isfinite(self.s11)):
            raise ValueError("s11 must be finite and have the same shape as frequency_hz.")
        self.z0_ohm = _positive(self.z0_ohm, "z0_ohm")


def _float_token(value: str) -> float:
    return float(value.strip().replace("D", "E").replace("d", "e"))


def _load_touchstone(path: Path, z0_ohm: float | None) -> S11Data:
    """Read Touchstone 1.x S11, including wrapped two-port data records."""
    port_count = int(path.suffix.lower()[2:-1])
    unit, data_kind, representation, reference = "ghz", "s", "ma", 50.0
    tokens: list[float] = []
    option_seen = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            raise ValueError("Touchstone 2.x keywords are not supported; export Touchstone 1.x.")
        if line.startswith("#"):
            if option_seen or tokens:
                raise ValueError("Touchstone option line must occur at most once, before data.")
            option_seen = True
            fields = line[1:].lower().split()
            if len(fields) >= 1:
                unit = fields[0]
            if len(fields) >= 2:
                data_kind = fields[1]
            if len(fields) >= 3:
                representation = fields[2]
            if len(fields) > 3:
                if len(fields) != 5 or fields[3] != "r":
                    raise ValueError("Expected Touchstone option '# unit S RI/MA/DB R reference'.")
                reference = _float_token(fields[4])
            continue
        try:
            tokens.extend(_float_token(item) for item in line.split())
        except ValueError as error:
            raise ValueError(f"Invalid Touchstone numeric data at line {line_number}.") from error
    if unit not in FREQUENCY_SCALES or data_kind != "s" or representation not in ("ri", "ma", "db"):
        raise ValueError("Touchstone must contain S-parameters in RI, MA or DB format.")
    reference = _positive(reference, "Touchstone reference resistance")
    if z0_ohm is not None and not np.isclose(z0_ohm, reference, rtol=1e-12, atol=0):
        raise ValueError("Requested R0 differs from Touchstone reference; renormalize the data first.")
    width = 1 + 2 * port_count**2
    if not tokens or len(tokens) % width:
        raise ValueError("Incomplete Touchstone data record, or unsupported noise data section.")
    records = np.asarray(tokens).reshape(-1, width)
    # In both one- and two-port Touchstone 1.x records, S11 is the first pair.
    s11 = _convert_pair(records[:, 1], records[:, 2], representation, "deg")
    return S11Data(records[:, 0] * FREQUENCY_SCALES[unit], s11, reference, str(path))


def _convert_pair(first: ArrayLike, second: ArrayLike, representation: str, phase_unit: str) -> ComplexArray:
    if representation == "ri":
        return complex_s11(real=first, imag=second)
    if representation in ("ma", "db"):
        return complex_s11(magnitude=first, phase=second,
                           magnitude_unit="db" if representation == "db" else "linear",
                           phase_unit=phase_unit)
    raise ValueError("representation must be 'ri', 'ma' or 'db'.")


def load_s11(
    path: str | Path,
    *,
    representation: str = "auto",
    frequency_unit: str | None = None,
    phase_unit: str = "deg",
    delimiter: str | None = None,
    frequency_column: str | int = 0,
    first_column: str | int = 1,
    second_column: str | int = 2,
    z0_ohm: float | None = None,
) -> S11Data:
    """Read CSV/TSV/text or Touchstone 1.x .s1p/.s2p without discarding phase.

    CSV column selectors may be zero-based indices or exact header names.
    Defaults select the first three columns. An explicit frequency unit is
    required unless the selected header includes Hz, kHz, MHz or GHz.
    Auto RI/MA/DB detection requires informative selected column headers.
    Touchstone always uses its own unit/format metadata and degree phases.
    """
    path = Path(path)
    if path.suffix.lower() in (".s1p", ".s2p"):
        return _load_touchstone(path, z0_ohm)
    if representation not in ("auto", "ri", "ma", "db"):
        raise ValueError("representation must be auto, ri, ma or db.")
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith(("#", "!"))]
    if not lines:
        raise ValueError(f"No data in {path}.")
    if delimiter is None:
        delimiter = next((value for value in (",", ";", "\t") if value in lines[0]), None)
    rows = list(csv.reader(lines, delimiter=delimiter)) if delimiter else [line.split() for line in lines]
    try:
        [_float_token(value) for value in rows[0]]
        header: list[str] | None = None
    except ValueError:
        header = [value.strip() for value in rows.pop(0)]
    if not rows:
        raise ValueError("Input contains a header but no numeric rows.")

    def column_index(selector: str | int) -> int:
        if isinstance(selector, int) or str(selector).isdigit():
            index = int(selector)
        elif header is not None and str(selector) in header:
            index = header.index(str(selector))
        else:
            raise ValueError(f"Column {selector!r} was not found in the header.")
        if index < 0:
            raise ValueError("Column indices cannot be negative.")
        return index

    indices = [column_index(item) for item in (frequency_column, first_column, second_column)]
    if len(set(indices)) != 3:
        raise ValueError("Frequency and the two S11 columns must be distinct.")
    if header is not None and max(indices) >= len(header):
        raise ValueError("A selected column is missing from the header.")
    try:
        selected = np.asarray([[_float_token(row[index]) for index in indices] for row in rows])
    except (IndexError, ValueError) as error:
        raise ValueError("A data row has missing columns or nonnumeric values.") from error
    headers = [header[index].lower() for index in indices] if header is not None else ["", "", ""]
    if frequency_unit is None:
        match = re.search(r"(?<![a-z])(ghz|mhz|khz|hz)(?![a-z])", headers[0])
        if match is None:
            raise ValueError("Specify frequency_unit: Hz, kHz, MHz or GHz; no unit found in the header.")
        frequency_unit = match.group(1)
    if frequency_unit.lower() not in FREQUENCY_SCALES:
        raise ValueError("frequency_unit must be Hz, kHz, MHz or GHz.")
    if representation == "auto":
        first, second = headers[1:]
        is_phase = "phase" in second or "ang" in second
        if ("real" in first or re.search(r"\bre\b|re\(", first)) and (
            "imag" in second or re.search(r"\bim\b|im\(", second)
        ):
            representation = "ri"
        elif is_phase and "db" in first:
            representation = "db"
        elif is_phase and ("mag" in first or "abs" in first):
            representation = "ma"
        else:
            raise ValueError("Cannot determine S11 representation; select ri, ma or db explicitly.")
    s11 = _convert_pair(selected[:, 1], selected[:, 2], representation, phase_unit)
    return S11Data(selected[:, 0] * FREQUENCY_SCALES[frequency_unit.lower()], s11,
                   50.0 if z0_ohm is None else z0_ohm, str(path))


@dataclass
class FitResult:
    antenna: str
    parameters: dict[str, float]
    frequency_hz: FloatArray
    s11_measured: ComplexArray
    s11_fitted: ComplexArray
    success: bool
    message: str
    diagnostics: dict
    model: dict

    def to_dict(self) -> dict:
        """JSON-compatible parameters in SI units and fit diagnostics."""
        return {"antenna": self.antenna, "success": self.success, "message": self.message,
                "parameters": self.parameters, "units": {"R_l": "ohm", "L": "H", "C": "F",
                "M": "H", "k": "dimensionless"}, "model": self.model, "diagnostics": self.diagnostics}

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        """Write parameter JSON and measured/fitted complex S11 CSV."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(str(prefix) + ".json")
        csv_path = Path(str(prefix) + ".csv")
        json_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frequency_hz", "s11_real", "s11_imag", "fit_real", "fit_imag",
                             "s11_db", "fit_db", "s11_phase_deg", "fit_phase_deg"])
            measured_db = 20 * np.log10(np.maximum(np.abs(self.s11_measured), 1e-300))
            fitted_db = 20 * np.log10(np.maximum(np.abs(self.s11_fitted), 1e-300))
            writer.writerows(zip(self.frequency_hz, self.s11_measured.real, self.s11_measured.imag,
                                  self.s11_fitted.real, self.s11_fitted.imag, measured_db, fitted_db,
                                  np.angle(self.s11_measured, deg=True), np.angle(self.s11_fitted, deg=True)))
        return json_path, csv_path

    def print_table(self) -> None:
        """Print physical loss elements, magnetic couplings, and diagnostics."""
        print(f"\nAnt. {self.antenna} | {'CONVERGED' if self.success else 'NOT CONVERGED'}")
        print("+------+----------------+----------------+----------------+----------------+")
        print("| Mode | R_loss [ohm]   | L [nH]         | C [pF]         | f_bare [GHz]   |")
        print("+------+----------------+----------------+----------------+----------------+")
        for mode in MODES[self.antenna]:
            resistance, inductance, capacitance = (self.parameters[f"R_l{mode}"],
                                                    self.parameters[f"L_{mode}"], self.parameters[f"C_{mode}"])
            frequency = 1 / (2 * np.pi * math.sqrt(inductance * capacitance))
            print(f"| {mode:^4} | {resistance:14.7g} | {inductance*1e9:14.7g} |"
                  f" {capacitance*1e12:14.7g} | {frequency*1e-9:14.7g} |")
        print("+------+----------------+----------------+----------------+----------------+")
        for pair in ("ps", "pt", "st"):
            if f"k_{pair}" in self.parameters:
                print(f"  k_{pair} = {self.parameters[f'k_{pair}']:+.8f}"
                      f"    M_{pair} = {self.parameters[f'M_{pair}']*1e9:+.8f} nH")
        print(f"  Complex S11 RMSE: {self.diagnostics['complex_rmse']:.6g}")
        print(f"  Jacobian rank: {self.diagnostics['jacobian_rank']}/{self.diagnostics['free_parameter_count']}")
        if self.diagnostics["structurally_underdetermined"]:
            print(f"  One-port structural coefficient limit: {self.diagnostics['structural_response_coefficient_limit']}")
        print(f"  {self.message}")
        for warning in self.diagnostics["warnings"]:
            print(f"  NOTE: {warning}")


class FiltennaRLCFitter:
    """Full intrinsic-admittance model and bounded complex least-squares fitter.

    ``d`` specifies the real radiation-port coupling vector, in p,s,t order,
    or a mapping with exactly the active mode keys. Ant. I requires d_p=1
    to preserve its prescribed scalar formula. No radiation conductance is
    added to intrinsic Y: Rr is applied once, at the radiation port.
    """

    def __init__(self, antenna: str | int = "III", *, R0: float = 50.0,
                 Rr: float = 50.0, n_f: float = 1.0, R_f: float = 0.0,
                 L_f: float = 0.0, d: Sequence[float] | Mapping[str, float] | None = None) -> None:
        self.antenna = _configuration(antenna)
        self.modes = MODES[self.antenna]
        self.R0 = _positive(R0, "R0")
        self.Rr = _positive(Rr, "Rr")
        self.n_f = _positive(n_f, "n_f")
        self.R_f = _positive(R_f, "R_f", allow_zero=True)
        self.L_f = _positive(L_f, "L_f", allow_zero=True)
        self.d_was_default = d is None
        if isinstance(d, Mapping):
            if set(d) != set(self.modes):
                raise ValueError("d must contain exactly the active mode keys.")
            d = [d[mode] for mode in self.modes]
        self.d = np.ones(len(self.modes)) if d is None else np.asarray(d, dtype=float).copy()
        if self.d.shape != (len(self.modes),) or not np.all(np.isfinite(self.d)):
            raise ValueError("d must be a finite real vector with one entry per active mode.")
        if self.antenna == "I" and self.d[0] != 1.0:
            raise ValueError("Ant. I requires d_p=1 to match the specified radiation resistance formula.")
        self.pairs = tuple(a + b for i, a in enumerate(self.modes) for b in self.modes[i + 1:])
        self.element_names = tuple(name for mode in self.modes for name in (f"R_l{mode}", f"L_{mode}", f"C_{mode}"))
        self.parameter_names = self.element_names + tuple(f"k_{pair}" for pair in self.pairs)

    def _validate_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        missing = set(self.parameter_names) - parameters.keys()
        extra = parameters.keys() - set(self.parameter_names) - {f"M_{pair}" for pair in self.pairs}
        if missing or extra:
            raise ValueError(f"Invalid parameter keys; missing={sorted(missing)}, unexpected={sorted(extra)}.")
        values = {name: _positive(parameters[name], name) for name in self.element_names}
        for pair in self.pairs:
            value = float(parameters[f"k_{pair}"])
            if not math.isfinite(value) or abs(value) >= 1:
                raise ValueError(f"k_{pair} must be finite and strictly between -1 and 1.")
            values[f"k_{pair}"] = value
        self._inductance_matrix(values, validate=True)
        for pair in self.pairs:
            if f"M_{pair}" in parameters:
                expected = values[f"k_{pair}"] * math.sqrt(values[f"L_{pair[0]}"] * values[f"L_{pair[1]}"])
                if not np.isclose(parameters[f"M_{pair}"], expected, rtol=1e-9, atol=1e-24):
                    raise ValueError(f"M_{pair} is inconsistent with k_{pair} and the inductances.")
        return values

    def _inductance_matrix(self, parameters: Mapping[str, float], *, validate: bool = False) -> FloatArray:
        inductances = np.asarray([parameters[f"L_{mode}"] for mode in self.modes])
        correlation = np.eye(len(self.modes))
        for i, first in enumerate(self.modes):
            for j in range(i + 1, len(self.modes)):
                correlation[i, j] = correlation[j, i] = parameters[f"k_{first}{self.modes[j]}"]
        if validate:
            try:
                np.linalg.cholesky(correlation)
            except np.linalg.LinAlgError as error:
                raise ValueError("The full inductance matrix must be positive definite; |k_ij|<1 alone is insufficient.") from error
        return correlation * np.sqrt(inductances[:, None] * inductances[None, :])

    def inductance_matrix(self, parameters: Mapping[str, float]) -> FloatArray:
        """Return the reciprocal, positive definite full L matrix in henries."""
        return self._inductance_matrix(self._validate_parameters(parameters))

    def _admittance(self, frequencies: FloatArray, parameters: Mapping[str, float]) -> ComplexArray:
        omega = 2 * np.pi * frequencies
        inverse_inductance = np.linalg.solve(self._inductance_matrix(parameters), np.eye(len(self.modes)))
        admittance = inverse_inductance[None, :, :] / (1j * omega[:, None, None])
        for index, mode in enumerate(self.modes):
            admittance[:, index, index] += 1 / parameters[f"R_l{mode}"] + 1j * omega * parameters[f"C_{mode}"]
        return admittance

    def admittance_matrix(self, frequency_hz: ArrayLike, parameters: Mapping[str, float]) -> ComplexArray:
        """Return Y(w), with shape (frequency_count, mode_count, mode_count)."""
        return self._admittance(_frequencies(frequency_hz), self._validate_parameters(parameters))

    def _two_port(self, frequencies: FloatArray, parameters: Mapping[str, float]) -> ComplexArray:
        admittance = self._admittance(frequencies, parameters)
        ports = np.zeros((len(self.modes), 2))
        ports[0, 0] = self.n_f
        ports[:, 1] = self.d
        rhs = np.broadcast_to(ports, (frequencies.size,) + ports.shape)
        modal_voltages = np.linalg.solve(admittance, rhs)
        impedance = np.einsum("ia,fib->fab", ports, modal_voltages)
        impedance[:, 0, 0] += self.R_f + 1j * 2 * np.pi * frequencies * self.L_f
        return impedance

    def two_port_impedance(self, frequency_hz: ArrayLike, parameters: Mapping[str, float]) -> ComplexArray:
        """Return Z11,Z12,Z21,Z22 using the complete Y inverse, shape (N,2,2).

        This matrix operation reproduces the supplied Delta_2/Delta_3
        cofactors, including all u*v*w and cross-radiation terms.
        """
        return self._two_port(_frequencies(frequency_hz), self._validate_parameters(parameters))

    def input_impedance(self, frequency_hz: ArrayLike, parameters: Mapping[str, float]) -> ComplexArray:
        """Return the input impedance after loading the radiation port by Rr.

        The loaded-admittance solve is the Woodbury-equivalent expression
        Z11-Z12*Z21/(Rr+Z22), and avoids subtractive cancellation.
        """
        frequencies = _frequencies(frequency_hz)
        values = self._validate_parameters(parameters)
        loaded = self._admittance(frequencies, values) + np.outer(self.d, self.d)[None, :, :] / self.Rr
        excitation = np.zeros((frequencies.size, len(self.modes), 1))
        excitation[:, 0, 0] = 1.0
        modal_voltages = np.linalg.solve(loaded, excitation)
        return self.R_f + 1j * 2 * np.pi * frequencies * self.L_f + self.n_f**2 * modal_voltages[:, 0, 0]

    def _predict(self, frequencies: FloatArray, parameters: Mapping[str, float]) -> ComplexArray:
        z = self._two_port(frequencies, parameters)
        # Evaluate the exact requested two-port termination formula. Scaling
        # each impedance by a common factor cancels out of this ratio and
        # prevents unnecessary overflow in the quadratic products.
        scale = np.maximum(np.max(np.abs(z), axis=(1, 2)), max(self.R0, self.Rr))
        z11, z12, z21, z22 = (z[:, i, j] / scale for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
        rr, r0 = self.Rr / scale, self.R0 / scale
        numerator_common = z11 * rr + z11 * z22 - z12 * z21
        reference_term = r0 * (rr + z22)
        return (numerator_common - reference_term) / (numerator_common + reference_term)

    def predict_s11(self, frequency_hz: ArrayLike, parameters: Mapping[str, float]) -> ComplexArray:
        """Evaluate the specified complex S11 at the feed reference plane."""
        return self._predict(_frequencies(frequency_hz), self._validate_parameters(parameters))

    def _initial_guess(self, frequencies: FloatArray, measured: ComplexArray) -> dict[str, float]:
        order = np.argsort(frequencies)
        ordered_frequency, ordered_s11 = frequencies[order], measured[order]
        minima, properties = find_peaks(-np.abs(ordered_s11), prominence=0.005)
        if minima.size:
            minima = minima[np.argsort(properties["prominences"])[::-1]]
        guesses = [float(ordered_frequency[index]) for index in minima[:len(self.modes)]]
        middle = float(np.median(frequencies))
        fallback = np.linspace(float(frequencies.min()), float(frequencies.max()), len(self.modes) + 2)[1:-1]
        for value in fallback:
            if len(guesses) >= len(self.modes):
                break
            guesses.append(float(value))
        guesses[0] = guesses[0] if guesses else middle
        result: dict[str, float] = {}
        for mode, resonance in zip(self.modes, guesses):
            inductance = 3e-9
            result.update({f"R_l{mode}": 1200.0, f"L_{mode}": inductance,
                           f"C_{mode}": 1 / ((2 * np.pi * resonance)**2 * inductance)})
        for pair in self.pairs:
            result[f"k_{pair}"] = 0.08
        if self.antenna == "I":
            # Linear susceptance regression is initialization only; the
            # objective always fits complex S11 rather than converted Z.
            usable = (np.abs(1 - measured) > 1e-5)
            z_input = self.R0 * (1 + measured[usable]) / (1 - measured[usable])
            z_mode = (z_input - self.R_f - 1j * 2 * np.pi * frequencies[usable] * self.L_f) / self.n_f**2
            valid = np.abs(z_mode) > 1e-8
            if np.count_nonzero(valid) >= 3:
                admittance = 1 / z_mode[valid]
                omega = 2 * np.pi * frequencies[usable][valid]
                design = np.column_stack((omega * 1e-12, -1 / (omega * 1e-9)))
                capacitance_pf, inverse_inductance_nh = np.linalg.lstsq(design, admittance.imag, rcond=None)[0]
                conductance = float(np.median(admittance.real) - 1 / self.Rr)
                if capacitance_pf > 0 and inverse_inductance_nh > 0 and conductance > 0:
                    result.update(R_lp=1 / conductance, L_p=1e-9 / inverse_inductance_nh,
                                  C_p=capacitance_pf * 1e-12)
        return result

    def fit(
        self,
        frequency_hz: ArrayLike,
        s11: ArrayLike | None = None,
        *,
        real: ArrayLike | None = None,
        imag: ArrayLike | None = None,
        magnitude: ArrayLike | None = None,
        phase: ArrayLike | None = None,
        magnitude_unit: str = "linear",
        phase_unit: str = "deg",
        initial: Mapping[str, float] | None = None,
        fixed_parameters: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        weights: ArrayLike | None = None,
        n_starts: int = 12,
        max_nfev: int = 3000,
        loss: str = "linear",
        f_scale: float = 0.02,
        seed: int | None = 42,
    ) -> FitResult:
        """Fit both real and imaginary S11 with scipy.optimize.least_squares.

        RLC values are optimized in log coordinates. A partial-correlation
        parameterization guarantees a positive definite full inductance
        matrix at every step; all three couplings may have either sign.
        Multiple starts are deterministic for a fixed seed.

        ``fixed_parameters`` accepts any active R_loss, L or C in SI units.
        Couplings remain free; arbitrary fixed coupling constraints are not
        implemented because independent bounds do not ensure positive
        definiteness. ``bounds`` overrides positive finite (lower,upper)
        limits for RLC elements. Defaults: R_loss 1..1e7 ohm, L 0.01..100 nH,
        C 0.001..100 pF. Weights multiply squared complex residuals; use
        1/sigma**2 for per-component independent noise variance.

        ``success`` reports optimizer termination, not uniqueness or physical
        validity of an inferred antenna geometry. See diagnostics warnings.
        """
        frequencies = _frequencies(frequency_hz).copy()
        alternatives = any(value is not None for value in (real, imag, magnitude, phase))
        if s11 is not None and alternatives:
            raise ValueError("Supply s11 or its real/imag or magnitude/phase components, not both.")
        measured = (np.asarray(s11, dtype=complex) if s11 is not None else
                    complex_s11(real=real, imag=imag, magnitude=magnitude, phase=phase,
                                magnitude_unit=magnitude_unit, phase_unit=phase_unit))
        data = S11Data(frequencies, measured, self.R0)
        measured = data.s11
        if np.unique(frequencies).size < 3:
            raise ValueError("Fitting requires at least three distinct frequencies.")
        if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
            raise ValueError("n_starts must be a positive integer.")
        if not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer.")
        if loss not in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
            raise ValueError("Unknown least_squares loss.")
        f_scale = _positive(f_scale, "f_scale")
        fixed = dict(fixed_parameters or {})
        limits = dict(bounds or {})
        for collection, label in ((fixed, "fixed_parameters"), (limits, "bounds")):
            unexpected = set(collection) - set(self.element_names)
            if unexpected:
                raise ValueError(f"{label} accepts active RLC element keys only: unexpected {sorted(unexpected)}.")
        fixed = {name: _positive(value, name) for name, value in fixed.items()}
        for name in self.element_names:
            default = (1.0, 1e7) if name.startswith("R_") else ((1e-11, 1e-7) if name.startswith("L_") else (1e-15, 1e-10))
            pair = limits.get(name, default)
            if len(pair) != 2:
                raise ValueError(f"bounds[{name!r}] must contain two values.")
            lower, upper = (_positive(value, f"{name} bound") for value in pair)
            if lower >= upper:
                raise ValueError(f"Lower bound must be less than upper bound for {name}.")
            if name in fixed and not lower <= fixed[name] <= upper:
                raise ValueError(f"Fixed {name} is outside its bounds.")
            limits[name] = (lower, upper)
        initial_values = self._initial_guess(frequencies, measured)
        if initial is not None:
            unexpected = set(initial) - set(self.parameter_names) - {f"M_{pair}" for pair in self.pairs}
            if unexpected:
                raise ValueError(f"Unknown initial parameter keys: {sorted(unexpected)}.")
            for name in self.parameter_names:
                if name in initial:
                    initial_values[name] = float(initial[name])
        initial_values.update(fixed)
        # M is an optional physical alias for an initial k. Validate a saved
        # result containing both, rather than silently ignoring its M values.
        if initial is not None:
            for pair in self.pairs:
                mutual_name, coupling_name = f"M_{pair}", f"k_{pair}"
                if mutual_name in initial:
                    mutual = float(initial[mutual_name])
                    if not math.isfinite(mutual):
                        raise ValueError(f"Initial {mutual_name} must be finite.")
                    first_l = _positive(initial_values[f"L_{pair[0]}"], f"L_{pair[0]}")
                    second_l = _positive(initial_values[f"L_{pair[1]}"], f"L_{pair[1]}")
                    if coupling_name not in initial:
                        initial_values[coupling_name] = mutual / math.sqrt(first_l * second_l)
                    initial_values[mutual_name] = mutual
        initial_values = self._validate_parameters(initial_values)
        supplied_initial = set(initial or {}) - set(fixed)
        for name in self.element_names:
            if name in supplied_initial and not limits[name][0] <= initial_values[name] <= limits[name][1]:
                raise ValueError(f"Initial {name} is outside its bounds.")
            initial_values[name] = float(np.clip(initial_values[name], *limits[name]))
        free_elements = [name for name in self.element_names if name not in fixed]
        scales = {name: (1.0 if name.startswith("R_") else 1e-9 if name.startswith("L_") else 1e-12)
                  for name in free_elements}
        lower = [math.log(limits[name][0] / scales[name]) for name in free_elements]
        upper = [math.log(limits[name][1] / scales[name]) for name in free_elements]
        coupling_limit = 0.999
        lower.extend([-coupling_limit] * len(self.pairs))
        upper.extend([coupling_limit] * len(self.pairs))
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)

        def encode(parameters: Mapping[str, float]) -> FloatArray:
            vector = [math.log(parameters[name] / scales[name]) for name in free_elements]
            if self.pairs:
                a = parameters["k_ps"]
                vector.append(a)
                if self.antenna == "III":
                    b = parameters["k_pt"]
                    c = (parameters["k_st"] - a * b) / math.sqrt((1 - a*a) * (1 - b*b))
                    vector.extend((b, c))
            return np.clip(np.asarray(vector), lower_array + 1e-10, upper_array - 1e-10)

        def decode(vector: FloatArray) -> dict[str, float]:
            parameters = dict(fixed)
            for index, name in enumerate(free_elements):
                parameters[name] = float(scales[name] * math.exp(vector[index]))
            if self.pairs:
                offset = len(free_elements)
                a = float(vector[offset])
                parameters["k_ps"] = a
                if self.antenna == "III":
                    b, c = float(vector[offset + 1]), float(vector[offset + 2])
                    parameters["k_pt"] = b
                    parameters["k_st"] = a*b + c*math.sqrt((1 - a*a) * (1 - b*b))
            return parameters

        if weights is None:
            weight_values = np.ones(frequencies.size)
        else:
            weight_values = np.asarray(weights, dtype=float)
            if weight_values.shape != frequencies.shape or not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0):
                raise ValueError("weights must be a finite, nonnegative array matching the frequencies.")
        positive_weights = weight_values > 0
        if np.unique(frequencies[positive_weights]).size < 3:
            raise ValueError("At least three distinct frequencies must have positive weight.")
        free_count = len(lower_array)
        if 2 * np.count_nonzero(positive_weights) <= free_count:
            raise ValueError("Not enough positively weighted complex samples for the free parameters.")
        sqrt_weights = np.sqrt(weight_values)

        def residual(vector: FloatArray) -> FloatArray:
            difference = (self._predict(frequencies, decode(vector)) - measured) * sqrt_weights
            return np.concatenate((difference.real, difference.imag))

        start = encode(initial_values)
        generator = np.random.default_rng(seed)
        best = None
        total_nfev = 0
        attempts = []
        if free_count == 0:
            fitted = self._predict(frequencies, fixed)
            parameters = fixed.copy()
            jacobian_singular = np.array([], dtype=float)
            rank, condition = 0, None
            success, message = True, "All circuit elements were fixed; evaluated without optimization."
            best_cost = 0.5 * float(np.sum(np.abs((fitted - measured) * sqrt_weights)**2))
            active_bounds: list[str] = []
        else:
            for attempt in range(n_starts):
                if attempt == 0:
                    candidate = start.copy()
                else:
                    # Keep paired LC resonance changes modest while exploring
                    # impedance scale, loss, and either sign of each coupling.
                    candidate_parameters = initial_values.copy()
                    for mode in self.modes:
                        scale_shift = generator.normal(0, 0.5)
                        frequency_shift = generator.normal(0, 0.055 if initial is not None else 0.12)
                        for name, multiplier in ((f"R_l{mode}", math.exp(generator.normal(0, 0.8))),
                                                 (f"L_{mode}", math.exp(scale_shift)),
                                                 (f"C_{mode}", math.exp(-scale_shift - 2*frequency_shift))):
                            if name not in fixed:
                                candidate_parameters[name] = float(np.clip(candidate_parameters[name] * multiplier, *limits[name]))
                    candidate = encode(candidate_parameters)
                    if self.pairs:
                        candidate[len(free_elements):] = generator.uniform(-0.35, 0.35, len(self.pairs))
                    candidate = np.clip(candidate, lower_array + 1e-9, upper_array - 1e-9)
                optimized = least_squares(residual, candidate, bounds=(lower_array, upper_array),
                                          method="trf", jac="3-point", x_scale="jac", loss=loss,
                                          f_scale=f_scale, max_nfev=max_nfev,
                                          ftol=1e-11, xtol=1e-11, gtol=1e-11)
                total_nfev += int(optimized.nfev)
                attempts.append({"start": attempt + 1, "cost": float(optimized.cost),
                                 "success": bool(optimized.success), "nfev": int(optimized.nfev)})
                if best is None or optimized.cost < best.cost:
                    best = optimized
                if best.success and np.max(np.abs(best.fun)) < 1e-10:
                    break
            parameters = decode(best.x)
            fitted = self._predict(frequencies, parameters)
            jacobian_singular = np.linalg.svd(best.jac, compute_uv=False)
            threshold = jacobian_singular[0] * 1e-7 if jacobian_singular.size else 0.0
            rank = int(np.count_nonzero(jacobian_singular > threshold))
            condition = (float(jacobian_singular[0] / jacobian_singular[-1])
                         if jacobian_singular.size and jacobian_singular[-1] > 0 else None)
            success, message, best_cost = bool(best.success), str(best.message), float(best.cost)
            coordinate_names = free_elements + (["k_ps"] if self.antenna == "II" else
                                                ["k_ps", "k_pt", "partial_st_given_p"] if self.antenna == "III" else [])
            active_bounds = [name for name, value in zip(coordinate_names, best.active_mask) if value != 0]
        for pair in self.pairs:
            parameters[f"M_{pair}"] = parameters[f"k_{pair}"] * math.sqrt(parameters[f"L_{pair[0]}"] * parameters[f"L_{pair[1]}"])
        warnings = []
        if self.antenna == "III":
            warnings.append("Ant. III has at most 11 independent one-port coefficients for 12 RLC/coupling elements; "
                            "use independently known fixed elements and inspect rank before interpreting physical values.")
        if any(self.d[index] == 0 for index in range(1, len(self.modes))):
            warnings.append("A hidden mode has zero radiation coupling; its absolute element scale may be unidentifiable.")
        if rank < free_count:
            warnings.append("The weighted Jacobian is rank deficient at relative tolerance 1e-7; fitted physical parameters are not unique or are poorly resolved.")
        elif condition is not None and condition > 1e6:
            warnings.append("The fit is ill-conditioned; small data errors may produce large parameter changes.")
        if self.d_was_default and self.antenna != "I":
            warnings.append("d was not supplied: all radiation coupling entries were fixed to 1.")
        if np.any(np.abs(measured) > 1 + 1e-6):
            warnings.append("Some measured |S11| values exceed 1; data were retained without clipping.")
        if active_bounds:
            warnings.append("An optimum touches parameter bounds: " + ", ".join(active_bounds))
        if not success:
            warnings.append("Optimization did not converge; this is the best candidate, not a completed parameter extraction.")
        if fixed:
            warnings.append("Fixed elements (not inferred from S11): " + ", ".join(f"{name}={value:g}" for name, value in sorted(fixed.items())))
        error = fitted - measured
        diagnostics = {
            "complex_rmse": float(np.sqrt(np.mean(np.abs(error)**2))),
            "weighted_complex_rmse": float(np.sqrt(np.sum(weight_values * np.abs(error)**2) / np.sum(weight_values))),
            "maximum_complex_error": float(np.max(np.abs(error))),
            "cost": best_cost, "loss": loss, "f_scale": f_scale,
            "sample_count": int(frequencies.size), "free_parameter_count": free_count,
            "structural_response_coefficient_limit": 4 * len(self.modes) - 1,
            "structurally_underdetermined": free_count > 4 * len(self.modes) - 1,
            "jacobian_rank": rank, "jacobian_rank_relative_tolerance": 1e-7,
            "jacobian_condition": condition, "jacobian_singular_values": jacobian_singular.tolist(),
            "jacobian_coordinates": "log(R_loss), log(L/nH), log(C/pF), partial correlations; robust-loss modified if applicable",
            "fixed_parameters": fixed, "active_bounds": active_bounds,
            "total_nfev": total_nfev, "attempts": attempts, "seed": seed,
            "frequency_min_hz": float(frequencies.min()), "frequency_max_hz": float(frequencies.max()),
            "warnings": warnings,
        }
        model = {"intrinsic_admittance": "diag(1/R_li) + j*omega*diag(C_i) + inv(L)/(j*omega)",
                 "R0_ohm": self.R0, "Rr_ohm": self.Rr, "n_f": self.n_f,
                 "R_f_ohm": self.R_f, "L_f_H": self.L_f,
                 "d": dict(zip(self.modes, self.d.tolist())), "d_was_default": self.d_was_default,
                 "coupling_parameter_limit": coupling_limit, "element_bounds_SI": limits}
        return FitResult(self.antenna, parameters, frequencies, measured, fitted, success, message, diagnostics, model)


def _assignments(values: Sequence[str] | None) -> dict[str, float]:
    result = {}
    for value in values or ():
        name, separator, number = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"Expected a unique parameter=value assignment, received {value!r}.")
        result[name] = _float_token(number)
    return result


def run_demo(output_dir: str | Path = "rlc_fit_results", *, n_starts: int = 6,
             max_nfev: int = 2000, seed: int = 42) -> list[FitResult]:
    """Generate reproducible complex samples over 2.6--4.4 GHz and fit all antennas.

    These are synthetic circuit samples, not HFSS/VNA measurements. Nearby,
    perturbed initial guesses and known L_t for III make this a reproducible
    numerical demonstration, not a claim of blind physical identification.
    """
    print("Synthetic circuit demonstration: 2.6--4.4 GHz, complex S11 (not HFSS/VNA data).")
    frequencies = np.linspace(2.6e9, 4.4e9, 241)
    generator = np.random.default_rng(seed)
    results = []
    for antenna in ("I", "II", "III"):
        modes = MODES[antenna]
        fitter = FiltennaRLCFitter(antenna, n_f=1.1, R_f=0.5, L_f=0.12e-9,
                                  Rr=50.0, d=[1.0, 0.35, 0.25][:len(modes)])
        physical = {}
        for mode, resistance, inductance, frequency in zip(modes, [1300.0, 1800.0, 1600.0],
                                                          [3e-9, 2.5e-9, 2e-9], [3.45e9, 3.7e9, 3.25e9]):
            physical.update({f"R_l{mode}": resistance, f"L_{mode}": inductance,
                             f"C_{mode}": 1 / ((2*np.pi*frequency)**2 * inductance)})
        physical.update({f"k_{pair}": {"ps": 0.11, "pt": 0.075, "st": -0.045}[pair] for pair in fitter.pairs})
        s11 = fitter.predict_s11(frequencies, physical)
        s11 += 2e-5 * (generator.normal(size=s11.size) + 1j*generator.normal(size=s11.size))
        initial = {name: value * (1.08 if name.startswith("R_") else 1.025 if name.startswith("L_") else
                                  0.98 if name.startswith("C_") else 0.9) for name, value in physical.items()}
        fixed = {"L_t": physical["L_t"]} if antenna == "III" else {}
        result = fitter.fit(frequencies, s11, initial=initial, fixed_parameters=fixed,
                            n_starts=n_starts, max_nfev=max_nfev, seed=seed)
        result.print_table()
        json_path, csv_path = result.save(Path(output_dir) / f"ant_{antenna.lower()}_fit")
        print(f"  Saved: {json_path} and {csv_path}")
        # Save the actual generating circuit separately from fitted values.
        truth_path = Path(output_dir) / f"ant_{antenna.lower()}_synthetic_truth.json"
        truth_path.write_text(json.dumps({"synthetic": True, "parameters": physical,
                                         "fixed_parameters": fixed, "model": result.model,
                                         "noise_standard_deviation_per_component": 2e-5}, indent=2) + "\n", encoding="utf-8")
        results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the intrinsic-admittance RLC filtenna circuit to complex S11.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, help="Complex S11 CSV/text or Touchstone .s1p/.s2p file.")
    parser.add_argument("--antenna", choices=("I", "II", "III"), default="III")
    parser.add_argument("--demo", action="store_true", help="Fit synthetic examples for all three configurations.")
    parser.add_argument("--representation", choices=("auto", "ri", "ma", "db"), default="auto")
    parser.add_argument("--frequency-unit", choices=("Hz", "kHz", "MHz", "GHz"))
    parser.add_argument("--phase-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--delimiter", help="CSV separator; autodetected when omitted.")
    parser.add_argument("--frequency-column", default="0", help="Zero-based index or exact header.")
    parser.add_argument("--first-column", default="1", help="Real, magnitude or dB column.")
    parser.add_argument("--second-column", default="2", help="Imaginary or phase column.")
    parser.add_argument("--r0", type=float, default=None, help="Reference resistance; use Touchstone metadata or 50 ohm.")
    parser.add_argument("--rr", type=float, default=50.0, help="Known radiation termination resistance in ohms.")
    parser.add_argument("--nf", type=float, default=1.0, help="Known source transformer ratio n_f:1.")
    parser.add_argument("--rf", type=float, default=0.0, help="Known feed series resistance in ohms.")
    parser.add_argument("--lf-nh", type=float, default=0.0, help="Known feed series inductance in nH.")
    parser.add_argument("--d", type=float, nargs="+", help="Known radiation coupling in p,s,t order.")
    parser.add_argument("--initial", type=Path, help="JSON flat parameter mapping, or result JSON with 'parameters'.")
    parser.add_argument("--fix", action="append", help="Known RLC element in SI units, e.g. --fix L_t=2e-9.")
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"), default="linear")
    parser.add_argument("--f-scale", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("rlc_fit_results"))
    args = parser.parse_args(argv)
    if args.demo and args.input is not None:
        parser.error("Choose --demo or --input, not both.")
    try:
        if args.demo or args.input is None:
            results = run_demo(args.output_dir, n_starts=args.starts, max_nfev=args.max_nfev, seed=args.seed)
            return 0 if all(result.success for result in results) else 2
        data = load_s11(args.input, representation=args.representation, frequency_unit=args.frequency_unit,
                        phase_unit=args.phase_unit, delimiter=args.delimiter, frequency_column=args.frequency_column,
                        first_column=args.first_column, second_column=args.second_column, z0_ohm=args.r0)
        fitter = FiltennaRLCFitter(args.antenna, R0=data.z0_ohm, Rr=args.rr, n_f=args.nf,
                                  R_f=args.rf, L_f=args.lf_nh*1e-9, d=args.d)
        initial = None
        if args.initial is not None:
            initial_document = json.loads(args.initial.read_text(encoding="utf-8-sig"))
            if not isinstance(initial_document, dict):
                raise ValueError("Initial JSON must contain an object mapping parameter names to numbers.")
            initial = initial_document.get("parameters", initial_document)
            if not isinstance(initial, dict):
                raise ValueError("Initial JSON must contain an object mapping parameter names to numbers.")
        result = fitter.fit(data.frequency_hz, data.s11, initial=initial,
                            fixed_parameters=_assignments(args.fix), n_starts=args.starts,
                            max_nfev=args.max_nfev, loss=args.loss, f_scale=args.f_scale, seed=args.seed)
        result.model["data_source"] = data.source
        result.print_table()
        json_path, csv_path = result.save(args.output_dir / f"ant_{args.antenna.lower()}_fit")
        print(f"\nSaved: {json_path}\nSaved: {csv_path}")
        return 0 if result.success else 2
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
