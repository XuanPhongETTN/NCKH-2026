"""Numerical and physical regression tests for the filtenna circuit fitter.

Run with ``python -m unittest -v test_rlc_circuit_fitter``.  The reference
calculations below use independent cofactor and loaded-admittance expressions
so the tests check the physical circuit, not merely the implementation API.
"""

import math
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rlc_circuit_fitter import FiltennaRLCFitter, FitResult, complex_s11, load_s11, main


def mode_parameters(antenna="III"):
    """Return a passive circuit with three separated uncoupled resonances."""
    modes = {"I": "p", "II": "ps", "III": "pst"}[antenna]
    values = {
        "p": (1400.0, 2.5e-9, 3.25e9),
        "s": (2200.0, 1.6e-9, 3.65e9),
        "t": (1800.0, 1.1e-9, 4.05e9),
    }
    parameters = {}
    for mode in modes:
        resistance, inductance, resonance = values[mode]
        parameters[f"R_l{mode}"] = resistance
        parameters[f"L_{mode}"] = inductance
        parameters[f"C_{mode}"] = 1.0 / ((2.0 * math.pi * resonance) ** 2 * inductance)
    if "s" in modes:
        parameters["k_ps"] = 0.10
    if "t" in modes:
        parameters["k_pt"] = 0.07
        parameters["k_st"] = -0.06
    return parameters


def independent_inductance(parameters, modes):
    result = np.diag([parameters[f"L_{mode}"] for mode in modes])
    for i, first in enumerate(modes):
        for j in range(i + 1, len(modes)):
            second = modes[j]
            mutual = parameters[f"k_{first}{second}"] * math.sqrt(
                parameters[f"L_{first}"] * parameters[f"L_{second}"]
            )
            result[i, j] = result[j, i] = mutual
    return result


class CircuitEquationTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.array([2.8e9, 3.25e9, 3.52e9, 3.9e9, 4.4e9])
        self.angular = 2.0 * np.pi * self.frequency

    def test_antenna_i_matches_direct_parallel_rlc_formula(self):
        parameters = mode_parameters("I")
        fitter = FiltennaRLCFitter("I", R0=50, Rr=73, n_f=1.35, R_f=0.8, L_f=0.18e-9)
        admittance = (
            1.0 / 73.0
            + 1.0 / parameters["R_lp"]
            + 1j * (
                self.angular * parameters["C_p"]
                - 1.0 / (self.angular * parameters["L_p"])
            )
        )
        expected_impedance = 0.8 + 1j * self.angular * 0.18e-9 + 1.35 ** 2 / admittance
        expected_s11 = (expected_impedance - 50) / (expected_impedance + 50)
        np.testing.assert_allclose(fitter.input_impedance(self.frequency, parameters), expected_impedance, rtol=2e-12)
        np.testing.assert_allclose(fitter.predict_s11(self.frequency, parameters), expected_s11, rtol=2e-12, atol=1e-13)

    def test_antenna_ii_inverse_inductance_cofactors(self):
        parameters = mode_parameters("II")
        lp, ls = parameters["L_p"], parameters["L_s"]
        mutual = parameters["k_ps"] * math.sqrt(lp * ls)
        inverse = np.array([[ls, -mutual], [-mutual, lp]]) / (lp * ls - mutual ** 2)
        expected = np.array([
            np.diag([1.0 / parameters["R_lp"], 1.0 / parameters["R_ls"]])
            + 1j * angular * np.diag([parameters["C_p"], parameters["C_s"]])
            + inverse / (1j * angular)
            for angular in self.angular
        ])
        actual = FiltennaRLCFitter("II").admittance_matrix(self.frequency, parameters)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)

    def test_antenna_iii_all_cofactors_and_signed_cycle_term(self):
        parameters = mode_parameters("III")
        lp, ls, lt = [parameters[f"L_{mode}"] for mode in "pst"]
        mps = parameters["k_ps"] * math.sqrt(lp * ls)
        mpt = parameters["k_pt"] * math.sqrt(lp * lt)
        mst = parameters["k_st"] * math.sqrt(ls * lt)
        determinant = lp * ls * lt + 2 * mps * mpt * mst - lp * mst ** 2 - ls * mpt ** 2 - lt * mps ** 2
        inverse = np.array([
            [ls * lt - mst ** 2, mpt * mst - mps * lt, mps * mst - mpt * ls],
            [mpt * mst - mps * lt, lp * lt - mpt ** 2, mps * mpt - lp * mst],
            [mps * mst - mpt * ls, mps * mpt - lp * mst, lp * ls - mps ** 2],
        ]) / determinant
        expected = np.array([
            np.diag([1 / parameters[f"R_l{mode}"] for mode in "pst"])
            + 1j * angular * np.diag([parameters[f"C_{mode}"] for mode in "pst"])
            + inverse / (1j * angular)
            for angular in self.angular
        ])
        fitter = FiltennaRLCFitter("III", d=[1.0, 0.75, -0.55])
        np.testing.assert_allclose(fitter.admittance_matrix(self.frequency, parameters), expected, rtol=2e-12, atol=1e-15)
        reversed_sign = dict(parameters, k_st=-parameters["k_st"])
        self.assertGreater(np.max(np.abs(fitter.predict_s11(self.frequency, parameters) - fitter.predict_s11(self.frequency, reversed_sign))), 1e-4)

    def test_two_port_and_radiation_loaded_admittance_are_equivalent(self):
        for antenna, modes, radiation in [("I", "p", [1.0]), ("II", "ps", [1.0, -0.7]), ("III", "pst", [1.0, 0.8, -0.6])]:
            with self.subTest(antenna=antenna):
                parameters = mode_parameters(antenna)
                d = np.array(radiation)
                fitter = FiltennaRLCFitter(antenna, R0=47.0, Rr=61.0, n_f=1.4, R_f=0.3, L_f=0.12e-9, d=radiation)
                inductance = independent_inductance(parameters, modes)
                source = np.zeros(len(modes))
                source[0] = 1.4
                ports = np.column_stack([source, d])
                expected_ports = []
                expected_input = []
                for angular in self.angular:
                    intrinsic = np.diag([1 / parameters[f"R_l{mode}"] for mode in modes]) + 1j * angular * np.diag([parameters[f"C_{mode}"] for mode in modes]) + np.linalg.inv(inductance) / (1j * angular)
                    port_impedance = ports.T @ np.linalg.solve(intrinsic, ports)
                    feed_impedance = 0.3 + 1j * angular * 0.12e-9
                    port_impedance[0, 0] += feed_impedance
                    expected_ports.append(port_impedance)
                    loaded = intrinsic + np.outer(d, d) / 61.0
                    expected_input.append(feed_impedance + source @ np.linalg.solve(loaded, source))
                expected_input = np.array(expected_input)
                np.testing.assert_allclose(fitter.two_port_impedance(self.frequency, parameters), expected_ports, rtol=3e-12, atol=1e-10)
                np.testing.assert_allclose(fitter.input_impedance(self.frequency, parameters), expected_input, rtol=3e-12, atol=1e-10)
                expected_s11 = (expected_input - 47.0) / (expected_input + 47.0)
                np.testing.assert_allclose(fitter.predict_s11(self.frequency, parameters), expected_s11, rtol=3e-12, atol=1e-12)
                self.assertTrue(np.all(np.abs(expected_s11) <= 1.0 + 1e-12))

    def test_pairwise_valid_but_indefinite_inductance_is_rejected(self):
        parameters = mode_parameters("III")
        parameters.update(k_ps=0.9, k_pt=0.9, k_st=-0.9)
        with self.assertRaises(ValueError):
            FiltennaRLCFitter("III").predict_s11(self.frequency, parameters)

    def test_invalid_physical_parameters_and_frequency_are_rejected(self):
        fitter = FiltennaRLCFitter("I")
        for key, value in [("R_lp", 0.0), ("L_p", -1e-9), ("C_p", float("nan"))]:
            with self.subTest(key=key):
                parameters = mode_parameters("I")
                parameters[key] = value
                with self.assertRaises(ValueError):
                    fitter.predict_s11(self.frequency, parameters)
        for frequency in [[0.0, 3e9], [3e9, float("inf")], [-3e9]]:
            with self.subTest(frequency=frequency):
                with self.assertRaises(ValueError):
                    fitter.predict_s11(frequency, mode_parameters("I"))


class ComplexInputTests(unittest.TestCase):
    def test_real_imaginary_and_magnitude_phase_encodings(self):
        expected = np.array([0.2 + 0.3j, -0.4 + 0.1j, -0.2 - 0.1j])
        np.testing.assert_allclose(complex_s11(real=expected.real, imag=expected.imag), expected, atol=1e-15)
        np.testing.assert_allclose(complex_s11(magnitude=np.abs(expected), phase=np.angle(expected, deg=True)), expected, atol=1e-15)
        np.testing.assert_allclose(complex_s11(magnitude=20 * np.log10(np.abs(expected)), phase=np.angle(expected), magnitude_unit="db", phase_unit="rad"), expected, atol=1e-15)

    def test_missing_ambiguous_or_nonfinite_input_is_rejected(self):
        cases = [
            {},
            {"real": [0.2]},
            {"magnitude": [0.2]},
            {"real": [0.2], "imag": [0.1], "magnitude": [0.3], "phase": [0.0]},
            {"real": [0.2, 0.3], "imag": [0.1]},
            {"real": [float("nan")], "imag": [0.1]},
            {"magnitude": [-0.2], "phase": [0.0]},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    complex_s11(**arguments)


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="filtenna_loader_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, contents):
        path = self.root / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_csv_infers_ri_unit_and_selects_named_columns(self):
        path = self.write("hfss.csv", "label,S11_imag,Frequency [GHz],S11_real\nfirst,-0.2,3.4,0.1\nsecond,0.3,3.5,-0.4\n")
        data = load_s11(path, frequency_column="Frequency [GHz]", first_column="S11_real", second_column="S11_imag", z0_ohm=75)
        np.testing.assert_allclose(data.frequency_hz, [3.4e9, 3.5e9])
        np.testing.assert_allclose(data.s11, [0.1 - 0.2j, -0.4 + 0.3j])
        self.assertEqual(data.z0_ohm, 75.0)
        self.assertEqual(data.source, str(path))

    def test_csv_magnitude_db_and_radian_phase(self):
        cases = [
            ("ma.csv", "frequency_MHz;S11_magnitude;S11_phase\n3400;0.5;90\n3500;0.25;-180\n", {}, [0.5j, -0.25]),
            ("db.tsv", "frequency_kHz\tS11_dB\tS11_angle\n3400000\t-20\t90\n3500000\t-6.020599913279624\t180\n", {}, [0.1j, -0.5]),
            ("radian.txt", "3.4 0.5 1.5707963267948966\n3.5 0.25 -3.141592653589793\n", {"representation": "ma", "frequency_unit": "GHz", "phase_unit": "rad"}, [0.5j, -0.25]),
        ]
        for name, contents, options, expected in cases:
            with self.subTest(name=name):
                data = load_s11(self.write(name, contents), **options)
                np.testing.assert_allclose(data.frequency_hz, [3.4e9, 3.5e9])
                np.testing.assert_allclose(data.s11, expected, atol=1e-15)

    def test_touchstone_wrapped_two_port_uses_s11_and_reference(self):
        # Touchstone orders S11,S21,S12,S22; the other pairs deliberately differ.
        path = self.write("measurement.s2p", "! wrapped two-port RI export\n# MHz S RI R 75\n3400 0.1 -0.2 0.9 0.8\n 0.7 0.6 0.5 0.4 ! S12 and S22\n3500 -4.0D-1 3.0D-1\n 0.2 0.1 -0.6 -0.5 -0.8 -0.9\n")
        data = load_s11(path)
        np.testing.assert_allclose(data.frequency_hz, [3.4e9, 3.5e9])
        np.testing.assert_allclose(data.s11, [0.1 - 0.2j, -0.4 + 0.3j])
        self.assertEqual(data.z0_ohm, 75.0)
        with self.assertRaisesRegex(ValueError, "reference|renormalize"):
            load_s11(path, z0_ohm=50.0)
        np.testing.assert_allclose(load_s11(path, z0_ohm=75.0).s11, data.s11)

    def test_touchstone_db_and_default_ma_options(self):
        db_path = self.write("db.s1p", "# Hz S DB R 50\n3.4e9 -20 90\n3.5e9 -6.020599913279624 -180\n")
        ma_path = self.write("default.s1p", "3.4 0.1 90\n3.5 0.5 -180\n")
        for path in [db_path, ma_path]:
            with self.subTest(path=path.name):
                data = load_s11(path)
                np.testing.assert_allclose(data.frequency_hz, [3.4e9, 3.5e9])
                np.testing.assert_allclose(data.s11, [0.1j, -0.5], atol=1e-15)

    def test_ambiguous_or_malformed_csv_is_rejected(self):
        cases = [
            ("empty.csv", "", {}),
            ("unit.csv", "frequency,S11_real,S11_imag\n3.4,0.1,0.2\n", {}),
            ("representation.csv", "frequency_GHz,first,second\n3.4,0.1,0.2\n", {}),
            ("missing.csv", "frequency_GHz,S11_real,S11_imag\n3.4,0.1\n", {}),
            ("bad-number.csv", "frequency_GHz,S11_real,S11_imag\n3.4,invalid,0.2\n", {}),
            ("short-header.csv", "frequency_GHz,S11_real\n3.4,0.1,0.2\n", {"representation": "ri"}),
            ("negative-frequency.csv", "frequency_GHz,S11_real,S11_imag\n-3.4,0.1,0.2\n", {}),
            ("nonfinite.csv", "frequency_GHz,S11_real,S11_imag\n3.4,NaN,0.2\n", {}),
            ("columns.csv", "frequency_GHz,S11_real,S11_imag\n3.4,0.1,0.2\n", {"second_column": 1}),
        ]
        for name, contents, options in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    load_s11(self.write(name, contents), **options)

    def test_unsupported_or_malformed_touchstone_is_rejected(self):
        contents = [
            "[Version] 2.0\n# GHz S RI R 50\n3.4 0.1 0.2\n",
            "# GHz Y RI R 50\n3.4 0.1 0.2\n",
            "# GHz S RI R 50\n3.4 0.1\n",
            "# GHz S RI R 50\n3.4 0.1 0.2\n# GHz S RI R 50\n",
            "# GHz S RI R -50\n3.4 0.1 0.2\n",
        ]
        for index, content in enumerate(contents):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    load_s11(self.write(f"invalid_{index}.s1p", content))


class SerializationAndCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="filtenna_output_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_result_json_and_both_complex_csv_columns_round_trip(self):
        measured = np.array([0.0j, 0.1 - 0.2j, -0.2 + 0.3j])
        predicted = np.array([0.0j, 0.12 - 0.19j, -0.21 + 0.32j])
        parameters = mode_parameters("II")
        parameters["M_ps"] = parameters["k_ps"] * math.sqrt(parameters["L_p"] * parameters["L_s"])
        result = FitResult(antenna="II", parameters=parameters, frequency_hz=np.array([3.3e9, 3.4e9, 3.5e9]), s11_measured=measured, s11_fitted=predicted, success=True, message="Synthetic serialization check", diagnostics={"complex_rmse": float(np.sqrt(np.mean(abs(predicted - measured) ** 2)))}, model={"R0": 50.0, "Rr": 60.0})
        json_path, csv_path = result.save(self.root / "nested" / "fit")
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, result.to_dict())
        self.assertEqual(saved["parameters"], parameters)
        self.assertEqual(saved["units"]["L"], "H")
        np.testing.assert_allclose(load_s11(csv_path).s11, measured, atol=1e-15)
        fitted = load_s11(csv_path, first_column="fit_real", second_column="fit_imag")
        np.testing.assert_allclose(fitted.frequency_hz, result.frequency_hz)
        np.testing.assert_allclose(fitted.s11, predicted, atol=1e-15)
        numeric_table = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        self.assertTrue(np.all(np.isfinite(numeric_table)))

    def test_cli_fits_csv_and_accepts_result_json_as_initial(self):
        frequency = np.linspace(3.0e9, 3.9e9, 41)
        fitter = FiltennaRLCFitter("I", R0=50.0, Rr=65.0, n_f=1.3, R_f=0.5, L_f=0.1e-9)
        truth = mode_parameters("I")
        measured = fitter.predict_s11(frequency, truth)
        data_path = self.root / "input.csv"
        np.savetxt(data_path, np.column_stack([frequency / 1e9, measured.real, measured.imag]), delimiter=",", header="frequency_GHz,S11_real,S11_imag", comments="")
        initial = {key: value * (1.03 if key.startswith("R") else 1.002) for key, value in truth.items()}
        initial_path = self.root / "initial.json"
        initial_path.write_text(json.dumps({"parameters": initial}), encoding="utf-8")
        output = self.root / "cli_results"
        command = [sys.executable, str(Path(__file__).with_name("rlc_circuit_fitter.py")), "--input", str(data_path), "--antenna", "I", "--rr", "65", "--nf", "1.3", "--rf", "0.5", "--lf-nh", "0.1", "--initial", str(initial_path), "--starts", "1", "--max-nfev", "300", "--output-dir", str(output)]
        process = subprocess.run(command, text=True, capture_output=True, timeout=60, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        saved = json.loads((output / "ant_i_fit.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["success"])
        self.assertLess(saved["diagnostics"]["complex_rmse"], 1e-7)
        reconstructed = load_s11(output / "ant_i_fit.csv", first_column="fit_real", second_column="fit_imag")
        np.testing.assert_allclose(reconstructed.s11, measured, atol=2e-7)
        for key, expected in truth.items():
            self.assertAlmostEqual(saved["parameters"][key] / expected, 1.0, delta=1e-4)

    def test_cli_invalid_initial_json_reports_a_controlled_error(self):
        data_path = self.root / "input.csv"
        data_path.write_text("frequency_GHz,S11_real,S11_imag\n3.4,0.1,0.2\n3.5,0.2,0.1\n", encoding="utf-8")
        initial_path = self.root / "initial.json"
        for invalid in [[], {"parameters": []}]:
            with self.subTest(invalid=invalid):
                initial_path.write_text(json.dumps(invalid), encoding="utf-8")
                standard_error = io.StringIO()
                with contextlib.redirect_stderr(standard_error), contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--input", str(data_path), "--antenna", "I", "--initial", str(initial_path), "--output-dir", str(self.root / "invalid")])
                self.assertEqual(code, 1)
                self.assertIn("Error:", standard_error.getvalue())
                self.assertFalse((self.root / "invalid").exists())


class FittingTests(unittest.TestCase):
    def test_noiseless_recovery_with_perturbed_initial_parameters(self):
        frequency = np.linspace(2.7e9, 4.6e9, 221)
        for antenna in ["I", "II", "III"]:
            with self.subTest(antenna=antenna):
                fitter = FiltennaRLCFitter(antenna, Rr=65.0, n_f=1.3, R_f=0.5, L_f=0.1e-9)
                truth = mode_parameters(antenna)
                measured = fitter.predict_s11(frequency, truth)
                initial = {}
                for key, value in truth.items():
                    factor = 1.08 if key.startswith("R_") else (1.015 if key.startswith("L_") else (0.99 if key.startswith("C_") else 0.92))
                    initial[key] = value * factor
                # A general three-mode driving-point response has at most 11
                # independent rational coefficients for 12 circuit unknowns.
                # One calibrated parameter is necessary for parameter recovery.
                fixed = {"L_t": truth["L_t"]} if antenna == "III" else None
                if fixed:
                    initial.update(fixed)
                result = fitter.fit(frequency, measured, initial=initial, fixed_parameters=fixed, n_starts=1, max_nfev=3000, seed=17)
                self.assertTrue(result.success, str(result.diagnostics))
                self.assertLess(np.sqrt(np.mean(np.abs(result.s11_fitted - measured) ** 2)), 2e-7)
                for key, expected in truth.items():
                    with self.subTest(parameter=key):
                        self.assertAlmostEqual(result.parameters[key] / expected, 1.0, delta=0.02)
                self.assertIsInstance(result.to_dict(), dict)

    def test_three_mode_fit_without_initial_parameters(self):
        frequency = np.linspace(2.7e9, 4.6e9, 161)
        fitter = FiltennaRLCFitter("III", Rr=65.0, n_f=1.3, R_f=0.5, L_f=0.1e-9)
        measured = fitter.predict_s11(frequency, mode_parameters("III"))
        result = fitter.fit(frequency, measured, n_starts=12, max_nfev=3000, seed=42)
        self.assertTrue(result.success, str(result.diagnostics))
        self.assertLess(np.sqrt(np.mean(np.abs(result.s11_fitted - measured) ** 2)), 1e-4)

    def test_invalid_measurements_are_rejected(self):
        fitter = FiltennaRLCFitter("I")
        for frequency, measured in [([3e9, 3.1e9], [0.2j]), ([3e9, 3.1e9], [0.2j, complex(float("nan"), 0)]), ([], [])]:
            with self.subTest(frequency=frequency, measured=measured):
                with self.assertRaises(ValueError):
                    fitter.fit(frequency, measured)


if __name__ == "__main__":
    unittest.main()
