#!/usr/bin/env python3
"""Build Ant. III in HFSS Driven Modal and save Filtenna_5G.aedt.

Requirements: an installed/licensed Ansys Electronics Desktop and PyAEDT.
Install the Python package with: python -m pip install pyaedt
Run: python hfss_pyaedt_modeler.py
Optional: --version 2025.2, --non-graphical, --output PATH, --overwrite.
Offline geometry check: python hfss_pyaedt_modeler.py --check-only

Geometry conventions (all geometry is parameterized in HFSS):
* The substrate occupies z=0..h, centered at x=y=0.
* L_slot is the TOTAL U centerline length: 2*L_slot_arm + W_slot.
  W_slot is the distance between arm centerlines; d_slot is cut width.
  The centered U opens toward +Y. Each arm is therefore 10.03 mm.
* Copper thickness is 0.035 mm, an additional modeling assumption.
* The idealized SMA coax has a 0.5 mm pin, 5 mm length, 0.3 mm wall.
  Its dielectric radius is calculated from its actual material permittivity
  for nominal 50 ohm TEM impedance. Connector threads are not modeled.
* An internal annular wave port is backed on -Z by a 0.5 mm PEC cap.
  The 30 mm air clearance includes that cap and the top copper thickness.
* Existing AEDT material definitions are retained. If absent, local materials
  are defined explicitly: RO4003C er=3.55, tan(delta)=0.0027; nominal PTFE
  er=2.1, tan(delta)=0.0002. These fallback properties are nondispersive.

This script builds and validates the project; it does not run the solver.
It always releases its own Desktop session, including when construction fails.

API references:
https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.html
https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.wave_port.html
RO4003C fallback design Dk and nominal loss data:
https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4003c-laminates
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path


LOG = logging.getLogger("ant_iii_modeler")
DEFAULT_PROJECT = Path(__file__).resolve().with_name("Filtenna_5G.aedt")
DESIGN_VARIABLES = {
    "f0": "3.5GHz",
    "W_p": "28.394mm",
    "L_p": "22.239mm",
    "W_g": "50mm",
    "h": "1.524mm",
    "L_slot": "32.06mm",
    "W_slot": "12mm",
    "d_slot": "0.5mm",
    "L_strip": "22.63mm",
    "W_strip": "1.5mm",
    "g_strip": "0.5mm",
    "y_feed": "-3.05mm",
}
EXTRA_VARIABLES = {
    "t_cu": "0.035mm",
    "coax_a": "0.5mm",
    "coax_length": "5mm",
    "coax_wall": "0.3mm",
    "cap_thickness": "0.5mm",
    "air_margin": "30mm",
    "L_slot_arm": "(L_slot-W_slot)/2",
}
# eta_0 / (2*pi), in ohms; the lossless TEM coax relation is
# Z0 = COAX_TEM_FACTOR / sqrt(er) * ln(b/a).
COAX_TEM_FACTOR = 59.9584916


def require(result, operation: str):
    """PyAEDT operations can return False instead of raising an exception."""
    if result is None or result is False:
        raise RuntimeError(f"HFSS operation failed: {operation}")
    return result


def check_dimensions(coax_er: float = 2.1) -> dict[str, float]:
    """Check fit/contact analytically, without needing PyAEDT or a license."""
    lengths = {
        key: float(value.removesuffix("mm"))
        for key, value in {**DESIGN_VARIABLES, **EXTRA_VARIABLES}.items()
        if value.endswith("mm")
    }
    if not math.isfinite(coax_er) or coax_er <= 1:
        raise ValueError("The coax dielectric must have a finite scalar er > 1.")
    for name, value in lengths.items():
        if not math.isfinite(value) or (name != "y_feed" and value <= 0):
            raise ValueError(f"Invalid dimension {name}={value} mm.")
    v = lengths
    arm = (v["L_slot"] - v["W_slot"]) / 2
    if arm <= 0 or v["d_slot"] >= min(arm, v["W_slot"]):
        raise ValueError("The U slot requires positive arms wider apart than the cut width.")
    if v["W_slot"] + v["d_slot"] >= v["W_p"] or arm + v["d_slot"] >= v["L_p"]:
        raise ValueError("The U-slot cut must fit strictly inside the patch.")
    if max(v["W_p"] + 2 * (v["g_strip"] + v["W_strip"]),
           v["L_p"], v["L_strip"]) >= v["W_g"]:
        raise ValueError("The patch and both parasitic strips must fit on the substrate.")
    # Feed is on the central tongue above the U base, clear of every cut edge.
    if not (-arm / 2 + v["d_slot"] / 2 + v["coax_a"] < v["y_feed"]
            < v["L_p"] / 2 - v["coax_a"]):
        raise ValueError("The feed pin intersects the U base or the patch edge.")
    if v["coax_a"] >= (v["W_slot"] - v["d_slot"]) / 2:
        raise ValueError("The feed pin intersects a U-slot arm.")
    b = v["coax_a"] * math.exp(50 * math.sqrt(coax_er) / COAX_TEM_FACTOR)
    c = b + v["coax_wall"]
    if c + abs(v["y_feed"]) >= v["W_g"] / 2:
        raise ValueError("The coax shield extends beyond the ground plane.")
    if v["coax_length"] <= v["t_cu"]:
        raise ValueError("The coax must extend below the ground copper.")
    return {
        "slot_arm_mm": arm,
        "coax_dielectric_radius_mm": b,
        "coax_outer_radius_mm": c,
        "coax_impedance_ohm": COAX_TEM_FACTOR / math.sqrt(coax_er) * math.log(b / v["coax_a"]),
    }


def ensure_dielectric(hfss, name: str, er: float, loss_tangent: float):
    """Use the installed material, or create a fully specified local fallback."""
    material = hfss.materials.exists_material(name)
    if not material:
        material = require(hfss.materials.add_material(name), f"create material {name}")
        material.permittivity = er
        material.permeability = 1
        material.conductivity = 0
        material.dielectric_loss_tangent = loss_tangent
        require(material.update(), f"update material {name}")
        LOG.warning("Created local %s: er=%g, tan(delta)=%g.", name, er, loss_tangent)
    return material


def build_geometry(hfss) -> None:
    """Steps 2-8: parameterized antenna, coax excitation, and radiation region."""
    modeler = hfss.modeler
    # These are design variables, so a project-level '$' prefix is unnecessary.
    for name, value in {**DESIGN_VARIABLES, **EXTRA_VARIABLES}.items():
        hfss[name] = value

    substrate_material = ensure_dielectric(hfss, "rogers ro4003c", 3.55, 0.0027)
    coax_material = ensure_dielectric(hfss, "teflon_based", 2.1, 0.0002)
    for name in ("copper", "pec", "air", "vacuum"):
        if not hfss.materials.exists_material(name):
            raise RuntimeError(f"Required AEDT material is unavailable: {name}")
    try:
        coax_er = float(coax_material.permittivity.value)
    except (ValueError, TypeError) as exc:
        raise ValueError("teflon_based must have scalar, nondispersive permittivity.") from exc
    dimensions = check_dimensions(coax_er)
    hfss["coax_er"] = str(coax_er)
    hfss["coax_b"] = f"coax_a*exp(50*sqrt(coax_er)/{COAX_TEM_FACTOR})"
    hfss["coax_c"] = "coax_b+coax_wall"
    LOG.info("U arm %.3f mm; coax dielectric radius %.6f mm; nominal Z0 %.3f ohm.",
             dimensions["slot_arm_mm"], dimensions["coax_dielectric_radius_mm"],
             dimensions["coax_impedance_ohm"])

    def box(name, origin, sizes, material="copper"):
        return require(modeler.create_box(origin=origin, sizes=sizes, name=name,
                                          material=material), f"create {name}")

    def cylinder(name, z, radius, height, material):
        return require(modeler.create_cylinder(
            orientation="Z", origin=["0mm", "y_feed", z], radius=radius,
            height=height, num_sides=0, name=name, material=material,
        ), f"create {name}")

    # 3. Substrate, 50 x 50 x 1.524 mm.
    substrate = box("Substrate", ["-W_g/2", "-W_g/2", "0mm"],
                    ["W_g", "W_g", "h"], substrate_material.name)
    substrate.solve_inside = True

    # 4. Finite-thickness copper avoids relying on sheet material assignments.
    ground = box("Ground", ["-W_g/2", "-W_g/2", "-t_cu"], ["W_g", "W_g", "t_cu"])
    patch = box("Patch", ["-W_p/2", "-L_p/2", "h"], ["W_p", "L_p", "t_cu"])

    # 5. Three overlapping cutting boxes form one connected U-shaped tool.
    slot_left = box("USlotTool", ["-W_slot/2-d_slot/2", "-L_slot_arm/2-d_slot/2", "h"],
                    ["d_slot", "L_slot_arm+d_slot", "t_cu"], "vacuum")
    slot_right = box("USlotRight", ["W_slot/2-d_slot/2", "-L_slot_arm/2-d_slot/2", "h"],
                     ["d_slot", "L_slot_arm+d_slot", "t_cu"], "vacuum")
    slot_base = box("USlotBase", ["-W_slot/2-d_slot/2", "-L_slot_arm/2-d_slot/2", "h"],
                    ["W_slot+d_slot", "d_slot", "t_cu"], "vacuum")
    slot_name = require(modeler.unite([slot_left.name, slot_right.name, slot_base.name]),
                        "unite the U-slot cutting tool")
    require(modeler.subtract(patch.name, slot_name, keep_originals=False), "cut U slot from patch")

    # 6. Edge-to-edge gap g_strip on the left and right of the patch.
    strip_left = box("Parasitic_Left", ["-W_p/2-g_strip-W_strip", "-L_strip/2", "h"],
                     ["W_strip", "L_strip", "t_cu"])
    strip_right = box("Parasitic_Right", ["W_p/2+g_strip", "-L_strip/2", "h"],
                      ["W_strip", "L_strip", "t_cu"])

    # 7. Coax: pin reaches the top copper; PTFE and shield terminate at z=0.
    pin = cylinder("Coax_Pin", "-coax_length", "coax_a", "coax_length+h+t_cu", "copper")
    dielectric = cylinder("Coax_PTFE", "-coax_length", "coax_b", "coax_length", coax_material.name)
    shield = cylinder("Coax_Shield", "-coax_length", "coax_c", "coax_length", "copper")
    # Use the full dielectric cylinder to bore the shield and ground first.
    require(modeler.subtract([shield.name, ground.name], dielectric.name, keep_originals=True),
            "open coax shield and ground aperture")
    require(modeler.subtract([substrate.name, dielectric.name], pin.name, keep_originals=True),
            "remove pin volume from substrate and coax dielectric")
    # Unite overlapping copper volumes to guarantee electrical contact.
    patch_name = require(modeler.unite([patch.name, pin.name]), "join feed pin to patch")
    ground_name = require(modeler.unite([ground.name, shield.name]), "join coax shield to ground")
    dielectric.solve_inside = True
    for name in (patch_name, ground_name, strip_left.name, strip_right.name):
        modeler[name].solve_inside = False
        modeler[name].color = (184, 115, 51)

    # Explicit -Z backing avoids automatic PEC-cap direction heuristics.
    cap = cylinder("Coax_PEC_Back", "-coax_length-cap_thickness", "coax_c", "cap_thickness", "pec")
    cap.solve_inside = False
    port = require(modeler.create_circle(
        orientation="XY", origin=["0mm", "y_feed", "-coax_length"],
        radius="coax_b", name="Coax_Port", material="vacuum",
    ), "create coax port disk")
    port_hole = require(modeler.create_circle(
        orientation="XY", origin=["0mm", "y_feed", "-coax_length"],
        radius="coax_a", name="Port_Pin_Cut", material="vacuum",
    ), "create port center cut")
    require(modeler.subtract(port.name, port_hole.name, keep_originals=False), "make annular port")

    # 8. Subtract all physical volumes; port is a sheet and is retained.
    air = box("AirBox", ["-W_g/2-air_margin", "-W_g/2-air_margin",
                         "-coax_length-cap_thickness-air_margin"],
              ["W_g+2*air_margin", "W_g+2*air_margin",
               "h+t_cu+coax_length+cap_thickness+2*air_margin"], "air")
    require(modeler.subtract(
        air.name,
        [substrate.name, patch_name, ground_name, strip_left.name, strip_right.name,
         dielectric.name, cap.name], keep_originals=True,
    ), "remove occupied volumes from AirBox")
    air = modeler[air.name]
    air.solve_inside = True
    air.transparency = 0.9
    # Cavity faces must never receive Radiation. Test every vertex against the
    # six OUTER planes, so internal faces are excluded after boolean subtraction.
    bounds = [float(value) for value in air.bounding_box]
    radiation_faces = []
    for face in air.faces:
        vertices = [vertex.position for vertex in face.vertices]
        if vertices and any(
            all(abs(float(vertex[axis]) - bounds[axis + side]) < 1e-5 for vertex in vertices)
            for axis in range(3) for side in (0, 3)
        ):
            radiation_faces.append(face.id)
    if len(radiation_faces) != 6:
        raise RuntimeError(f"Expected six exterior AirBox faces, found {len(radiation_faces)}.")
    require(hfss.assign_radiation_boundary_to_faces(radiation_faces, name="Radiation"),
            "assign Radiation to exterior faces")
    require(hfss.wave_port(
        assignment=port.name, create_port_sheet=False, create_pec_cap=False,
        integration_line=[["coax_a", "y_feed", "-coax_length"],
                          ["coax_b", "y_feed", "-coax_length"]],
        modes=1, impedance=50, name="CoaxPort", renormalize=True, deembed=0,
    ), "assign 50 ohm coax wave port")


def create_analysis(hfss) -> None:
    """Step 9: adaptive solution, 401 discrete points, and far-field context."""
    setup = require(hfss.create_setup(name="Setup_3p5GHz", setup_type="HFSSDriven"),
                    "create Driven Modal setup")
    setup.props["Frequency"] = "f0"
    setup.props["MaxDeltaS"] = 0.01
    setup.props["MaximumPasses"] = 20
    setup.props["MinimumPasses"] = 2
    setup.props["MinimumConvergedPasses"] = 2
    require(setup.update(), "update adaptive convergence settings")
    require(hfss.create_linear_count_sweep(
        setup=setup.name, unit="GHz", start_frequency=2.6, stop_frequency=4.4,
        num_of_freq_points=401, name="Sweep_2p6_to_4p4GHz", sweep_type="Discrete",
        save_fields=True, save_rad_fields=True,
    ), "create 401-point discrete frequency sweep")
    # This context enables realized-gain reports after the user solves the design.
    require(hfss.insert_infinite_sphere(
        definition="Theta-Phi", theta_start=0, theta_stop=180, theta_step=2,
        phi_start=-180, phi_stop=180, phi_step=2, name="FarField_3D",
    ), "create far-field sphere")


def build_project(project_path: Path = DEFAULT_PROJECT, *, version: str | None = None,
                  non_graphical: bool = False, overwrite: bool = False) -> Path:
    """Execute the complete sequence in a dedicated Desktop and close it."""
    check_dimensions()
    project_path = Path(project_path).expanduser().resolve()
    if project_path.suffix.lower() != ".aedt":
        raise ValueError("The output filename must end in .aedt.")
    if project_path.exists() and not overwrite:
        raise FileExistsError(f"{project_path} already exists. Use --overwrite to replace it.")
    if Path(str(project_path) + ".lock").exists():
        raise RuntimeError(f"Close the open/locked project before rebuilding: {project_path}")
    # ansys.aedt.core is the supported namespace of the 'pyaedt' distribution.
    try:
        from ansys.aedt.core import Hfss
    except ImportError as exc:
        raise RuntimeError("PyAEDT is unavailable. Install it with: python -m pip install pyaedt") from exc

    project_path.parent.mkdir(parents=True, exist_ok=True)
    hfss = None
    try:
        # 1. 'Modal' is the current PyAEDT key for HFSS Driven Modal.
        # Start an unnamed, fresh project to avoid opening/modifying an old file.
        hfss = Hfss(project=None, design="Ant_III", solution_type="Modal", version=version,
                    non_graphical=non_graphical, new_desktop=True, close_on_exit=False)
        hfss.modeler.model_units = "mm"
        build_geometry(hfss)
        create_analysis(hfss)
        hfss.modeler.fit_all()

        # 10. Save even if the subsequent HFSS validation reports a problem,
        # so the constructed design can be inspected rather than discarded.
        require(hfss.save_project(file_name=str(project_path), overwrite=overwrite), "save project")
        if not project_path.is_file():
            raise RuntimeError(f"HFSS did not create the expected project: {project_path}")
        validation_log = project_path.with_name(project_path.stem + "_validation.log")
        if hfss.validate_simple(log_file=str(validation_log)) != 1:
            raise RuntimeError(f"Project saved, but HFSS validation failed. See {validation_log}")
        LOG.info("Saved and validated: %s", project_path)
        LOG.info("Setup prepared; no solver run was requested or performed.")
        return project_path
    finally:
        if hfss is not None:
            # Only the new Desktop owned by this script is closed.
            released = hfss.release_desktop(close_projects=True, close_desktop=True)
            if released is False:
                LOG.warning("Desktop release failed; close the script's AEDT session manually.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROJECT, help="Destination .aedt file")
    parser.add_argument("--version", default=None, help="Installed AEDT version, for example 2025.2")
    parser.add_argument("--non-graphical", action="store_true", help="Start AEDT without a GUI")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing unlocked output project")
    parser.add_argument("--check-only", action="store_true", help="Check dimensions without starting AEDT")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.check_only:
            for name, value in check_dimensions().items():
                print(f"{name}: {value:.6f}")
            print("Geometry checks passed. HFSS has not been started or validated.")
        else:
            build_project(args.output, version=args.version,
                          non_graphical=args.non_graphical, overwrite=args.overwrite)
    except (OSError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
