"""Convert SMIRNOFF non-bonded parameters into tensors."""

import collections
import copy
import typing

import openff.interchange.components.potentials
import openff.interchange.models
import openff.toolkit
import openff.units
import torch

import smee
import smee.utils

if typing.TYPE_CHECKING:
    import smirnoff_plugins.collections.nonbonded

_UNITLESS = openff.units.unit.dimensionless
_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians
_KCAL_PER_MOL = openff.units.unit.kilocalories / openff.units.unit.mole
_ELEMENTARY_CHARGE = openff.units.unit.elementary_charge


def convert_nonbonded_handlers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    handler_type: str,
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
    parameter_cols: tuple[str, ...],
    attribute_cols: tuple[str, ...] | None = None,
    has_exclusions: bool = True,
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    """Convert a list of SMIRNOFF non-bonded handlers into a tensor potential and
    associated parameter maps.

    Notes:
        This function assumes that all parameters come from the same force field

    Args:
        handlers: The list of SMIRNOFF non-bonded handlers to convert.
        handler_type: The type of non-bonded handler being converted.
        topologies: The topologies associated with each handler.
        v_site_maps: The virtual site maps associated with each handler.
        parameter_cols: The ordering of the parameter array columns.
        attribute_cols: The handler attributes to include in the potential *in addition*
            to the intra-molecular scaling factors.
        has_exclusions: Whether the handlers are excepted to define exclusions.

    Returns:
        The potential containing tensors of the parameter values, and a list of
        parameter maps which map the parameters to the interactions they apply to.
    """
    attribute_cols = attribute_cols if attribute_cols is not None else []

    assert len(topologies) == len(handlers), "topologies and handlers must match"
    assert len(v_site_maps) == len(handlers), "v-site maps and handlers must match"

    if has_exclusions:
        attribute_cols = (
            "scale_12",
            "scale_13",
            "scale_14",
            "scale_15",
            *attribute_cols,
        )

    potential = smee.converters.openff._openff._handlers_to_potential(
        handlers,
        handler_type,
        parameter_cols,
        attribute_cols,
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }
    attribute_to_idx = {column: i for i, column in enumerate(potential.attribute_cols)}

    parameter_maps = []

    for handler, topology, v_site_map in zip(
        handlers, topologies, v_site_maps, strict=True
    ):
        assignment_map = collections.defaultdict(lambda: collections.defaultdict(float))

        n_particles = topology.n_atoms + (
            0 if v_site_map is None else len(v_site_map.keys)
        )

        for topology_key, parameter_key in handler.key_map.items():
            if isinstance(topology_key, openff.interchange.models.VirtualSiteKey):
                continue

            atom_idx = topology_key.atom_indices[0]
            assignment_map[atom_idx][parameter_key_to_idx[parameter_key]] += 1.0

        for topology_key, parameter_key in handler.key_map.items():
            if not isinstance(topology_key, openff.interchange.models.VirtualSiteKey):
                continue

            v_site_idx = v_site_map.key_to_idx[topology_key]

            if parameter_key.associated_handler != "Electrostatics":
                assignment_map[v_site_idx][parameter_key_to_idx[parameter_key]] += 1.0
            else:
                for i, atom_idx in enumerate(topology_key.orientation_atom_indices):
                    mult_key = copy.deepcopy(parameter_key)
                    mult_key.mult = i

                    assignment_map[atom_idx][parameter_key_to_idx[mult_key]] += 1.0
                    assignment_map[v_site_idx][parameter_key_to_idx[mult_key]] += -1.0

        assignment_matrix = torch.zeros(
            (n_particles, len(potential.parameters)), dtype=torch.float64
        )

        for particle_idx in assignment_map:
            for parameter_idx, count in assignment_map[particle_idx].items():
                assignment_matrix[particle_idx, parameter_idx] = count

        if has_exclusions:
            exclusion_to_scale = smee.utils.find_exclusions(topology, v_site_map)
            exclusions = torch.tensor([*exclusion_to_scale])
            exclusion_scale_idxs = torch.tensor(
                [[attribute_to_idx[scale]] for scale in exclusion_to_scale.values()],
                dtype=torch.int64,
            )
        else:
            exclusions = torch.zeros((0, 2), dtype=torch.int64)
            exclusion_scale_idxs = torch.zeros((0, 1), dtype=torch.int64)

        parameter_map = smee.NonbondedParameterMap(
            assignment_matrix=assignment_matrix.to_sparse(),
            exclusions=exclusions,
            exclusion_scale_idxs=exclusion_scale_idxs,
        )
        parameter_maps.append(parameter_map)

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "vdW",
    {
        "epsilon": _KCAL_PER_MOL,
        "sigma": _ANGSTROM,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_vdw(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFvdWCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    mixing_rules = {handler.mixing_rule for handler in handlers}
    assert len(mixing_rules) == 1, "multiple mixing rules found"
    mixing_rule = next(iter(mixing_rules))

    if mixing_rule != "lorentz-berthelot":
        raise NotImplementedError("only Lorentz-Berthelot mixing rules are supported.")

    return convert_nonbonded_handlers(
        handlers,
        smee.PotentialType.VDW,
        topologies,
        v_site_maps,
        ("epsilon", "sigma"),
        ("cutoff", "switch_width"),
    )


@smee.converters.smirnoff_parameter_converter(
    "DoubleExponential",
    {
        "epsilon": _KCAL_PER_MOL,
        "r_min": _ANGSTROM,
        "alpha": _UNITLESS,
        "beta": _UNITLESS,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_dexp(
    handlers: list[
        "smirnoff_plugins.collections.nonbonded.SMIRNOFFDoubleExponentialCollection"
    ],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    import smee.potentials.nonbonded

    (
        potential,
        parameter_maps,
    ) = smee.converters.openff.nonbonded.convert_nonbonded_handlers(
        handlers,
        "DoubleExponential",
        topologies,
        v_site_maps,
        ("epsilon", "r_min"),
        ("cutoff", "switch_width", "alpha", "beta"),
    )
    potential.type = smee.PotentialType.VDW
    potential.fn = smee.EnergyFn.VDW_DEXP

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "DampedExp6810",
    {
        "rho": _ANGSTROM,
        "beta": _ANGSTROM**-1,
        "c6": _KCAL_PER_MOL * _ANGSTROM**6,
        "c8": _KCAL_PER_MOL * _ANGSTROM**8,
        "c10": _KCAL_PER_MOL * _ANGSTROM**10,
        "force_at_zero": _KCAL_PER_MOL * _ANGSTROM**-1,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
        "switch_width": _ANGSTROM,
    },
)
def convert_dampedexp6810(
    handlers: list[
        "smirnoff_plugins.collections.nonbonded.SMIRNOFFDampedExp6810Collection"
    ],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    import smee.potentials.nonbonded

    (
        potential,
        parameter_maps,
    ) = smee.converters.openff.nonbonded.convert_nonbonded_handlers(
        handlers,
        "DampedExp6810",
        topologies,
        v_site_maps,
        ("rho", "beta", "c6", "c8", "c10"),
        ("cutoff", "switch_width", "force_at_zero"),
    )
    potential.type = smee.PotentialType.VDW
    potential.fn = smee.EnergyFn.VDW_DAMPEDEXP6810

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "Multipole",
    {
        # Molecular multipole moments
        "dipoleX": _ELEMENTARY_CHARGE * _ANGSTROM,
        "dipoleY": _ELEMENTARY_CHARGE * _ANGSTROM,
        "dipoleZ": _ELEMENTARY_CHARGE * _ANGSTROM,
        "quadrupoleXX": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleXY": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleXZ": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleYX": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleYY": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleYZ": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleZX": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleZY": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        "quadrupoleZZ": _ELEMENTARY_CHARGE * _ANGSTROM**2,
        # Local frame definition
        "axisType": _UNITLESS,
        "multipoleAtomZ": _UNITLESS,
        "multipoleAtomX": _UNITLESS,
        "multipoleAtomY": _UNITLESS,
        # Damping and polarizability (these may not be present in current force fields)
        "thole": _UNITLESS,
        "dampingFactor": _ANGSTROM,
        "polarity": _ANGSTROM**3,
        # Cutoff and scaling
        "cutoff": _ANGSTROM,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
    },
    depends_on=["Electrostatics"],
)
def convert_multipole(
    handlers: list[
        "smirnoff_plugins.collections.nonbonded.SMIRNOFFMultipoleCollection"
    ],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
    dependencies: dict[
        str, tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]
    ],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:

    potential_chg, parameter_maps_chg = dependencies["Electrostatics"]

    (
        potential_pol,
        parameter_maps_pol,
    ) = smee.converters.openff.nonbonded.convert_nonbonded_handlers(
        handlers,
        "Multipole",
        topologies,
        v_site_maps,
        (
            "dipoleX", "dipoleY", "dipoleZ",
            "quadrupoleXX", "quadrupoleXY", "quadrupoleXZ",
            "quadrupoleYX", "quadrupoleYY", "quadrupoleYZ", 
            "quadrupoleZX", "quadrupoleZY", "quadrupoleZZ",
            "axisType", "multipoleAtomZ", "multipoleAtomX", "multipoleAtomY",
            "thole", "dampingFactor", "polarity"
        ),
        ("cutoff",),
        has_exclusions=False,
    )

    cutoff_idx_pol = potential_pol.attribute_cols.index("cutoff")
    cutoff_idx_chg = potential_chg.attribute_cols.index("cutoff")

    assert torch.isclose(
        potential_pol.attributes[cutoff_idx_pol],
        potential_chg.attributes[cutoff_idx_chg],
    )

    potential_chg.fn = smee.EnergyFn.POLARIZATION

    potential_chg.parameter_cols = (
        *potential_chg.parameter_cols,
        *potential_pol.parameter_cols,
    )
    potential_chg.parameter_units = (
        *potential_chg.parameter_units,
        *potential_pol.parameter_units,
    )
    potential_chg.parameter_keys = [
        *potential_chg.parameter_keys,
        *potential_pol.parameter_keys,
    ]

    # Handle different numbers of columns between charge and polarizability potentials
    n_chg_cols = potential_chg.parameters.shape[1]
    n_pol_cols = potential_pol.parameters.shape[1]
    
    # Pad charge parameters with zeros for the new polarizability columns
    parameters_chg = torch.cat(
        (potential_chg.parameters, torch.zeros(potential_chg.parameters.shape[0], n_pol_cols, dtype=potential_chg.parameters.dtype)), dim=1
    )
    parameters_pol = potential_pol.parameters
    parameters_pol[:, 17] = parameters_pol[:, 18]**(1/6)
    # Pad polarizability parameters with zeros for the charge columns
    parameters_pol = torch.cat(
        (torch.zeros(potential_pol.parameters.shape[0], n_chg_cols, dtype=potential_pol.parameters.dtype), potential_pol.parameters), dim=1
    )
    potential_chg.parameters = torch.cat((parameters_chg, parameters_pol), dim=0)

    for parameter_map_chg, parameter_map_pol in zip(
        parameter_maps_chg, parameter_maps_pol, strict=True
    ):
        parameter_map_chg.assignment_matrix = torch.block_diag(
            parameter_map_chg.assignment_matrix.to_dense(),
            parameter_map_pol.assignment_matrix.to_dense(),
        ).to_sparse()

    return potential_chg, parameter_maps_chg


def _make_v_site_electrostatics_compatible(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFElectrostaticsCollection],
):
    """Attempts to make electrostatic potentials associated with virtual sites more
    consistent with other parameters so that they can be more easily converted to
    tensors.

    Args:
        handlers: The list of SMIRNOFF electrostatic handlers to make compatible.
    """
    for handler_idx, handler in enumerate(handlers):
        if not any(
            key.associated_handler == "Electrostatics" for key in handler.potentials
        ):
            continue

        handler = copy.deepcopy(handler)
        potentials = {}

        for key, potential in handler.potentials.items():
            # for some reason interchange lists this as electrostatics and not v-sites
            if key.associated_handler != "Electrostatics":
                potentials[key] = potential
                continue

            assert key.mult is None

            for i in range(len(potential.parameters["charge_increments"])):
                mult_key = copy.deepcopy(key)
                mult_key.mult = i

                mult_potential = copy.deepcopy(potential)
                mult_potential.parameters = {
                    "charge": potential.parameters["charge_increments"][i]
                }
                assert mult_key not in potentials
                potentials[mult_key] = mult_potential

        handler.potentials = potentials
        handlers[handler_idx] = handler


def _convert_charge_increment_handlers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    assert len(topologies) == len(handlers), "topologies and handlers must match"
    assert len(v_site_maps) == len(handlers), "v-site maps and handlers must match"

    handlers_base, handlers_bci = [], []

    for handler in handlers:
        handler_base = copy.deepcopy(handler)
        handler_bci = copy.deepcopy(handler)

        for key in handler.potentials:
            if key.associated_handler == "ChargeIncrementModel":
                handler_base.potentials.pop(key)
                if key.mult != 0:
                    handler_bci.potentials.pop(key)
            else:
                handler_bci.potentials.pop(key)

        for key, val in handler.key_map.items():
            if val.associated_handler == "ChargeIncrementModel":
                handler_base.key_map.pop(key)
                if val.mult != 0:
                    handler_bci.key_map.pop(key)
            else:
                handler_bci.key_map.pop(key)

        handlers_base.append(handler_base)
        handlers_bci.append(handler_bci)

    potential, parameter_maps = convert_nonbonded_handlers(
        handlers_base,
        "Electrostatics",
        topologies,
        v_site_maps,
        ("charge",),
        ("cutoff",),
    )

    for key in potential.parameter_keys:
        key.associated_handler = "ChargeModel"

    potential_bci = smee.converters.openff._openff._handlers_to_potential(
        handlers_bci, "Electrostatics", ("charge_increment",), ()
    )

    if potential.exceptions is not None and potential_bci.exceptions is not None:
        raise NotImplementedError("exceptions are not supported for charge increments")

    potential.parameter_keys.extend(potential_bci.parameter_keys)
    potential.parameters = torch.vstack(
        [potential.parameters, potential_bci.parameters]
    )

    parameter_key_to_idx = {
        parameter_key: i for i, parameter_key in enumerate(potential.parameter_keys)
    }

    for handler_bci, parameter_map in zip(handlers_bci, parameter_maps, strict=True):
        assignment_matrix = torch.zeros(
            (len(parameter_map.assignment_matrix), len(potential.parameters)),
            dtype=torch.float64,
        )

        n_charges = parameter_map.assignment_matrix.shape[1]
        assignment_matrix[:, :n_charges] += parameter_map.assignment_matrix.to_dense()

        for topology_key, parameter_key in handler_bci.key_map.items():
            if not isinstance(
                topology_key, openff.interchange.models.ChargeIncrementTopologyKey
            ):
                raise NotImplementedError("only charge increment keys are supported")
            if len(topology_key.other_atom_indices) != 1:
                raise NotImplementedError("only 1-1 charge increments are supported")

            atom_idx_0 = topology_key.this_atom_index
            atom_idx_1 = topology_key.other_atom_indices[0]

            param_idx = parameter_key_to_idx[parameter_key]

            assignment_matrix[atom_idx_0, param_idx] += 1.0
            assignment_matrix[atom_idx_1, param_idx] += -1.0

        parameter_map.assignment_matrix = assignment_matrix.to_sparse()

    return potential, parameter_maps


@smee.converters.smirnoff_parameter_converter(
    "Electrostatics",
    {
        "charge": _ELEMENTARY_CHARGE,
        "charge_increment": _ELEMENTARY_CHARGE,
        "scale_12": _UNITLESS,
        "scale_13": _UNITLESS,
        "scale_14": _UNITLESS,
        "scale_15": _UNITLESS,
        "cutoff": _ANGSTROM,
    },
)
def convert_electrostatics(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFElectrostaticsCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None],
) -> tuple[smee.TensorPotential, list[smee.NonbondedParameterMap]]:
    handlers = [*handlers]
    _make_v_site_electrostatics_compatible(handlers)

    has_charge_increment = any(
        key.associated_handler == "ChargeIncrementModel"
        for handler in handlers
        for key in handler.potentials
    )

    if has_charge_increment:
        return _convert_charge_increment_handlers(handlers, topologies, v_site_maps)

    return convert_nonbonded_handlers(
        handlers, "Electrostatics", topologies, v_site_maps, ("charge",), ("cutoff",)
    )
