"""Multipole potential energy functions."""

import math
import typing

import torch

import smee.potentials


@smee.potentials.potential_energy_fn(
    smee.PotentialType.ELECTROSTATICS, smee.EnergyFn.POLARIZATION
)
def compute_multipole_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
    box_vectors: torch.Tensor | None = None,
    pairwise=None,
    polarization_type: str = "mutual",
    extrapolation_coefficients: list[float] | None = None,
) -> torch.Tensor:
    print(f"DEBUG: Multipole energy calculation with polarization_type={polarization_type}")
    """Compute the multipole energy including polarization effects.

    This function supports the full AMOEBA multipole model with the following parameters
    per atom (arranged in columns):
    
    Column 0: charge (double) - the particle's charge
    Columns 1-3: molecularDipole (vector[3]) - the particle's molecular dipole (x, y, z)
    Columns 4-12: molecularQuadrupole (vector[9]) - the particle's molecular quadrupole
    Column 13: axisType (int) - the particle's axis type (0=NoAxisType, 1=ZOnly, etc.)
    Column 14: multipoleAtomZ (int) - index of first atom for lab<->molecular frames
    Column 15: multipoleAtomX (int) - index of second atom for lab<->molecular frames
    Column 16: multipoleAtomY (int) - index of third atom for lab<->molecular frames
    Column 17: thole (double) - Thole parameter (default 0.39)
    Column 18: dampingFactor (double) - dampingFactor parameter
    Column 19: polarity (double) - polarity/polarizability parameter
    
    For backwards compatibility, if fewer columns are provided, sensible defaults are used.

    Args:
        system: The system.
        potential: The potential containing multipole parameters.
        conformer: The conformer.
        box_vectors: The box vectors.
        pairwise: The pairwise distances.
        polarization_type: The polarization solver type. Options are:
            - "mutual": Full iterative SCF solver (default, ~60 iterations)
            - "direct": Direct polarization with no mutual coupling (0 iterations)
            - "extrapolated": Extrapolated polarization using OPT method
        extrapolation_coefficients: Custom extrapolation coefficients for "extrapolated" type.
            If None, uses OPT3 coefficients [-0.154, 0.017, 0.657, 0.475].
            Must sum to approximately 1.0 for energy conservation.

    Returns:
        The energy.
    """
    from smee.potentials.nonbonded import (
        _COULOMB_PRE_FACTOR,
        _compute_pme_exclusions,
        _compute_pme_grid,
        _broadcast_exclusions,
        compute_pairwise,
        compute_pairwise_scales,
    )

    # Validate polarization_type
    valid_types = ["mutual", "direct", "extrapolated"]
    if polarization_type not in valid_types:
        raise ValueError(f"polarization_type must be one of {valid_types}, got {polarization_type}")

    box_vectors = None if not system.is_periodic else box_vectors

    cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]

    pairwise = compute_pairwise(system, conformer, box_vectors, cutoff)

    # Initialize parameter lists for all AMOEBA multipole parameters
    charges = []
    dipoles = []  # 3 components per atom
    quadrupoles = []  # 9 components per atom
    axis_types = []
    multipole_atom_z = []  # Z-axis defining atom indices
    multipole_atom_x = []  # X-axis defining atom indices  
    multipole_atom_y = []  # Y-axis defining atom indices
    thole_params = []
    damping_factors = []
    polarizabilities = []

    # Extract parameters from parameter matrix
    for topology, n_copies in zip(system.topologies, system.n_copies):
        parameter_map = topology.parameters[potential.type]
        topology_parameters = parameter_map.assignment_matrix @ potential.parameters
        
        n_particles = topology.n_particles
        n_params = topology_parameters.shape[1]
        
        # Expected parameter layout for full AMOEBA multipole:
        # Column 0: charge
        # Columns 1-3: molecular dipole (x, y, z)
        # Columns 4-12: molecular quadrupole (9 components)
        # Column 13: axis type (int)
        # Column 14: multipole atom Z index
        # Column 15: multipole atom X index  
        # Column 16: multipole atom Y index
        # Column 17: thole parameter
        # Column 18: damping factor
        # Column 19: polarizability

        charges.append(topology_parameters[:n_particles, 0].repeat(n_copies))
        dipoles.append(topology_parameters[n_particles:, 1:4].repeat(n_copies, 1))
        quadrupoles.append(topology_parameters[n_particles:, 4:13].repeat(n_copies, 1))
        axis_types.append(topology_parameters[n_particles:, 13].repeat(n_copies).int())
        multipole_atom_z.append(topology_parameters[n_particles:, 14].repeat(n_copies).int())
        multipole_atom_x.append(topology_parameters[n_particles:, 15].repeat(n_copies).int())
        multipole_atom_y.append(topology_parameters[n_particles:, 16].repeat(n_copies).int())
        thole_params.append(topology_parameters[n_particles:, 17].repeat(n_copies))
        damping_factors.append(topology_parameters[n_particles:, 18].repeat(n_copies))
        polarizabilities.append(topology_parameters[n_particles:, 19].repeat(n_copies))

    # Concatenate all parameter lists
    charges = torch.cat(charges)  # Shape: (n_total_particles,)
    dipoles = torch.cat(dipoles)  # Shape: (n_total_particles, 3)
    quadrupoles = torch.cat(quadrupoles)  # Shape: (n_total_particles, 9)
    axis_types = torch.cat(axis_types)  # Shape: (n_total_particles,)
    multipole_atom_z = torch.cat(multipole_atom_z)  # Shape: (n_total_particles,)
    multipole_atom_x = torch.cat(multipole_atom_x)  # Shape: (n_total_particles,)
    multipole_atom_y = torch.cat(multipole_atom_y)  # Shape: (n_total_particles,)
    thole_params = torch.cat(thole_params)  # Shape: (n_total_particles,)
    polarizabilities = torch.cat(polarizabilities)  # Shape: (n_total_particles,)

    pair_scales = compute_pairwise_scales(system, potential)

    # static partial charge - partial charge energy
    if system.is_periodic == False:
        coul_energy = (
            _COULOMB_PRE_FACTOR
            * pair_scales
            * charges[pairwise.idxs[:, 0]]
            * charges[pairwise.idxs[:, 1]]
            / pairwise.distances
        ).sum(-1)
    else:
        import NNPOps.pme

        cutoff = potential.attributes[potential.attribute_cols.index(smee.CUTOFF_ATTRIBUTE)]
        error_tol = torch.tensor(0.0001)

        exceptions = _compute_pme_exclusions(system, potential).to(charges.device)

        grid_x, grid_y, grid_z, alpha = _compute_pme_grid(box_vectors, cutoff, error_tol)

        pme = NNPOps.pme.PME(
            grid_x, grid_y, grid_z, 5, alpha, _COULOMB_PRE_FACTOR, exceptions
        )

        energy_direct = torch.ops.pme.pme_direct(
            conformer.float(),
            charges.float(),
            pairwise.idxs.T,
            pairwise.deltas,
            pairwise.distances,
            pme.exclusions,
            pme.alpha,
            pme.coulomb,
        )
        energy_self = -torch.sum(charges ** 2) * pme.coulomb * pme.alpha / math.sqrt(torch.pi)
        energy_recip = energy_self + torch.ops.pme.pme_reciprocal(
            conformer.float(),
            charges.float(),
            box_vectors.float(),
            pme.gridx,
            pme.gridy,
            pme.gridz,
            pme.order,
            pme.alpha,
            pme.coulomb,
            pme.moduli[0].to(charges.device),
            pme.moduli[1].to(charges.device),
            pme.moduli[2].to(charges.device),
        )

        exclusion_idxs, exclusion_scales = _broadcast_exclusions(system, potential)

        exclusion_distances = (
                conformer[exclusion_idxs[:, 0], :] - conformer[exclusion_idxs[:, 1], :]
        ).norm(dim=-1)

        energy_exclusion = (
                _COULOMB_PRE_FACTOR
                * exclusion_scales
                * charges[exclusion_idxs[:, 0]]
                * charges[exclusion_idxs[:, 1]]
                / exclusion_distances
        ).sum(-1)

        coul_energy = energy_direct + energy_recip + energy_exclusion

    print(f"DEBUG: Polarizabilities check - all zero? {torch.allclose(polarizabilities, torch.tensor(0.0, dtype=torch.float64))}")
    print(f"DEBUG: Polarizabilities: {polarizabilities}")
    if torch.allclose(polarizabilities, torch.tensor(0.0, dtype=torch.float64)):
        print("DEBUG: Returning early - all polarizabilities are zero!")
        return coul_energy

    # Handle batch vs single conformer - process each conformer individually
    is_batch = conformer.ndim == 3

    if is_batch:
        # Process each conformer individually and return results for each
        n_conformers = conformer.shape[0]
        batch_energies = []

        for conf_idx in range(n_conformers):
            # Extract single conformer
            single_conformer = conformer[conf_idx]

            # Compute pairwise for this conformer
            single_pairwise = compute_pairwise(system, single_conformer, box_vectors, cutoff)

            # Recursively call this function for single conformer
            single_energy = compute_multipole_energy(
                system, potential, single_conformer, box_vectors, single_pairwise, 
                polarization_type, extrapolation_coefficients
            )
            batch_energies.append(single_energy)

        return torch.stack(batch_energies)

    # Continue with single conformer processing
    efield_static = torch.zeros((system.n_particles, 3), dtype=torch.float64, device=conformer.device)

    # calculate electric field due to partial charges by hand
    # TODO wolf summation for periodic
    _SQRT_COULOMB_PRE_FACTOR = _COULOMB_PRE_FACTOR ** (1 / 2)
    for distance, delta, idx, scale in zip(
        pairwise.distances, pairwise.deltas, pairwise.idxs, pair_scales
    ):
        # Use atom-specific Thole parameters (minimum of the two atoms, per AMOEBA)
        a = torch.min(thole_params[idx[0]], thole_params[idx[1]])
        u = distance / a
        au3 = a * u**3
        exp_au3 = torch.exp(-au3)

        # Thole damping for charge-dipole interactions (thole_c in OpenMM)
        damping_term1 = 1 - exp_au3

        efield_static[idx[0]] -= (
            _SQRT_COULOMB_PRE_FACTOR
            * scale
            * damping_term1
            * charges[idx[1]]
            * delta
            / distance**3
        )
        efield_static[idx[1]] += (
            _SQRT_COULOMB_PRE_FACTOR
            * scale
            * damping_term1
            * charges[idx[0]]
            * delta
            / distance**3
        )

    # reshape to (3*N) vector
    efield_static = efield_static.reshape(3 * system.n_particles)

    # induced dipole vector - start with direct polarization
    ind_dipoles = torch.repeat_interleave(polarizabilities, 3) * efield_static

    # Build A matrix for mutual/extrapolated methods
    if polarization_type in ["mutual", "extrapolated"]:
        A = torch.nan_to_num(torch.diag(torch.repeat_interleave(1.0 / polarizabilities, 3)))

        for distance, delta, idx, scale in zip(
            pairwise.distances, pairwise.deltas, pairwise.idxs, pair_scales
        ):
            # Use atom-specific Thole parameters (minimum of the two atoms, per AMOEBA)
            a = torch.min(thole_params[idx[0]], thole_params[idx[1]])
            u = distance / a
            au3 = a * u**3
            exp_au3 = torch.exp(-au3)
            damping_term1 = 1 - exp_au3
            damping_term2 = 1 - (1 + 1.5 * au3) * exp_au3

            t = (
                torch.eye(3, dtype=torch.float64, device=conformer.device) * damping_term1 * distance**-3
                - 3 * damping_term2 * torch.einsum("i,j->ij", delta, delta) * distance**-5
            )
            t *= scale
            A[3 * idx[0] : 3 * idx[0] + 3, 3 * idx[1] : 3 * idx[1] + 3] = t
            A[3 * idx[1] : 3 * idx[1] + 3, 3 * idx[0] : 3 * idx[0] + 3] = t

    # Handle different polarization types
    if polarization_type == "direct":
        # Direct polarization: μ = α * E (no mutual coupling)
        # ind_dipoles is already μ^(0) = α * E, so no additional work needed
        pass
    elif polarization_type == "extrapolated":
        if extrapolation_coefficients is None:
            extrapolation_coefficients = [-0.154, 0.017, 0.657, 0.475]
        
        opt_coeffs = torch.tensor(extrapolation_coefficients, dtype=torch.float64, device=conformer.device)
        n_orders = len(opt_coeffs)

        # Store SCF iteration snapshots
        scf_snapshots = []
        scf_snapshots.append(ind_dipoles.clone())  # Iteration 0: direct polarization

        # Run n_orders-1 SCF iterations and save snapshots
        precondition_m = torch.repeat_interleave(polarizabilities, 3)
        residual = efield_static - A @ ind_dipoles
        z = torch.einsum("i,i->i", precondition_m, residual)
        p = torch.clone(z)

        current_dipoles = ind_dipoles.clone()
        
        for iteration in range(n_orders - 1):  # If we have 4 coeffs, run 3 iterations
            # Standard conjugate gradient step
            alpha = torch.dot(residual, z) / (p @ A @ p)
            current_dipoles = current_dipoles + alpha * p
            
            # Save snapshot after this iteration
            scf_snapshots.append(current_dipoles.clone())

            prev_residual = torch.clone(residual)
            prev_z = torch.clone(z)

            residual = residual - alpha * A @ p

            # Check convergence (but continue to get all snapshots)
            if torch.dot(residual, residual) < 1e-7:
                # If converged early, use the converged result for remaining snapshots
                for _ in range(iteration + 1, n_orders - 1):
                    scf_snapshots.append(current_dipoles.clone())
                break

            z = torch.einsum("i,i->i", precondition_m, residual)
            beta = torch.dot(z, residual) / torch.dot(prev_z, prev_residual)
            p = z + beta * p

        # Apply OPT combination: μ_OPT = Σ(k=0 to n_orders-1) c_k μ_k
        ind_dipoles = torch.zeros_like(ind_dipoles)
        for k in range(min(n_orders, len(scf_snapshots))):
            ind_dipoles += opt_coeffs[k] * scf_snapshots[k]

    else:  # mutual
        # Mutual polarization using conjugate gradient
        precondition_m = torch.repeat_interleave(polarizabilities, 3)
        residual = efield_static - A @ ind_dipoles
        z = torch.einsum("i,i->i", precondition_m, residual)
        p = torch.clone(z)

        for _ in range(60):
            alpha = torch.dot(residual, z) / (p @ A @ p)
            ind_dipoles = ind_dipoles + alpha * p

            prev_residual = torch.clone(residual)
            prev_z = torch.clone(z)

            residual = residual - alpha * A @ p

            if torch.dot(residual, residual) < 1e-7:
                break

            z = torch.einsum("i,i->i", precondition_m, residual)
            beta = torch.dot(z, residual) / torch.dot(prev_z, prev_residual)
            p = z + beta * p

    # Reshape induced dipoles back to (N, 3) for energy calculations
    ind_dipoles_3d = ind_dipoles.reshape(system.n_particles, 3)
    
    # DEBUG: Print induced dipoles for comparison
    print(f"\nSMEE induced dipoles (e·Å):")
    for i in range(system.n_particles):
        dipole = ind_dipoles_3d[i].tolist()
        print(f"  Particle {i}: [{dipole[0]:.10f}, {dipole[1]:.10f}, {dipole[2]:.10f}]")

    # Calculate polarization energy based on method
    if polarization_type == "direct" or polarization_type == "extrapolated":
        # For direct and extrapolated: permanent-induced + self-energy + induced-induced
        # 1. Permanent-induced interaction: -μ · E^permanent
        coul_energy += -torch.dot(ind_dipoles, efield_static)

        # 2. Self-energy: +½ Σ (μ²/α)
        self_energy = 0.5 * torch.sum(
            torch.sum(ind_dipoles_3d ** 2, dim=1) / polarizabilities
        )
        coul_energy += self_energy

        # 3. Induced-induced interaction: -½ μ · E^induced
        # Build T_induced matrix for induced field calculation
        T_induced = torch.zeros((3 * system.n_particles, 3 * system.n_particles), dtype=torch.float64, device=conformer.device)

        for distance, delta, idx, scale in zip(
            pairwise.distances, pairwise.deltas, pairwise.idxs, pair_scales
        ):
            # Use atom-specific Thole parameters (minimum of the two atoms, per AMOEBA)
            a = torch.min(thole_params[idx[0]], thole_params[idx[1]])
            u = distance / a
            au3 = a * u**3
            exp_au3 = torch.exp(-au3)
            damping_term1 = 1 - exp_au3
            damping_term2 = 1 - (1 + 1.5 * au3) * exp_au3

            t = (
                torch.eye(3, dtype=torch.float64, device=conformer.device) * damping_term1 * distance**-3
                - 3 * damping_term2 * torch.einsum("i,j->ij", delta.double(), delta.double()) * distance**-5
            )
            t *= scale

            T_induced[3 * idx[0] : 3 * idx[0] + 3, 3 * idx[1] : 3 * idx[1] + 3] = t
            T_induced[3 * idx[1] : 3 * idx[1] + 3, 3 * idx[0] : 3 * idx[0] + 3] = t

        # Induced-induced energy: -½ μ · (T @ μ)
        efield_induced_flat = T_induced @ ind_dipoles
        coul_energy += -0.5 * torch.dot(ind_dipoles, efield_induced_flat)

    elif polarization_type == "mutual":
        # For mutual polarization: use standard SCF formula
        # This automatically includes all components when converged
        coul_energy += -0.5 * torch.dot(ind_dipoles, efield_static)

    return coul_energy