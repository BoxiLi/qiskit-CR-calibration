"""
This file is based on the tutorial of qiskit-dynamics.
"""

# Configure to use JAX internally
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from qiskit_dynamics.array import Array

Array.set_default_backend("jax")
from qiskit.test.mock import FakeLagos
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate
from qiskit.circuit import Parameter
from qiskit.providers.backend import QubitProperties
from qiskit import pulse
from qiskit.transpiler import InstructionProperties

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
)
from qiskit_dynamics import Solver
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library.calibration import RoughDragCal
from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal


def initilize_qiskit_dynamics_backend(
    f0, f1, a0, a1, J, cr_detuning=0.0, custom_experiment_result_function=None
):
    dim = 4

    v0 = f0
    anharm0 = a0
    r0 = 1.0e9 / 2 / np.pi

    v1 = f1
    anharm1 = a1
    r1 = 1.0e9 / 2 / np.pi

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))

    ident = np.eye(dim, dtype=complex)
    full_ident = np.eye(dim**2, dtype=complex)

    N0 = np.kron(ident, N)
    N1 = np.kron(N, ident)

    a0 = np.kron(ident, a)
    a1 = np.kron(a, ident)

    a0dag = np.kron(ident, adag)
    a1dag = np.kron(adag, ident)

    static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
    static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

    static_ham_full = (
        static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
    )

    drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
    drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

    # build solver
    dt = 1 / 4.5e9

    solver = Solver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],
        rotating_frame=static_ham_full,
        hamiltonian_channels=["d0", "d1", "u0", "u1"],
        channel_carrier_freqs={"d0": v0, "d1": v1, "u0": v1 + cr_detuning, "u1": v0},
        dt=dt,
    )

    # Consistent solver option to use throughout notebook
    # if "cuda" in str(jax.devices()[0]):
    #     solver_options = {
    #         "method": "jax_RK4_parallel",
    #         "max_dt": dt * 10
    #     }
    # else:
    solver_options = {
        "method": "jax_odeint",
        "atol": 1e-6,
        "rtol": 1e-8,
        "hmax": dt * 10,
    }

    if custom_experiment_result_function is None:
        backend = DynamicsBackend(
            solver=solver,
            subsystem_dims=[dim, dim],  # for computing measurement data
            solver_options=solver_options,  # to be used every time run is called
        )
    else:
        backend = DynamicsBackend(
            solver=solver,
            subsystem_dims=[dim, dim],  # for computing measurement data
            solver_options=solver_options,  # to be used every time run is called
            experiment_result_function=custom_experiment_result_function,
        )

    target = backend.target

    # qubit properties
    target.qubit_properties = [
        QubitProperties(frequency=v0),
        QubitProperties(frequency=v1),
    ]

    # add instructions
    target.add_instruction(XGate(), properties={(0,): None, (1,): None})
    target.add_instruction(SXGate(), properties={(0,): None, (1,): None})

    target.add_instruction(CXGate(), properties={(0, 1): None, (1, 0): None})

    # Add RZ instruction as phase shift for drag cal
    phi = Parameter("phi")
    with pulse.build() as rz0:
        pulse.shift_phase(phi, pulse.DriveChannel(0))
        pulse.shift_phase(phi, pulse.ControlChannel(1))

    with pulse.build() as rz1:
        pulse.shift_phase(phi, pulse.DriveChannel(1))
        pulse.shift_phase(phi, pulse.ControlChannel(0))

    target.add_instruction(
        RZGate(phi),
        {
            (0,): InstructionProperties(calibration=rz0),
            (1,): InstructionProperties(calibration=rz1),
        },
    )
    cals = Calibrations(libraries=[FixedFrequencyTransmon(basis_gates=["x", "sx"])])

    # rabi experiments for qubit 0
    rabi0 = RoughXSXAmplitudeCal(
        [0], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 27)
    )

    # rabi experiments for qubit 1
    rabi1 = RoughXSXAmplitudeCal(
        [1], cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 27)
    )

    rabi0_data = rabi0.run().block_for_results()
    rabi1_data = rabi1.run().block_for_results()

    cal_drag0 = RoughDragCal([0], cals, backend=backend, betas=np.linspace(-20, 20, 15))
    cal_drag1 = RoughDragCal([1], cals, backend=backend, betas=np.linspace(-20, 20, 15))

    cal_drag0.set_experiment_options(reps=[3, 5, 7])
    cal_drag1.set_experiment_options(reps=[3, 5, 7])

    drag0_data = cal_drag0.run().block_for_results()
    drag1_data = cal_drag1.run().block_for_results()

    backend.set_options(control_channel_map={(0, 1): 0, (1, 0): 1})
    backend.target.update_from_instruction_schedule_map(cals.get_inst_map())
    return backend
