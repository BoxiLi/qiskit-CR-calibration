"""
Generation and calibration of CR pulse.
"""

import os
import warnings
import multiprocessing as mpl
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

import numpy as np
from numpy import pi

try:
    import jax
    import jax.numpy as jnp
except:
    warnings("JAX not install, multi-derivative pulse doesn't work.")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import scipy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import symengine as sym
import matplotlib.pyplot as plt

from qiskit import pulse, circuit, schedule, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.pulse.library import (
    Gaussian,
    GaussianSquare,
    GaussianSquareDrag,
    SymbolicPulse,
    Waveform,
)
from qiskit.pulse import Play, ShiftPhase, DriveChannel, ScheduleBlock
from ._custom_jax import _value_and_jacfwd

from .job_util import (
    amp_to_omega_GHz,
    omega_GHz_to_amp,
    load_job_data,
    save_job_data,
    async_execute,
    read_calibration_data,
    save_calibration_data,
)

# Logger setup after importing logging
import logging

logger = logging.getLogger(__name__)


# %% Compute the DRAG pulse shape using JAX
def _complex_square_root(x):
    """
    Calculate the complex square root of a given complex number.

    Parameters:
    - x (complex): The input complex number.

    Returns:
    - complex: The complex square root of the input.
    """
    a = jnp.real(x)
    b = jnp.imag(x)
    result = jnp.sign(a) * (
        jnp.sqrt((jnp.abs(x) + a) / 2)
        + 1.0j * jnp.sign(b) * jnp.sqrt((jnp.abs(x) - a) / 2)
    )
    result = jnp.where(
        jnp.abs(b / a) < 1.0e-3,
        # Taylor expansion
        jnp.sign(a)
        * jnp.sqrt(jnp.abs(a))
        * (1 + (1.0j * b / a) / 2 - (1.0j * b / a) ** 2 / 8),
        result,
    )
    result = jnp.where(a == 0.0, 1.0j * b, result)
    return result


def perturbative_pulse_transform_single_photon(pulse_fun, gap, scale=1.0):
    """
    Add perturbative DRAG correction to a pulse based on the pulse function and energy gap.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The gap parameter for the transformation.
    - scale (float, optional): Scaling factor for the transformation. Default is 1.

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse, dt_pulse = _value_and_jacfwd(pulse_fun)(t, params)
        return pulse - 1.0j * dt_pulse / gap * scale

    return new_pulse_fun


def perturbative_pulse_transform_two_photon(pulse_fun, gap, gaps):
    """
    Add perturbative DRAG correction to a two-photon transition based on the pulse function and energy gaps.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The common gap parameter for the transformation.
    - gaps (tuple): Tuple containing two gap parameters (gap01, gap12).

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse, dt_pulse = _value_and_jacfwd(pulse_fun)(t, params)
        return _complex_square_root(pulse**2 - 2 * 1.0j * pulse * dt_pulse / gap)

    return new_pulse_fun


def exact_pulse_transform_single_photon(pulse_fun, gap, ratio=1, scale=1.0):
    """
    Apply an exact DRAG correction using Givens rotation to a pulse. It only works for single-photon transitions.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The gap parameter for the transformation.
    - ratio (float, optional): Ratio parameter. Default is 1.
    - scale (float, optional): Scaling factor for the transformation. Default is 1.

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse = pulse_fun(t, params)
        phase = jnp.where(pulse != 0.0, jnp.exp(1.0j * jnp.angle(pulse)), 0.0)

        def renorm_pulse_fun(t, params):
            pulse = pulse_fun(t, params)
            angle = jnp.angle(pulse)
            theta2 = jnp.arctan(
                -ratio * (params["omega"] / 2) * jnp.abs(pulse) / (gap / 2)
            )
            return theta2, angle

        (theta2, angle), (dt_theta2, dt_angle) = _value_and_jacfwd(renorm_pulse_fun)(
            t, params
        )
        dt_angle = jnp.where(pulse != 0.0, dt_angle, 0.0)
        angle = jnp.where(pulse != 0.0, angle, 0.0)
        new_g = phase * (-(gap + dt_angle) * jnp.tan(theta2) + scale * 1.0j * dt_theta2)
        return jnp.where(
            params["omega"] == 0.0,
            0.0,
            new_g / (ratio * (params["omega"])),
        )

    return new_pulse_fun


def pulse_shape(t, params):
    """
    Define a pulse shape function based on the specified parameters.

    Parameters:
    - t (float): Time parameter.
    - params (dict): Dictionary containing pulse parameters.

    Returns:
    - float: Value of the pulse function at the given time.
    """

    def pulse_fun(t, params):
        order = params["order"]
        if order == "2":
            fun = (
                lambda t, params: t / params["t_r"]
                - (jnp.cos(t * pi / params["t_r"]) * jnp.sin(t * pi / params["t_r"]))
                / pi
            )
        elif order == "1":
            fun = lambda t, params: jnp.sin(jnp.pi / 2 / params["t_r"] * t) ** 2
        elif order == "3":
            t_r = params["t_r"]
            fun = (
                lambda t, params: (jnp.cos(3 * pi * t / t_r)) / 16
                - (9 / 16) * jnp.cos(pi * t / t_r)
                + 1 / 2
            )
        else:
            raise ValueError("Pulse shape order not defined.")
        if "t_r" in params:
            t_r = params["t_r"]
            t_tol = params["t_tol"]
            max_value = fun(params["t_r"], params)
            pulse = max_value
            pulse = jnp.where(t < t_r, fun(t, params), pulse)
            pulse = jnp.where(t > t_tol - t_r, fun(t_tol - t, params), pulse)
            return pulse
        else:
            raise NotImplementedError

    if "drag_scale" not in params or params["drag_scale"] is None:
        drag_scale = jnp.ones(3)
    else:
        drag_scale = jnp.array(params["drag_scale"])

    if "no_drag" in params:
        return pulse_fun(t, params)

    detuning = params["Delta"]
    gap02 = 2 * detuning + params["a1"]
    gap12 = detuning + params["a1"]
    gap01 = detuning

    if "01" in params:
        # Only single-derivative DRAG with gap for level 0 and 1. Use beta from the input parameters as a scaling factor for the DRAG coefficient.
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01, scale=params.get("beta", 0.0)
        )
    elif "01_scale" in params:
        # Only single-derivative DRAG with gap for level 0 and 1.
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01, scale=drag_scale[0]
        )
    elif "12" in params:
        # Only single-derivative DRAG with gap for level 1 and 2.
        pulse_fun = perturbative_pulse_transform_single_photon(pulse_fun, gap12)
    elif "02" in params:
        # Only single-derivative DRAG with gap for level 0 and 2.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02, (gap01, gap12)
        )
    elif "exact" in params:
        # Full correction with Givens rotatino for the two single-photon transitions and the perturbative DRAG correction for the two-photon process.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02 / params["drag_scale"][2], (gap01, gap12)
        )
        pulse_fun = exact_pulse_transform_single_photon(
            pulse_fun, gap01 / params["drag_scale"][0], ratio=1
        )
        pulse_fun = exact_pulse_transform_single_photon(
            pulse_fun,
            gap12 / params["drag_scale"][1],
            ratio=jnp.sqrt(2),
            # *(1 - gap12/gap01)
        )
    elif "sw" in params:
        # Full correction for all the three transitions, only use perturabtive DRAG correction.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02 / drag_scale[2], (gap01, gap12)
        )
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01 / drag_scale[0]
        )
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun,
            gap12 / drag_scale[1],
        )
    else:
        raise ValueError("Unknown DRAG type.")
    final_pulse = pulse_fun(t, params)
    return final_pulse


# %% Build CR tomography circuit and send jobs


def get_default_cr_params(backend, qc, qt):
    """
    Get the default parameters for the echoed CNOT gate from Qiskit.

    Parameters:
    - backend (qiskit.providers.backend.Backend): The Qiskit backend.
    - qc (int): Control qubit index.
    - qt (int): Target qubit index.

    Returns:
    Tuple[Dict, Dict]: Tuple containing dictionaries of parameters for the echoed CNOT gate for control and target qubits.
    """
    inst_sched_map = backend.instruction_schedule_map

    def _filter_fun(instruction, pulse="ZX"):
        if pulse == "ZX" and "CR90p_u" in instruction[1].pulse.name:
            return True
        elif pulse == "ZX" and "CX_u" in instruction[1].pulse.name:
            return True
        elif pulse == "IX" and "CR90p_d" in instruction[1].pulse.name:
            return True
        elif pulse == "IX" and "CX_d" in instruction[1].pulse.name:
            return True
        return False

    def get_cx_gate_schedule(qc, qt):
        from qiskit.pulse import PulseError

        try:
            return inst_sched_map.get("ecr", (qc, qt))
        except PulseError:
            return inst_sched_map.get("cx", (qc, qt))

    gate_schedule = get_cx_gate_schedule(qc, qt)
    cr_sched_default = gate_schedule.filter(instruction_types=[Play]).filter(
        _filter_fun
    )
    cr_instruction = cr_sched_default.instructions[0][1]
    cr_pulse = cr_instruction.pulse

    ix_sched_default = gate_schedule.filter(instruction_types=[Play]).filter(
        partial(_filter_fun, pulse="IX")
    )
    ix_instruction = ix_sched_default.instructions[0][1]
    ix_channel = ix_instruction.channel
    ix_pulse = ix_instruction.pulse
    return cr_pulse.parameters, ix_pulse.parameters


def _get_modulated_symbolic_pulse(pulse_class, backend, params, frequency_offset):
    """
    Add a detuning to the pulse by pulse shape modulation with exp(2*pi*I*delta*t).

    Parameters:
    - pulse_class (type): The class of the pulse.
    - backend (qiskit.providers.backend.Backend): The Qiskit backend.
    - params (dict): Pulse parameters.
    - frequency_offset (float): Frequency offset for detuning.

    Returns:
    SymbolicPulse: Modulated symbolic pulse.
    """
    default_pulse = pulse_class(**params)
    pulse_parameters = default_pulse.parameters
    pulse_parameters["delta"] = frequency_offset * backend.dt
    _t, _delta = sym.symbols("t, delta")
    my_pulse = SymbolicPulse(
        pulse_type="GaussianModulated",
        duration=default_pulse.duration,
        parameters=pulse_parameters,
        envelope=default_pulse.envelope * sym.exp(2 * sym.pi * sym.I * _delta * _t),
        name="modulated Gaussian",
    )
    return my_pulse


def _add_symbolic_gaussian_pulse(pulse1, pulse2):
    """
    Add two symbolic GaussianSquare pulses.
    It is used in the target drive for the direct CNOT gate calibration to separate the parameter from the echoed CNOT gate.

    Parameters:
    - pulse1 (SymbolicPulse): First symbolic GaussianSquare pulse.
    - pulse2 (SymbolicPulse): Second symbolic GaussianSquare pulse.

    Returns:
    SymbolicPulse: Resulting pulse after adding the two pulses.
    """
    edited_params2 = {key + "_": value for key, value in pulse2.parameters.items()}
    edited_pulse2_envelop = pulse2.envelope
    for p_str in pulse2.parameters.keys():
        old_p = sym.symbols(p_str)
        new_p = sym.symbols(p_str + "_")
        edited_pulse2_envelop = edited_pulse2_envelop.xreplace(old_p, new_p)
    pulse_parameters = pulse1.parameters.copy()
    pulse_parameters.update(edited_params2)
    if pulse1.duration != pulse2.duration:
        raise RuntimeError("Pulse duration must be the same.")
    my_pulse = SymbolicPulse(
        pulse_type="SumGaussian",
        duration=pulse1.duration,
        parameters=pulse_parameters,
        envelope=pulse1.envelope + edited_pulse2_envelop,
        name="added_pulse",
    )
    return my_pulse


def _add_waveform(waveform1, waveform2):
    """Add two qiskit waveform samplings.

    Parameters:
    - waveform1 (Waveform): First qiskit waveform.
    - waveform2 (Waveform): Second qiskit waveform.

    Returns:
    Waveform: Resulting waveform after adding the two waveforms.
    """
    if len(waveform1.samples) != len(waveform2.samples):
        raise ValueError("The two waveforms do not have the same length.")
    return Waveform(
        samples=waveform1.samples + waveform2.samples,
        name=waveform1.name,
    )


def get_custom_pulse_shape(params, backend, control_qubit, frequency_offset=0.0):
    """Compute the custom sine-shaped pulse with DRAG correction for the custom CR gate. It is only possible to get the array sampling of the waveform, no symbolic expressions.

    Parameters:
    - params (dict): Dictionary containing parameters for pulse shape computation.
    - backend: Qiskit backend.
    - control_qubit: Index of the control qubit.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.

    Returns:
    Waveform: Computed pulse shape with DRAG correction.
    """

    def regulate_array_length(array):
        result_array = np.zeros((len(array) // 16 + 1) * 16)
        result_array[: len(array)] = array
        return result_array

    params = params.copy()
    # Computation of the DRAG pulse needs the final time in ns, transfer "duration" to the final time "t_tol".
    if "duration" in params:
        params["t_tol"] = params["duration"] * backend.dt * 1.0e9
    elif "t_tol" in params:
        pass
    else:
        ValueError(
            "The total time of the CR pulse t_tol is not defined in the parameter dictionary."
        )
    # If amp is given, comute the the drive strength in 2*pi*GHz
    params["omega"] = 2 * pi * amp_to_omega_GHz(backend, control_qubit, params["amp"])
    # Choose the DRAG type
    if "drag_type" in params and params["drag_type"]:
        params[params["drag_type"]] = True
    else:
        params["no_drag"] = True
    del params["drag_type"]

    # Generate time array
    tlist = np.arange(0, params["t_tol"], backend.dt * 1.0e9)
    tlist = regulate_array_length(tlist)

    # Compute pulse shape with DRAG correction using JAX
    pulse_shape_sample = params["amp"] * jax.vmap(pulse_shape, (0, None))(
        tlist,
        params,
    )
    del params["amp"]

    # Conjugate the pulse shape array (because of the qiskit convention) and apply frequency offset
    pulse_shape_sample = np.array(pulse_shape_sample).conjugate()
    pulse_shape_sample = pulse_shape_sample * np.exp(
        2
        * np.pi
        * 1.0j
        * frequency_offset
        * backend.dt
        * np.arange(len(pulse_shape_sample))
    )
    pulse_shape_sample = pulse_shape_sample * np.exp(1.0j * params["angle"])

    # Return the computed waveform
    return Waveform(
        samples=pulse_shape_sample,
        name="CR drive",
    )


def get_cr_schedule(qubits, backend, **kwargs):
    """
    Generate the qiskit Schedule for CR pulse, including both the CR drive and the target.

    Args:
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      **kwargs: Additional keyword arguments:
        - cr_params (dict): Parameters for the CR pulse.
        - ix_params (dict): Parameters for the IX pulse.
        - x_gate_ix_params (dict, optional): Additional parameters for the X-gate on the target IX pulse. Default is None.
        - frequency_offset (float, optional): Frequency offset in . Default is 0.0.

    Returns:
      Schedule: Qiskit Schedule for the CR pulse.
    """
    cr_params = kwargs["cr_params"].copy()
    ix_params = kwargs["ix_params"].copy()
    frequency_offset = kwargs.get("frequency_offset", 0.0)
    x_gate_ix_params = kwargs.get("x_gate_ix_params", None)

    cr_params["risefall_sigma_ratio"] = 2.0
    ix_params["risefall_sigma_ratio"] = 2.0
    if x_gate_ix_params is not None:
        x_gate_ix_params["risefall_sigma_ratio"] = 2.0
    if "width" in cr_params:
        del cr_params["width"]
    if "width" in ix_params:
        del ix_params["width"]
    qc, qt = qubits

    with pulse.build(backend) as cr_schedule:
        control_channel = pulse.control_channels(qc, qt)[0]
        target_channel = pulse.drive_channel(qt)
        if "beta" not in ix_params:
            ix_params["beta"] = 0.0

        # Generate CR pulse waveform
        if "drag_type" in cr_params:
            cr_pulse = get_custom_pulse_shape(
                cr_params, backend, qc, frequency_offset=frequency_offset
            )
        else:
            cr_pulse = _get_modulated_symbolic_pulse(
                GaussianSquare, backend, cr_params, frequency_offset
            )
        # Generate IX pulse waveform
        if "drag_type" in ix_params:
            ix_pulse = get_custom_pulse_shape(
                ix_params, backend, qc, frequency_offset=frequency_offset
            )
            if x_gate_ix_params is not None:
                x_gate_pulse = get_custom_pulse_shape(
                    x_gate_ix_params, backend, qc, frequency_offset
                )
                ix_pulse = _add_waveform(ix_pulse, x_gate_pulse)
        else:
            ix_pulse = _get_modulated_symbolic_pulse(
                GaussianSquareDrag, backend, ix_params, frequency_offset
            )
            if x_gate_ix_params is not None:
                x_gate_pulse = _get_modulated_symbolic_pulse(
                    GaussianSquare, backend, x_gate_ix_params, frequency_offset
                )
                ix_pulse = _add_symbolic_gaussian_pulse(ix_pulse, x_gate_pulse)
        # Check if CR and IX pulse durations are equal
        if cr_pulse.duration != ix_pulse.duration:
            raise RuntimeError(
                f"CR and IX pulse duration are not equal {cr_pulse.duration} and {ix_pulse.duration}."
            )

        pulse.play(cr_pulse, control_channel)
        pulse.play(ix_pulse, target_channel)

        # Add the detuning
        tmp_circ = QuantumCircuit(backend.num_qubits)
        if backend.name == "DynamicsBackend":
            pass
            tmp_circ.rz(
                +2 * pi * frequency_offset * backend.dt * ix_pulse.duration,
                qt,
            )
        else:
            tmp_circ.rz(
                -2 * pi * frequency_offset * backend.dt * ix_pulse.duration,
                qt,
            )
        pulse.call(schedule(tmp_circ, backend=backend))
    return cr_schedule


def _generate_indiviual_circuit(
    duration,
    qubits,
    backend,
    cr_params,
    ix_params,
    x_gate_ix_params,
    frequency_offset,
    control_states,
):
    """
    Generate tomography circuits for a given CR pulse duration.

    Args:
      duration (float): Duration of the CR pulse.
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      cr_params (dict): Parameters for the CR pulse.
      ix_params (dict): Parameters for the IX pulse.
      x_gate_ix_params (dict): Parameters for the X-gate on the IX pulse.
      frequency_offset (float): Frequency offset.
      control_states (Tuple): Control states for the tomography circuits.

    Returns:
      List[QuantumCircuit]: List of generated tomography circuits.
    """
    cr_gate = circuit.Gate("cr", num_qubits=2, params=[])
    (qc, qt) = qubits
    circ_list = []
    cr_params["duration"] = int(duration)
    ix_params["duration"] = int(duration)
    if x_gate_ix_params is not None:
        x_gate_ix_params["duration"] = int(duration)

    # Get CR pulse schedule
    cr_sched = get_cr_schedule(
        (qc, qt),
        backend,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
    )

    for basis in ["X", "Y", "Z"]:
        for control in control_states:
            tomo_circ = circuit.QuantumCircuit(
                backend.num_qubits, 2
            )  # use all qubits to avoid error
            if control in (1, "1"):
                tomo_circ.x(qc)  # flip control from |0> to |1>
            tomo_circ.append(cr_gate, [qc, qt])
            tomo_circ.barrier(qc, qt)
            if basis == "X":
                tomo_circ.h(qt)
            elif basis == "Y":
                tomo_circ.sdg(qt)
                tomo_circ.h(qt)
            tomo_circ.measure(qc, 0)
            tomo_circ.measure(qt, 1)
            tomo_circ.add_calibration(gate=cr_gate, qubits=[qc, qt], schedule=cr_sched)
            circ_list.append(tomo_circ)
    return circ_list


def get_cr_tomo_circuits(
    qubits,
    backend,
    cr_times,
    cr_params,
    ix_params,
    x_gate_ix_params=None,
    frequency_offset=0.0,
    control_states=(0, 1),
):
    """
    Build an array of cross resonance schedules for the Hamiltonian tomography experiment.

    Args:
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      cr_times (List[int]): List of CR pulse durations.
      cr_params (dict): Parameters for the CR pulse
      ix_params (dict): Parameters for the IX pulse.
      x_gate_ix_params (dict, optional): Parameters for the X-gate on the IX pulse. Default is None.
      frequency_offset (float, optional): Frequency offset. Default is 0.0.
      control_states (Tuple, optional): Control states for the tomography circuits. Default is (0, 1).

    Returns:
      List[QuantumCircuit]: List of generated tomography circuits.
    """
    tomo_circs = []

    cr_params = cr_params.copy()
    ix_params = ix_params.copy()
    if x_gate_ix_params is not None:
        x_gate_ix_params = x_gate_ix_params.copy()
    tomo_circs = []

    tmp_fun = partial(
        _generate_indiviual_circuit,
        qubits=qubits,
        backend=backend,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
        control_states=control_states,
    )

    # with mpl.Pool(10) as p:
    #     tomo_circs = p.map(tmp_fun, cr_times)
    tomo_circs = map(tmp_fun, cr_times)

    tomo_circs = sum(tomo_circs, [])
    logger.info("Tomography circuits have been generated.")
    return tomo_circs


def _send_decoy_cicuit(backend, session):
    """
    Send a very simple decoy circuit to the backend to prevent idling.

    Parameters:
    - backend (Backend): Qiskit backend.
    - session (Session): Qiskit session.
    """
    qc = QuantumCircuit(1)
    qc.x(0)
    qc = transpile(
        qc,
        backend,
        scheduling_method="asap",
        optimization_level=1,
    )
    job = session.run(
        "circuit-runner",
        inputs={
            "circuits": qc,
            "skip_transpilation": True,
            "shots": 10,
        },
    )
    return job.job_id()


def send_cr_tomography_job(
    qubits,
    backend,
    cr_params,
    ix_params,
    cr_times,
    x_gate_ix_params=None,
    blocking=False,
    frequency_offset=0.0,
    session=None,
    shots=1024,
    control_states=(0, 1),
    decoy=False,
):
    """
    Send the tomography job for CR pulse.

    Parameters:
    - qubits (Tuple): Tuple of control and target qubits.
    - backend (Backend): Qiskit backend.
    - cr_params (dict): Parameters for the CR pulse.
    - ix_params (dict): Parameters for the IX pulse.
    - cr_times (List[float]): List of widths of the CR pulses.
    - x_gate_ix_params (dict, optional): Parameters for the X-gate IX pulse. Default is None.
    - blocking (bool, optional): If True, block until the job is completed. Default is False.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.
    - ZI_correction (float, optional): ZI interaction rate correction. Default is 0.0.
    - session (Session, optional): Qiskit session. Default is None.
    - shots (int, optional): Number of shots. Default is 1024.
    - control_states (Tuple, optional): Tuple of control states. Default is (0, 1).
    - decoy (bool, optional): If True, send a decoy circuit to prevent idling. Default is True.

    Returns:
    str: Job ID.
    """
    # Create circuits
    cr_tomo_circ_list = get_cr_tomo_circuits(
        qubits,
        backend,
        cr_times,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
        control_states=control_states,
    )
    if decoy:
        _send_decoy_cicuit(backend, session)

    transpiled_tomo_circ_list = transpile(
        cr_tomo_circ_list,
        backend,
        # scheduling_method="asap",
        optimization_level=1,
    )
    # Send jobs
    if session is not None:
        job = session.run(
            "circuit-runner",
            inputs={
                "circuits": transpiled_tomo_circ_list,
                "skip_transpilation": True,
                "shots": shots,
            },
        )
    else:
        job = backend.run(transpiled_tomo_circ_list, shots=shots)
    tag = "CR tomography"
    # job.update_tags([tag])  # qiskit-ibm-runtime does not support yet update_tags, will support soon.
    parameters = {
        "backend": backend.name,
        "qubits": qubits,
        "cr_times": cr_times,
        "shots": shots,
        "cr_params": cr_params,
        "ix_params": ix_params,
        "x_gate_ix_params": x_gate_ix_params,
        "frequency_offset": frequency_offset,
        "dt": backend.dt,
    }
    logger.info(
        tag
        + ": "
        + job.job_id()
        + "\n"
        + "\n".join([f"{key}: {val}" for key, val in parameters.items()])
        + "\n"
    )
    if blocking:
        save_job_data(job, backend=backend, parameters=parameters)
    else:
        async_execute(save_job_data, job, backend=backend, parameters=parameters)
    return job.job_id()


def shifted_parameter_cr_job(
    qubits,
    backend,
    cr_params,
    ix_params,
    cr_times,
    prob_ix_strength,
    x_gate_ix_params=None,
    frequency_offset=0.0,
    shots=1024,
    blocking=False,
    session=None,
    control_states=(0, 1),
    mode="CR",
):
    """
    Send the tomography job for CR pulse with a shifted amplitude on the target drive.

    Parameters:
    - qubits (Tuple): Tuple of control and target qubits.
    - backend (Backend): Qiskit backend.
    - cr_params (dict): Parameters for the CR pulse.
    - ix_params (dict): Parameters for the IX pulse.
    - cr_times (List[float]): List of widths of the CR pulses.
    - prob_ix_strength (float): Strength of the probability amplitude shift.
    - x_gate_ix_params (dict, optional): Parameters for the X-gate IX pulse. Default is None.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.
    - shots (int, optional): Number of shots. Default is 1024.
    - blocking (bool, optional): If True, block until the job is completed. Default is False.
    - session (Session, optional): Qiskit session. Default is None.
    - control_states (Tuple, optional): Tuple of control states. Default is (0, 1).
    - mode (str, optional): Operation mode ("CR" or "IX"). Default is "CR".

    Returns:
    str: Job ID.
    """
    if mode == "CR":
        ix_params = ix_params.copy()
        ix_params["amp"] += prob_ix_strength
    else:
        x_gate_ix_params = x_gate_ix_params.copy()
        x_gate_ix_params["amp"] += prob_ix_strength
    tomo_id = send_cr_tomography_job(
        qubits,
        backend,
        cr_params,
        ix_params,
        cr_times,
        x_gate_ix_params,
        blocking=blocking,
        session=session,
        frequency_offset=frequency_offset,
        shots=shots,
        control_states=control_states,
    )
    return tomo_id


# %% Analyze CR tomography data
# Part of the following code is from https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/hamiltonian-tomography.ipynb
def _get_omega(eDelta, eOmega_x, eOmega_y):
    r"""Return \Omega from parameter arguments."""
    eOmega = np.sqrt(eDelta**2 + eOmega_x**2 + eOmega_y**2)
    return eOmega


def _avg_X(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average X Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eXt = (
        -eDelta * eOmega_x
        + eDelta * eOmega_x * np.cos(eOmega * t + t0)
        + eOmega * eOmega_y * np.sin(eOmega * t + t0)
    ) / eOmega**2
    return eXt


def _avg_Y(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average Y Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eYt = (
        eDelta * eOmega_y
        - eDelta * eOmega_y * np.cos(eOmega * t + t0)
        - eOmega * eOmega_x * np.sin(eOmega * t + t0)
    ) / eOmega**2
    return eYt


def _avg_Z(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average Z Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eZt = (
        eDelta**2 + (eOmega_x**2 + eOmega_y**2) * np.cos(eOmega * t + t0)
    ) / eOmega**2
    return eZt


def fit_evolution(tlist, eXt, eYt, eZt, p0, include, normalize):
    """
    Use curve_fit to determine fit parameters of X,Y,Z Pauli measurements together.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli measurements.
    - eYt (array-like): Y Pauli measurements.
    - eZt (array-like): Z Pauli measurements.
    - p0 (array-like): Initial guess for fit parameters.
    - include (array-like): Boolean array specifying which Pauli measurements to include in the fit.
    - normalize (bool): Whether to normalize the measurements.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """

    def fun(tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0):
        """Stack average X,Y,Z Pauli measurements vertically."""
        result = np.vstack(
            [
                _avg_X(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Y(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Z(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
            ]
        )[include].flatten()
        return result

    data = np.asarray(
        [
            eXt,
            eYt,
            eZt,
        ]
    )[include].flatten()
    params, cov = curve_fit(fun, tlist, data, p0=p0, method="trf")
    return params, cov


def fit_rt_evol(tlist, eXt, eYt, eZt, p0, include, normalize):
    """
    Use curve_fit to determine fit parameters of X,Y,Z Pauli measurements together.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli measurements.
    - eYt (array-like): Y Pauli measurements.
    - eZt (array-like): Z Pauli measurements.
    - p0 (array-like): Initial guess for fit parameters.
    - include (array-like): Boolean array specifying which Pauli measurements to include in the fit.
    - normalize (bool): Whether to normalize the measurements.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """

    def fun(tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0):
        """
        Stack average X,Y,Z Pauli measurements vertically.
        """
        result = np.vstack(
            [
                _avg_X(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Y(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Z(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
            ]
        )[include].flatten()
        return result

    data = np.asarray(
        [
            eXt,
            eYt,
            eZt,
        ]
    )[include].flatten()
    params, cov = curve_fit(fun, tlist, data, p0=p0, method="trf")
    return params, cov


def recursive_fit(tlist, eXt, eYt, eZt, p0):
    """
    Perform recursive fitting of X, Y, Z Pauli expectation values.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli expectation values.
    - eYt (array-like): Y Pauli expectation values.
    - eZt (array-like): Z Pauli expectation values.
    - p0 (array-like): Initial guess for fit parameters.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """
    params = p0.copy()
    # First fit with Z measurement only, no normalization
    include = np.array([False, False, True])
    params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)

    # Second fit with Z measurement only, normalization applied
    include = np.array([False, False, True])
    params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=True)

    # Third fit with Y and Z measurements, no normalization
    include = np.array([False, True, True])
    params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)

    # Fourth fit with X, Y, and Z measurements, no normalization
    include = np.array([True, True, True])
    params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)

    # Fifth fit with X, Y, and Z measurements, normalization applied
    include = np.array([True, True, True])
    params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=True)
    return params, cov


def get_interation_rates_MHz(ground_params, excited_params, ground_cov, excited_cov):
    """
    Determine two-qubits interaction rates from fits to ground and excited control qubit data.

    Parameters:
    - ground_params (array-like): Fit parameters for the ground state.
    - excited_params (array-like): Fit parameters for the excited state.
    - ground_cov (array-like): Covariance matrix for the ground state fit.
    - excited_cov (array-like): Covariance matrix for the excited state fit.

    Returns:
    Tuple[array-like, array-like]: Interaction rates and their standard deviations.
    """
    Delta0, Omega0_x, Omega0_y = ground_params[:3]
    Delta1, Omega1_x, Omega1_y = excited_params[:3]
    Delta0_var, Omega0_x_var, Omega0_y_var = np.diag(ground_cov)[:3]
    Delta1_var, Omega1_x_var, Omega1_y_var = np.diag(excited_cov)[:3]

    # Interaction rates
    IX = 0.5 * (Omega0_x + Omega1_x) / 2 / pi
    IY = 0.5 * (Omega0_y + Omega1_y) / 2 / pi
    IZ = 0.5 * (Delta0 + Delta1) / 2 / pi
    ZX = 0.5 * (Omega0_x - Omega1_x) / 2 / pi
    ZY = 0.5 * (Omega0_y - Omega1_y) / 2 / pi
    ZZ = 0.5 * (Delta0 - Delta1) / 2 / pi

    # Standard deviations
    IX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    IY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    IZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi
    ZX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    ZY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    ZZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi

    return [[IX, IY, IZ], [ZX, ZY, ZZ]], [
        [IX_std, IY_std, IZ_std],
        [ZX_std, ZY_std, ZZ_std],
    ]


def _estimate_period(data, cr_times):
    """
    Estimate the period of the oscillatory data using peak finding.

    Parameters:
    - data (array-like): Oscillatory data.
    - cr_times (array-like): Corresponding time values.

    Returns:
    float: Estimated period of the oscillatory data.
    """
    peaks_high, properties = scipy.signal.find_peaks(data, prominence=0.5)
    peaks_low, properties = scipy.signal.find_peaks(-data, prominence=0.5)
    peaks = sorted(np.concatenate([peaks_low, peaks_high]))
    if len(peaks) <= 2:
        return cr_times[-1] - cr_times[0]
    return 2 * np.mean(np.diff(cr_times[peaks]))


def _get_normalized_cr_tomography_data(job_id):
    """Retrieve and normalize CR tomography data from a job. Renormalize the data to (-1, 1).

    Args:
        job_id (str): The ID of the job containing CR tomography data.

    Returns:
        Tuple[array-like, array-like]: A tuple containing the CR times and normalized tomography data.
    """
    data = load_job_data(job_id)
    result = data["result"]
    dt = data["parameters"]["dt"]
    cr_times = data["parameters"]["cr_times"]
    shots = data["parameters"]["shots"]

    # IBM classified data
    # Trace out the control, notice that IBM uses reverse qubit indices labeling.
    target_data = (
        np.array(
            [
                (result.get_counts(i).get("00", 0) + result.get_counts(i).get("01", 0))
                for i in range(len(result.results))
            ]
        )
        / shots
    )

    if 6 * len(cr_times) == len(target_data):
        # two-qubit tomography, with control on 0 and 1
        splitted_data = target_data.reshape((len(cr_times), 6)).transpose()
    elif 3 * len(cr_times) == len(target_data):
        # single tomography
        splitted_data = target_data.reshape((len(cr_times), 3)).transpose()
    else:
        ValueError(
            "The number of data points does not match the number of tomography settings."
        )
    splitted_data = splitted_data * 2 - 1

    scale = np.max(splitted_data) - np.min(splitted_data)
    average = (np.max(splitted_data) + np.min(splitted_data)) / 2
    splitted_data = 2 * (splitted_data - average) / scale

    return cr_times, splitted_data, dt


def process_single_qubit_tomo_data(job_id, show_plot=False):
    """Process and analyze single qubit tomography data from a job.

    Args:
        job_id (str): The ID of the job containing single qubit tomography data.
        show_plot (bool, optional): Whether to generate and display plots. Default is False.

    Returns:
        dict: Dictionary containing the processed results including <X>, <Y>, and <Z>.

    Note:
        Noticed that the measured value is not the IX, IY, IZ in the sense of CR, but the single qubit dynamics when the control qubit is in |0> or |1>.
    """
    cr_times, splitted_data, dt = _get_normalized_cr_tomography_data(job_id)
    signal_x, signal_y, signal_z = splitted_data

    period = _estimate_period(signal_z, cr_times)
    cutoff = -1
    params, cov = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[:cutoff],
        signal_y[:cutoff],
        signal_z[:cutoff],
        p0=np.array([1, 1, 1, 1 / (period * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    if show_plot:
        plot_cr_ham_tomo(
            cr_times * dt * 1.0e6,
            splitted_data,
            ground_params=params,
            ground_cov=cov,
        )
        plt.show()
    return {
        "IX": params[1] / 2 / pi,
        "IY": params[2] / 2 / pi,
        "IZ": params[0] / 2 / pi,
    }


## TODO remove dt in the signiture.
def process_zx_tomo_data(job_id, show_plot=False):
    """Process and analyze ZX tomography data from a job.

    Args:
        job_id (str): The ID of the job containing ZX tomography data.
        show_plot (bool, optional): Whether to generate and display plots. Default is False.

    Returns:
        dict: Dictionary containing the processed results including IX, IY, IZ, ZX, ZY, and ZZ.

    Note:
        The effective coupling strength is in the unit of MHz.
    """
    cr_times, splitted_data, dt = _get_normalized_cr_tomography_data(job_id)
    signal_x = splitted_data[:2]
    signal_y = splitted_data[2:4]
    signal_z = splitted_data[4:6]

    period0 = _estimate_period(signal_z[0], cr_times)
    period1 = _estimate_period(signal_z[1], cr_times)

    cutoff = -1
    _i = 0
    while True:
        try:
            ground_params, ground_cov = recursive_fit(
                cr_times[:cutoff] * dt * 1.0e6,
                signal_x[0][:cutoff],
                signal_y[0][:cutoff],
                signal_z[0][:cutoff],
                p0=np.array([1, 1, 1, 1 / (period0 * dt * 1.0e6) * 2 * pi, 0.0]),
            )
            break
        except RuntimeError as e:
            _i += 1
            period0 *= 2
            if _i > 16:
                raise e
            

    excited_params, excited_cov = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[1][:cutoff],
        signal_y[1][:cutoff],
        signal_z[1][:cutoff],
        p0=np.array([1, 1, 1, 1 / (period1 * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    # ground_params, ground_cov = excited_params, excited_cov
    if show_plot:
        plot_cr_ham_tomo(
            cr_times * dt * 1.0e6,
            splitted_data,
            ground_params,
            excited_params,
            ground_cov,
            excited_cov,
        )
        plt.show()
    [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz(
        ground_params, excited_params, ground_cov, excited_cov
    )[0]
    return {"IX": IX, "IY": IY, "IZ": IZ, "ZX": ZX, "ZY": ZY, "ZZ": ZZ}


def plot_cr_ham_tomo(
    cr_times,
    tomography_data,
    ground_params,
    excited_params=None,
    ground_cov=None,
    excited_cov=None,
):
    """Plot Hamiltonian tomography data and curve fits with interaction rates.

    Args:
        cr_times (np.ndarray): Array of CR times.
        tomography_data (np.ndarray): Averaged tomography data.
        ground_params (np.ndarray): Parameters of the ground fit.
        excited_params (np.ndarray, optional): Parameters of the excited fit. Default is None.
        ground_cov (np.ndarray, optional): Covariance matrix of the ground fit. Default is None.
        excited_cov (np.ndarray, optional): Covariance matrix of the excited fit. Default is None.
    """
    colors = ["tab:blue", "tab:red"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4), sharey=True)
    is_single_qubit_tomo = excited_params is None

    if not is_single_qubit_tomo:
        ground_avg = tomography_data[0::2, :]
        excited_avg = tomography_data[1::2, :]
    else:
        ground_avg = tomography_data
    # Scatter plot and curve for X(t)
    ax1.scatter(
        cr_times,
        ground_avg[0, :],
        lw=0.3,
        color=colors[0],
        label=r"control in $|0\rangle$" if not is_single_qubit_tomo else None,
    )
    ax1.plot(cr_times, _avg_X(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax1.set_ylabel(r"$\langle X(t) \rangle$", fontsize="small")
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    # Scatter plot and curve for Y(t)
    ax2.scatter(
        cr_times, ground_avg[1, :], lw=0.3, color=colors[0], label="control in |0>"
    )
    ax2.plot(cr_times, _avg_Y(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax2.set_ylabel(r"$\langle Y(t) \rangle$", fontsize="small")
    ax2.set_xticklabels([])
    # Scatter plot and curve for Z(t)
    ax3.scatter(
        cr_times,
        ground_avg[2, :],
        lw=0.3,
        color=colors[0],
        label=r"control in $|0\rangle$",
    )
    ax3.plot(cr_times, _avg_Z(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax3.set_ylabel(r"$\langle Z(t) \rangle$", fontsize="small")
    ax3.set_yticklabels([])
    ax3.set_xlabel(r"Time $(\mu s)$", fontsize="small")

    if not is_single_qubit_tomo:
        ax1.scatter(
            cr_times,
            excited_avg[0, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax1.plot(cr_times, _avg_X(cr_times, *excited_params), lw=2.0, color=colors[1])
        ax1.legend(loc=4, fontsize="x-small")
        ax2.scatter(
            cr_times,
            excited_avg[1, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax2.plot(cr_times, _avg_Y(cr_times, *excited_params), lw=2.0, color=colors[1])
        ax3.scatter(
            cr_times,
            excited_avg[2, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax3.plot(cr_times, _avg_Z(cr_times, *excited_params), lw=2.0, color=colors[1])

    if not is_single_qubit_tomo:
        coeffs, errors = get_interation_rates_MHz(
            ground_params, excited_params, ground_cov, excited_cov
        )
        ax3.text(
            cr_times[-1] / 2,
            -2.55,
            "ZX = %.3f (%2.f) MHz   ZY = %.3f (%2.f) MHz   ZZ = %.3f (%2.f) MHz"
            % (
                coeffs[1][0],
                errors[1][0] * 1000,
                coeffs[1][1],
                errors[1][1] * 1000,
                coeffs[1][2],
                errors[1][2] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )
        ax3.text(
            cr_times[-1] / 2,
            -2.9,
            "IX = %.3f (%2.f) MHz   IY = %.3f (%2.f) MHz   IZ = %.3f (%2.f) MHz"
            % (
                coeffs[0][0],
                errors[0][0] * 1000,
                coeffs[0][1],
                errors[0][1] * 1000,
                coeffs[0][2],
                errors[0][2] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )
    else:
        coeffs = ground_params / 2 / pi
        errors = np.diagonal(ground_cov) ** 0.5 / 2 / pi
        ax3.text(
            cr_times[-1] / 2,
            -2.55,
            "X = %.3f (%2.f) MHz   Y = %.3f (%2.f) MHz   Z = %.3f (%2.f) MHz"
            % (
                coeffs[1],
                errors[1] * 1000,
                coeffs[2],
                errors[2] * 1000,
                coeffs[0],
                errors[0] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )


# %% Iterative calibration
def _compute_drive_scale(
    coeff_dict_1, coeff_dict_2, ix_params, prob_ix_strength, conjugate_pulse
):
    vau1 = coeff_dict_1["IX"] + 1.0j * coeff_dict_1["IY"]
    vau2 = coeff_dict_2["IX"] + 1.0j * coeff_dict_2["IY"]
    A1 = ix_params["amp"] * np.exp(1.0j * ix_params["angle"])
    A2 = (ix_params["amp"] + prob_ix_strength) * np.exp(1.0j * ix_params["angle"])
    if conjugate_pulse:
        return (vau2 - vau1) / np.conjugate(A2 - A1)
    else:
        return (vau2 - vau1) / (A2 - A1)


def _angle(c):
    """
    User arctan to calculate the angle such that -1 won't be transfered to the angle parameters, in this way, a negative ZX will remains the same, not transfered to a positive ZX.

    The output range is (-pi/2, pi/2)
    """
    if c.real == 0.0:
        return np.pi / 2
    return np.arctan(c.imag / c.real)


def update_pulse_params(
    coeff_dict_1,
    coeff_dict_2,
    cr_params,
    ix_params,
    prob_ix_strength,
    target_IX_strength=None,
    backend_name=None,
    real_only=False,
):
    """
    Update pulse parameters for CR and IX gates based on tomography results. Refer to arxiv 2303.01427 for the derivation.

    Args:
        coeff_dict_1 (dict): Coefficients from the first tomography job.
        coeff_dict_2 (dict): Coefficients from the second tomography job.
        cr_params (dict): Parameters for the CR gate pulse.
        ix_params (dict): Parameters for the IX gate pulse.
        prob_ix_strength (float): The strength of the IX gate pulse.
        target_IX_strength (float, optional): Target angle for IX gate. Default is 0.
        backend_name (str):
        real_only: Update only the real part of the drive.

    Returns:
        tuple: Updated CR parameters, Updated IX parameters, None (for compatibility with CR-only calibration).
    """
    cr_params = cr_params.copy()
    ix_params = ix_params.copy()
    if backend_name == "DynamicsBackend":
        sign = -1  # dynamics end has a different convension
    else:
        sign = 1
    if "ZX" in coeff_dict_1:
        phi0 = _angle(coeff_dict_1["ZX"] + 1.0j * coeff_dict_1["ZY"])
    else:
        # No CR tomography, only single target qubit tomography.
        phi0 = 0.0
    cr_params["angle"] -= sign * phi0

    drive_scale = _compute_drive_scale(
        coeff_dict_1, coeff_dict_2, ix_params, prob_ix_strength, backend_name
    )
    logger.info(
        "Estimated drive scale: \n"
        f"{drive_scale/1000}" + "\n"
        f"{drive_scale/np.exp(1.j*_angle(drive_scale))/1000}" + "\n"
        f"{_angle(drive_scale)}"
    )

    # Update the IX drive strength and phase
    vau1 = coeff_dict_1["IX"] + 1.0j * coeff_dict_1["IY"]
    A1 = ix_params["amp"] * np.exp(sign * 1.0j * ix_params["angle"])
    new_drive = (target_IX_strength - vau1) / drive_scale + A1
    if real_only:
        ix_params["amp"] = np.real(new_drive)
    else:
        new_angle = _angle(new_drive)
        ix_params["amp"] = np.real(new_drive / np.exp(1.0j * new_angle))
        ix_params["angle"] = sign * (new_angle - phi0)
    if "ZX" in coeff_dict_1:
        return (
            cr_params,
            ix_params,
            None,
            np.real(drive_scale / np.exp(1.0j * _angle(drive_scale)) / 1000),
        )
    else:
        return (
            cr_params,
            None,
            ix_params,
            np.real(drive_scale / np.exp(1.0j * _angle(drive_scale)) / 1000),
        )


def _update_frequency_offset(old_calibration_data, mode, backend_name):
    """
    This should not be used separatly because applying more than once will lead to the wrong offset.
    """
    new_calibration_data = deepcopy(old_calibration_data)
    if mode == "CR":
        coeffs_dict = old_calibration_data["coeffs"]
        frequency_offset_key = "frequency_offset"
    else:
        coeffs_dict = old_calibration_data["x_gate_coeffs"]
        frequency_offset_key = "x_gate_frequency_offset"

    if backend_name == "DynamicsBackend":
        correction = coeffs_dict["IZ"] * 1.0e6
    else:
        correction = -0.5 * coeffs_dict["IZ"] * 1.0e6
    new_calibration_data[frequency_offset_key] = (
        old_calibration_data.get(frequency_offset_key, 0.0) + correction
    )
    if np.abs(coeffs_dict["IZ"]) > 2.0:
        logger.warning(
            "Frequency offset larger than 2MHz, update is not applied. Please check if the fit is accurate."
        )
        return old_calibration_data
    logger.info(
        f"Frequency offset is updated to {new_calibration_data[frequency_offset_key]} Hz"
    )
    return new_calibration_data


def iterative_cr_pulse_calibration(
    qubits,
    backend,
    cr_times,
    session,
    gate_name,
    initial_calibration_data=None,
    verbose=False,
    threshold_MHz=0.015,
    restart=False,
    rerun_last_calibration=True,
    max_repeat=4,
    shots=None,
    mode="CR",
    IX_ZX_ratio=None,
    save_result=True,
    control_states=None,
):
    """
    Iteratively calibrates CR pulses on the given qubits to remove the IX, ZY, IY terms. The result is saved in the carlibtaion data file and can be accessed via `read_calibration_data`.

    Args:
        qubits (list): Qubits involved in the calibration.
        backend: Quantum backend.
        cr_times (list): CR gate times for tomography.
        session: Qiskit runtime session.
        gate_name (str): Name of the CR gate. This will be used to identify the calibration data when retrieving the information.
        initial_calibration_data (dict, optional): Initial parameters for calibration.
        verbose (bool, optional): Whether to display detailed logging information. Default is False.
        threshold_MHz (float, optional): Error threshold for calibration. Default is 0.015 GHz.
        restart (bool, optional): Whether to restart calibration from scratch. Default is False.
        rerun_last_calibration (bool, optional): Whether to rerun the last calibration. Default is True.
        max_repeat (int, optional): Maximum number of calibration repetitions. Default is 4.
        shots (int, optional): Number of shots for each tomography job. Default is 1024.
        mode (str, optional): Calibration mode, "CR" or "IX-pi". Default is "CR". The CR mode is used to measure the ZX and ZZ strength. It only updates the phase of CR drive and the IX drive but not IY. The "IX-pi" mode updates the target drive for a CNOT gate.
    """
    if control_states is None:
        if mode == "CR":
            control_states = (0, 1)
        elif mode == "IX-pi":
            control_states = (1,)
        else:
            ValueError("Mode must be either 'CR' or 'IX-pi/2'.")
    if not restart:
        try:  # Load existing calibration
            logger.info("Loading existing calibration data...")
            qubit_calibration_data = read_calibration_data(backend, gate_name, (qubits))
            qubit_calibration_data = deepcopy(qubit_calibration_data)
            if mode == "IX-pi":
                qubit_calibration_data["x_gate_ix_params"]

        except KeyError:  #
            restart = True  # we need to overwrite rerun_last_calibration=True
            logger.warning(
                f"Failed to find the calibration data for the gate{backend.name, gate_name, (qubits)}. Restarting from scratch."
            )
    if restart:
        if not rerun_last_calibration:
            logger.warning(
                f"Last calibration job for {gate_name} not found or not used. Starting from scratch."
            )
            rerun_last_calibration = True
        if mode == "CR":
            if initial_calibration_data is None:
                raise ValueError(
                    "Starting calibration from scratch, but initial parameters are not provided."
                )
            if (
                "cr_params" not in initial_calibration_data
                or "ix_params" not in initial_calibration_data
            ):
                raise ValueError(
                    "The initial pulse parameters for the CR and target drive must be provided."
                )
            qubit_calibration_data = initial_calibration_data
        else:
            logger.info("Loading existing calibration data...")
            qubit_calibration_data = read_calibration_data(backend, gate_name, (qubits))
            qubit_calibration_data = deepcopy(qubit_calibration_data)
            qubit_calibration_data["x_gate_ix_params"] = qubit_calibration_data[
                "ix_params"
            ].copy()
            qubit_calibration_data["x_gate_ix_params"]["amp"] = 0.0
            qubit_calibration_data["x_gate_ix_params"]["angle"] = 0.0
            if "beta" in qubit_calibration_data["x_gate_ix_params"]:
                del qubit_calibration_data["x_gate_ix_params"]["beta"]
            qubit_calibration_data["x_gate_frequency_offset"] = qubit_calibration_data[
                "frequency_offset"
            ]
    shots = 512 if shots is None else shots

    if mode == "CR" and IX_ZX_ratio is None:
        IX_ZX_ratio = 0.0
    elif mode == "IX-pi" and IX_ZX_ratio is None:
        IX_ZX_ratio = -2
    try:
        target_IX_strength = qubit_calibration_data["coeffs"]["ZX"] * IX_ZX_ratio
    except:
        target_IX_strength = 0.0
    logger.info(f"Target IX / ZX ratio: {IX_ZX_ratio}")

    def _get_error(coeff_dict, mode, target_IX):
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, coeff_dict["IY"], coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        return max_error

    def _error_smaller_than(coeff_dict, threshold_MHz, mode, target_IX=None):
        """
        Compare the measured coupling strength and check if the error terms are smaller than the threshold.

        Args:
            coeff_dict (dict): Coefficients from tomography job.
            threshold_MHz (float): Error threshold for calibration.
            mode (str): Calibration mode, "CR" or "IX-pi/2".
            target_IX (float, optional): Target IX angle. Default is None.

        Returns:
            bool: True if error is smaller than threshold, False otherwise.
        """
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, coeff_dict["IY"], coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        error_type = ["IX", "IY", "ZY"][np.argmax(np.abs(error))]
        logger.info(f"Remaining dominant error: {error_type}: {max_error} MHz" + "\n")
        return max_error < threshold_MHz

    def _step_cr(qubit_calibration_data, n):
        """
        Submit two jobs, one with the given pulse parameter. If the calibration is not finished, submit another one with a shifted amplitude for the target drive. This will be used to calculate a new set of pulse parameters and returned.

        Args:
            qubit_calibration_data (dict): Calibration data.
            prob_ix_strength (float): Strength of the IX gate pulse.
            n (int): Calibration iteration number.

        Returns:
            tuple: Updated qubit_calibration_data, Calibration success flag.
        """
        cr_params = qubit_calibration_data["cr_params"]
        ix_params = qubit_calibration_data["ix_params"]
        x_gate_ix_params = None
        frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, we can skip the first experiment that just rerun the tomography for the same pulse parameters.
            tomo_id1 = qubit_calibration_data["calibration_job_id"]
        else:
            # Run tomography experiment for the given parameters.
            tomo_id1 = send_cr_tomography_job(
                (qubits),
                backend,
                cr_params,
                ix_params,
                cr_times,
                x_gate_ix_params=x_gate_ix_params,
                frequency_offset=frequency_offset,
                blocking=True,
                session=session,
                shots=shots,
                control_states=control_states,
            )
        coeff_dict_1 = process_zx_tomo_data(tomo_id1, show_plot=verbose)

        target_IX_strength = (
            IX_ZX_ratio
            * np.sign(coeff_dict_1["ZX"])
            * np.sqrt(coeff_dict_1["ZX"] ** 2 + coeff_dict_1["ZY"] ** 2)
        )
        if verbose:
            logger.info("Tomography results:\n" + str(coeff_dict_1) + "\n")

        qubit_calibration_data.update(
            {
                "calibration_job_id": tomo_id1,
                "coeffs": coeff_dict_1,
            }
        )

        # Interrupt the process if the calibration is successful or maximal repeat number is reached.
        if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
            if not (not rerun_last_calibration and n == 1):
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, mode, backend.name
                )
        if _error_smaller_than(coeff_dict_1, threshold_MHz, mode, target_IX_strength):
            logger.info("Successfully calibrated.")
            return qubit_calibration_data, True
        if n > max_repeat:
            logger.info(
                f"Maximum repeat number {max_repeat} reached, calibration terminates."
            )
            return qubit_calibration_data, True

        # If not completed, send another job with shifted IX drive amplitude.
        # Remark: There is a strange observation that the Omega_GHz_amp_ratio measured here is not the same as the one estimated from single-qubit gate duration. There is a
        Omega_GHz_amp_ratio = qubit_calibration_data.get(
            "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[1], 1)
        )
        logger.info(f"Omega[GHz]/amp: {Omega_GHz_amp_ratio}")
        Omega_GHz_amp_ratio = np.real(Omega_GHz_amp_ratio)
        prob_ix_strength_MHz = target_IX_strength - coeff_dict_1["IX"]
        if np.abs(prob_ix_strength_MHz) > 0.1:  # Minimum 0.1 MHz
            prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / Omega_GHz_amp_ratio
            logger.info(f"Probe amp shift [MHz]: {prob_ix_strength_MHz} MHz")
        else:
            prob_ix_strength = (
                np.sign(prob_ix_strength_MHz) * 0.1e-3 / Omega_GHz_amp_ratio
            )
            logger.info("Probe amp shift [MHz]: 0.1 MHz")
        logger.info(f"Probe amp shift (amp): {prob_ix_strength}")

        tomo_id2 = shifted_parameter_cr_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            prob_ix_strength,
            x_gate_ix_params=x_gate_ix_params,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=shots,
            control_states=control_states,
        )
        coeff_dict_2 = process_zx_tomo_data(tomo_id2, show_plot=verbose)
        if verbose:
            logger.info(coeff_dict_2)
        # Compute the new parameters.
        (
            cr_params,
            updated_ix_params,
            updated_x_gate_ix_params,
            omega_amp_ratio,
        ) = update_pulse_params(
            coeff_dict_1,
            coeff_dict_2,
            cr_params,
            ix_params,
            prob_ix_strength,
            target_IX_strength=target_IX_strength,
            backend_name=backend.name,
            # only update the real part, imaginary part can be
            # unstable for small pulse ampliutude.
            real_only=True,
        )
        # This should not be added before the second experiment because it should only have a different IX drive amplitude.
        qubit_calibration_data.update(
            {
                "cr_params": cr_params,
                "ix_params": updated_ix_params,
                "_omega_amp_ratio": np.real(omega_amp_ratio),
            }
        )
        return qubit_calibration_data, False

    def _step_ix(qubit_calibration_data, n):
        """
        Submit two jobs, one with the given pulse parameter. If the calibration is not finished, submit another one with a shifted amplitude for the target drive. This will be used to calculate a new set of pulse parameters and returned.

        Args:
            qubit_calibration_data (dict): Calibration data.
            prob_ix_strength (float): Strength of the IX gate pulse.
            n (int): Calibration iteration number.

        Returns:
            tuple: Updated qubit_calibration_data, Calibration success flag.
        """
        if len(control_states) == 2:
            process_data_fun = process_zx_tomo_data
        else:
            process_data_fun = process_single_qubit_tomo_data
        cr_params = qubit_calibration_data["cr_params"]
        ix_params = qubit_calibration_data["ix_params"]
        x_gate_ix_params = qubit_calibration_data["x_gate_ix_params"]
        frequency_offset = qubit_calibration_data.get("x_gate_frequency_offset", 0.0)
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, we can skip the first experiment that just rerun the tomography for the same pulse parameters.
            tomo_id1 = qubit_calibration_data["x_gate_calibration_job_id"]
        else:
            # Run tomography experiment for the given parameters.
            tomo_id1 = send_cr_tomography_job(
                (qubits),
                backend,
                cr_params,
                ix_params,
                cr_times,
                x_gate_ix_params=x_gate_ix_params,
                frequency_offset=frequency_offset,
                blocking=True,
                session=session,
                shots=shots,
                control_states=control_states,
            )
        coeff_dict_1 = process_data_fun(tomo_id1, show_plot=verbose)

        if verbose:
            logger.info("Tomography results:\n" + str(coeff_dict_1) + "\n")

        qubit_calibration_data.update(
            {
                "x_gate_calibration_job_id": tomo_id1,
                "x_gate_coeffs": coeff_dict_1,
            }
        )
        # Interrupt the process if the calibration is successful or maximal repeat number is reached.
        if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
            if not (not rerun_last_calibration and n == 1):
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, mode, backend.name
                )
        if _error_smaller_than(coeff_dict_1, threshold_MHz, mode, target_IX_strength):
            logger.info("Successfully calibrated.")
            return qubit_calibration_data, True
        if n > max_repeat:
            logger.info(
                f"Maximum repeat number {max_repeat} reached, calibration terminates."
            )
            return qubit_calibration_data, True

        # If not completed, send another job with shifted IX drive amplitude.
        Omega_GHz_amp_ratio = qubit_calibration_data.get(
            "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[1], 1)
        )
        Omega_GHz_amp_ratio = np.real(Omega_GHz_amp_ratio)
        logger.info(f"Omega[GHz]/amp: {Omega_GHz_amp_ratio}")
        prob_ix_strength_MHz = target_IX_strength - coeff_dict_1["IX"]
        if np.abs(prob_ix_strength_MHz) > 0.1:  # Minimum 0.1 MHz
            prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / Omega_GHz_amp_ratio
            logger.info(f"Probe amp shift [MHz]: {prob_ix_strength_MHz} MHz")
        else:
            prob_ix_strength = (
                np.sign(prob_ix_strength_MHz) * 0.1e-3 / Omega_GHz_amp_ratio
            )
            logger.info("Probe amp shift [MHz]: 0.1 MHz")
        logger.info(f"Probe amp shift (amp): {prob_ix_strength}")
        tomo_id2 = shifted_parameter_cr_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            prob_ix_strength,
            x_gate_ix_params=x_gate_ix_params,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=shots,
            control_states=control_states,
            mode=mode,
        )
        coeff_dict_2 = process_data_fun(tomo_id2, show_plot=verbose)
        if verbose:
            logger.info(coeff_dict_2)
        # Compute the new parameters.
        cr_params, _, updated_x_gate_ix_params, omega_amp_ratio = update_pulse_params(
            coeff_dict_1,
            coeff_dict_2,
            cr_params,
            x_gate_ix_params,
            prob_ix_strength,
            target_IX_strength=target_IX_strength,
            backend_name=backend.name,
        )

        qubit_calibration_data.update(
            {
                "x_gate_ix_params": updated_x_gate_ix_params,
                "_omega_amp_ratio": np.real(omega_amp_ratio),
            }
        )
        return qubit_calibration_data, False

    succeed = False
    n = 1
    error = np.inf
    while (
        not succeed and n <= max_repeat + 1
    ):  # +1 because we need one last run for the calibration data.
        logger.info(f"\n\nCR calibration round {n}: ")
        if mode == "CR":
            qubit_calibration_data, succeed = _step_cr(qubit_calibration_data, n)
        else:
            qubit_calibration_data, succeed = _step_ix(qubit_calibration_data, n)
        target_IX_strength = qubit_calibration_data["coeffs"]["ZX"] * IX_ZX_ratio
        new_error = _get_error(
            qubit_calibration_data["coeffs"], mode, target_IX_strength
        )
        if save_result and new_error < error:
            save_calibration_data(backend, gate_name, qubits, qubit_calibration_data)
            logger.info("CR calibration data saved.")
        n += 1
        shots = 2 * shots if shots < 2048 else shots
    if not succeed:
        logger.warn(f"CR calibration failed after {n} round.")


def iy_drag_calibration(
    qubits,
    backend,
    gate_name,
    cr_times,
    session,
    verbose=False,
    threshold_MHz=0.015,
    delta_beta=None,
    shots=1024,
):
    """Calibrate the IY-DRAG pulse for the qubits and a precalibrated CR pulse. It samples 3 "beta" value in the "ix_params" and perform an linear fit to obtain the correct IY-DRAG coefficient "beta" that zeros the ZZ interaction.

    Args:
        qubits (Tuple): Tuple containing the qubits involved in the gate.
        backend: The quantum backend.
        gate_name (str): Name of the gate for calibration. The pre calibrated CR pulse will be read from the database.
        cr_times (List): List of control pulse durations for CR experiments.
        session: Qiskit runtime session.
        verbose (bool, optional): Whether to display additional information. Defaults to False.
        threshold_MHz (float, optional): The error threshold for calibration in MHz. Defaults to 0.015.
        delta_beta (float, optional): The step size for beta parameter calibration. Defaults to None.
    """
    logger.info("\n" + f"Calibrating the IY-DRAG pulse for {qubits}-{gate_name}.")
    qubit_calibration_data = read_calibration_data(backend, gate_name, qubits)
    cr_params = qubit_calibration_data["cr_params"]
    ix_params = qubit_calibration_data["ix_params"]
    frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)

    # Sample three different IY strength.
    old_beta = ix_params.get("beta", 0.0)
    if "drag_type" in ix_params:
        ix_params["drag_type"] = "01"
        default_delta_beta = 2.0
    elif "drag_type" not in ix_params:
        default_delta_beta = 100.0
    else:
        raise ValueError("Unknown drag type.")
    delta_beta = (
        old_beta
        if (old_beta > default_delta_beta and delta_beta is None)
        else delta_beta
    )
    delta_beta = default_delta_beta if delta_beta is None else delta_beta
    beta_list = np.array([0.0, -delta_beta, delta_beta]) + old_beta

    ZZ_coeff_list = []
    for _, beta in enumerate(beta_list):
        if np.abs(beta - old_beta) < 1.0e-6:
            _shots = shots * 2
        else:
            _shots = shots
        ix_params["beta"] = beta
        job_id = send_cr_tomography_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=_shots,
        )
        coeff_dict = process_zx_tomo_data(job_id, show_plot=verbose)
        ZZ_coeff_list.append(coeff_dict["ZZ"])
        if abs(beta - old_beta) < 1.0e-5 and abs(coeff_dict["ZZ"]) < threshold_MHz:
            logger.info(
                f"ZZ error {round(coeff_dict['ZZ'], 3)} MHz, no need for further calibration."
            )
            qubit_calibration_data.update(
                {
                    "calibration_job_id": job_id,
                    "coeffs": coeff_dict,
                }
            )
            save_calibration_data(backend, gate_name, qubits, qubit_calibration_data)
            if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, "CR", backend.name
                )
            return

    logger.info(f"ZZ sampling measurements complete : {ZZ_coeff_list}." + "\n")

    # Fit a linear curve.
    fun = lambda x, a, b: a * x + b
    par, _ = curve_fit(fun, beta_list, ZZ_coeff_list)
    calibrated_beta = -par[1] / par[0]
    logger.info(f"Calibrated IY beta: {calibrated_beta}" + "\n")

    if verbose:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
        plt.scatter(beta_list, ZZ_coeff_list)
        x_line = np.linspace(min(beta_list), max(beta_list))
        y_line = fun(x_line, *par)
        plt.plot(x_line, y_line)
        plt.xlabel("beta")
        plt.ylabel("ZZ [MHz]")
        plt.show()

    # Perform a final tomography measurement.
    ix_params["beta"] = calibrated_beta
    job_id = send_cr_tomography_job(
        qubits,
        backend,
        cr_params,
        ix_params,
        cr_times,
        frequency_offset=frequency_offset,
        blocking=True,
        session=session,
        shots=shots * 2,
    )

    # Compute the interaction strength and save the calibration data.
    coeff_dict = process_zx_tomo_data(job_id, show_plot=verbose)
    logger.info(f"Updated coupling strength: {coeff_dict}")
    qubit_calibration_data.update(
        {
            "calibration_job_id": job_id,
            "coeffs": coeff_dict,
            "ix_params": ix_params,
        }
    )
    if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
        qubit_calibration_data = _update_frequency_offset(
            qubit_calibration_data, "CR", backend.name
        )

    save_calibration_data(backend, gate_name, qubits, qubit_calibration_data)
    logger.info(f"IY-DRAG calibration complete, new calibration data saved.")
