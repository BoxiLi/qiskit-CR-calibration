"""
Generation of echoed and direct CNOT gate based on calibrated CR pulse data.
"""
from copy import deepcopy

import numpy as np
from numpy import pi
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import qiskit
from qiskit import pulse, circuit, schedule, transpile
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.transpiler import InstructionProperties

from .job_util import (
    load_job_data,
    save_job_data,
    read_calibration_data,
    save_calibration_data,
)
from .cr_pulse import (
    process_zx_tomo_data,
    get_cr_schedule,
    _get_normalized_cr_tomography_data,
)

# Logger setup after importing logging
import logging

logger = logging.getLogger(__name__)

# %% Esitmate the gate time for a CNOT gate


def fit_function(
    func,
    xdata,
    ydata,
    init_params,
    show_plot=False,
):
    """
    Fit a given function using ``scipy.optimize.curve_fit``.

    Args:
        func (Callable): The function to fit the data. The first arguments is the input x data.
        xdata (ArrayLike): X axis.
        ydata (ArrayLike): Y axis.
        init_params (ArrayLike): Initial parameter for x data
        show_plot (bool, optional): If True, plot the fitted function and the given data. Defaults to False.

    Returns:
        list: The fitted parameters.
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    if func is None:
        func = lambda x, A, B, f, phi: (A * np.cos(2 * pi * f * x - phi) + B)

    try:
        fitparams, conv = curve_fit(
            func,
            xdata,
            ydata,
            p0=init_params,
        )
    except RuntimeError as e:
        warnings.warn("RuntimeError in the optimization")
        fitparams = init_params

    xline = np.linspace(xdata[0], xdata[-1], 100)
    yline = func(xline, *fitparams)

    diff = np.sum(np.abs(ydata - func(xdata, *fitparams))) / len(ydata)
    if diff > 0.1:
        logger.warning(
            "The fitting error is too large. The calibrated value could be wrong. Please double check the calibration data."
        )

    if show_plot:
        fig = plt.figure()
        plt.plot(xdata, ydata, "x", c="k")
        plt.plot(xline, yline)
        plt.ylabel("Signal [a.u.]")
        plt.ylim([-1.1, 1.1])
    return fitparams


def _find_peak_index(ydata, xdata, height, positive=False):
    """Find the index of the first (close to time 0) peak in the given data.

    Args:
        ydata (np.ndarray): The y-values of the data.
        xdata (np.ndarray): The x-values of the data.
        height (float): The minimum height of peaks to be detected.
        positive (bool, optional): If True, find positive peaks; otherwise, find negative peaks. Defaults to False.

    Returns:
        int: The index of the estimated peak.

    Raises:
        Warning: If no peaks are found in the data. It returns then the maximal index.
    """
    extreme_indices = find_peaks(ydata, height=height)[0]
    if positive:
        # Positive maximum
        extreme_indices = np.array([i for i in extreme_indices if xdata[i] > 0])
    if extreme_indices.size > 0:
        # Closest one to 0.
        estimated_index = extreme_indices[
            np.argmax(-np.abs(xdata[extreme_indices] - 0.0))
        ]
    else:
        logger.warning("Peak not found! Consider increase the drive strength.")
        estimated_index = np.argmax(ydata)
    return estimated_index


def _find_closest_extreme_point(f, phi, estimated_x_value):
    """Find the closest extreme point to the estimated maximum or minimum.

    Args:
        f (float): Frequency of the sinusoidal function.
        phi (float): Phase of the sinusoidal function.
        estimated_x_value (float): The estimated x-value of the maximum or minimum.

    Returns:
        float: The phase of the closest extreme point.
    """
    extreme_points = (phi + np.array([-np.pi, 0.0, np.pi])) / f / 2 / pi
    _diff_estimation = -np.abs(extreme_points - estimated_x_value)
    return extreme_points[np.argmax(_diff_estimation)]


def estimate_cr_width(job_id, angle="90"):
    """
    Estimate the duration of the CR pulse from the calibrated tomography experiment data.

    This function takes the job ID of a tomography experiment for a calibrated CR pulse. It computes the duration of the CR pulse based on the observed signal. The function returns the estimated duration of the CR pulse as an integer that satisfies the qiskit-pulse requirement.

    Args:
        job_id (str): The ID of the calibrated tomography experiment job.
        angle (str, optional): The angle of the CR gate, either "90" or "45". Defaults to "90".

    Returns:
        int: The estimated duration of the CR pulse.

    Note:
        Due to the truncation of the pulse time to a mulitple of 16, there is additional esitmation error introduced. This could be compensated for by adjusting the pulse amplitude. Here we omit this as the error is small.
    """

    def _helper(cr_times, signal_z):
        # Rough estimation of the first positive maximum
        estimated_index = _find_peak_index(
            -signal_z, cr_times, height=0.75, positive=True
        )

        # Curve fit

        A, B, f, phi = fit_function(
            lambda x, A, B, f, phi: (A * np.cos(2 * pi * f * x - phi) + B),
            cr_times,
            signal_z,
            init_params=[-1, 0, 1 / 2 / (cr_times[estimated_index]), 0.0],
            show_plot=False,
        )

        fitted_first_min = _find_closest_extreme_point(
            f, phi, cr_times[estimated_index]
        )
        period = 1 / f
        if angle == "90":
            estimated_result = fitted_first_min - period / 4
        elif angle == "45":
            estimated_result = fitted_first_min - period / 8 * 3
        else:
            raise ValueError("Unknown angle")
        return estimated_result

    cr_times, splitted_data = _get_normalized_cr_tomography_data(job_id)
    signal_z0 = splitted_data[4]
    signal_z1 = splitted_data[5]
    est1 = _helper(cr_times, signal_z0)
    est2 = _helper(cr_times, signal_z1)
    if np.abs(est2 - est1) > 100:
        logger.warning(
            "The estimated CR pulse time is different for control qubit in state 0 and 1. This indicate that the ZX interaction strength is not well calibrated."
        )
    estimated_result = (est1 + est2) / 2
    estimated_result = (estimated_result + 8) // 16 * 16
    return int(estimated_result)


# %%  ################# Calibrate a CNOT gate


def _get_ZX_sign(job_id):
    coupling_strength = process_zx_tomo_data(job_id, show_plot=False)
    return np.sign(coupling_strength["ZX"])


def create_cr45_pairs(
    backend, qubits, cr_params, ix_params, ZX_sign, cr45_duration, frequency_offset=0.0
):
    """
    Generate CR pulse schedule for a 45 degree ZX rotation.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        cr_params (dict): Dictionary containing the base CR pulse parameters.
        ix_params (dict): Dictionary containing the IX pulse parameters.
        ZX_sign (int): Sign of the ZX interaction (1 or -1).
        cr45_duration (int): Duration of the CR pulses for a 45 degree rotation.
        frequency_offset (float, optional): Frequency offset. Defaults to 0.0.

    Returns:
        list: Pulse schedule for a CR45m and CR45p pulse schedule.
    """
    cr_params = deepcopy(cr_params)
    ix_params = deepcopy(ix_params)
    cr_params["duration"] = cr45_duration
    ix_params["duration"] = cr45_duration
    cr_sched1 = get_cr_schedule(
        qubits,
        backend,
        cr_params=cr_params,
        ix_params=ix_params,
        frequency_offset=frequency_offset,
    )
    cr_params["angle"] += pi
    ix_params["angle"] += pi
    cr_sched2 = get_cr_schedule(
        qubits,
        backend,
        cr_params=cr_params,
        ix_params=ix_params,
        frequency_offset=frequency_offset,
    )
    if ZX_sign == 1:
        return cr_sched2, cr_sched1
    else:
        return cr_sched1, cr_sched2


def create_echoed_cnot_schedule(
    backend, qubits, calibration_data, reverse_direction=False
):
    """
    Generate a schedule for a echod CNOT gate based on CR pulse.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        cr_params (dict): Dictionary containing the base CR pulse parameters.
        ix_params (dict): Dictionary containing the IX pulse parameters.
        ZX_sign (int): Sign of the ZX interaction (1 or -1).
        cr45_duration (int): Duration of the CR pulses for a 45 degree rotation.
        cr90_duration (int): Duration of the CR pulses for a 90 degree rotation.
        idle_duration (int): Duration of the idle time between the CR pulses.
        frequency_offset (float, optional): Frequency offset. Defaults to 0.0.

    Returns:
        qsikit.pulse.Schedule: A pulse schedule for a CNOT gate with echo.
    """
    cr_params = calibration_data["cr_params"]
    ix_params = calibration_data["ix_params"]
    calibrated_cr_tomo_id = calibration_data["calibration_job_id"]
    duration = estimate_cr_width(calibrated_cr_tomo_id, angle="45")

    frequency_offset = calibration_data["frequency_offset"]
    ZX_sign = _get_ZX_sign(calibrated_cr_tomo_id)

    cr45m_sched, cr45p_sched = create_cr45_pairs(
        backend,
        qubits,
        cr_params,
        ix_params,
        ZX_sign,
        duration,
        frequency_offset=frequency_offset,
    )
    CR45p = Gate("CR45p", 2, [])
    CR45m = Gate("CR45m", 2, [])
    backend = deepcopy(backend)
    backend.target.add_instruction(
        CR45m,
        {qubits: InstructionProperties(calibration=cr45m_sched)},
        name="CR45m",
    )
    backend.target.add_instruction(
        CR45p,
        {qubits: InstructionProperties(calibration=cr45p_sched)},
        name="CR45p",
    )
    circ = QuantumCircuit(2, 2)
    if reverse_direction:
        circ.h(0)
        circ.h(1)
    circ.sx(1)
    circ.x(0)
    circ.append(CR45p, [0, 1])
    circ.x(0)
    circ.append(CR45m, [0, 1])
    circ.rz(np.pi / 2, 0)
    if reverse_direction:
        circ.h(0)
        circ.h(1)

    try:
        basis_gates = backend.configuration().basis_gates
    except AttributeError:
        basis_gates = ["measure", "sx", "x", "rz"]
    transpile_options = {
        "basis_gates": basis_gates + ["CR45p", "CR45m"],
        "initial_layout": qubits,
        "optimization_level": 1,
    }
    transpiled_circ = transpile(circ, backend=backend, **transpile_options)
    # Transfer Schedule to ScheduleBlock using pulse builder. Otherwise qiskit_ibm_provider rejects it.
    with pulse.build(backend) as cnot_sched:
        qiskit.pulse.builder.call(schedule(transpiled_circ, backend=backend))
    return cnot_sched


def create_cr90m(
    backend,
    qubits,
    cr_params,
    ix_params,
    ZX_sign,
    cr90_duration,
    x_gate_ix_params,
    frequency_offset=0.0,
):
    """
    Generate CR pulse schedule for a 90 degree -ZX rotation.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        cr_params (dict): Dictionary containing the base CR pulse parameters.
        ix_params (dict): Dictionary containing the IX pulse parameters.
        ZX_sign (int): Sign of the ZX interaction (1 or -1).
        cr90_duration (int): Duration of the CR pulse for a 90 degree rotation.
        frequency_offset (float, optional): Frequency offset. Defaults to 0.0.

    Returns:
        qiskit.pulse.Schedule: A pulse schedule for a CR90m pulse.
    """
    cr_params = deepcopy(cr_params)
    ix_params = deepcopy(ix_params)
    if ZX_sign == 1:
        cr_params["angle"] += pi
        ix_params["angle"] += pi
    cr_params["duration"] = cr90_duration
    ix_params["duration"] = cr90_duration
    if x_gate_ix_params is not None:
        x_gate_ix_params["duration"] = cr90_duration
        if ZX_sign == 1:
            x_gate_ix_params["angle"] += pi
    return get_cr_schedule(
        qubits,
        backend,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
    )


def create_direct_cnot_schedule(
    backend, qubits, calibration_data, with_separate_x=False, reverse_direction=False
):
    """
    Generate a schedule for a direct CNOT pulse.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        calibration_data (dict): Dictionary containing calibration data.
        with_separate_x (bool, optional): If True, includes a separate X gate before the CR90m. Defaults to False.

    Returns:
        qiskit.pulse.Schedule: A pulse schedule for a direct CNOT pulse.
    """
    cr_params = calibration_data["cr_params"]
    ix_params = calibration_data["ix_params"]
    if with_separate_x:
        x_gate_ix_params = None
    else:
        x_gate_ix_params = calibration_data["x_gate_ix_params"]
    calibrated_cr_tomo_id = calibration_data["calibration_job_id"]
    rz0_correction = calibration_data["rz0_correction"]
    frequency_offset = calibration_data["x_gate_frequency_offset"]
    ZX_sign = _get_ZX_sign(calibrated_cr_tomo_id)
    duration = estimate_cr_width(calibrated_cr_tomo_id, angle="90")

    cr90m_sched = create_cr90m(
        backend,
        qubits,
        cr_params,
        ix_params,
        ZX_sign,
        duration,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
    )
    CR90m = Gate("CR90m", 2, [])
    backend = deepcopy(backend)
    backend.target.add_instruction(
        CR90m,
        {qubits: InstructionProperties(calibration=cr90m_sched)},
        name="CR90m",
    )

    circ = QuantumCircuit(2)
    if reverse_direction:
        circ.h(0)
        circ.h(1)
    if with_separate_x:
        circ.rx(pi / 2, 1)
    circ.append(CR90m, [0, 1])
    circ.rz(pi / 2, 0)
    circ.rz(rz0_correction, 0)
    if reverse_direction:
        circ.h(0)
        circ.h(1)

    try:
        basis_gates = backend.configuration().basis_gates
    except AttributeError:
        basis_gates = ["measure", "sx", "x", "rz"]
    transpile_options = {
        "basis_gates": basis_gates + ["CR90m"],
        "initial_layout": qubits,
        "optimization_level": 1,
    }
    transpiled_circ = transpile(circ, backend=backend, **transpile_options)
    # Transfer Schedule to ScheduleBlock using pulse builder. Otherwise qiskit_ibm_provider rejects it.
    with pulse.build(backend) as cnot_sched:
        qiskit.pulse.builder.call(schedule(transpiled_circ, backend=backend))
    return cnot_sched


# %% Calibration of the RZ0 phase correction
def get_rz0_amplification_circuits(num_gates_list, phase_list):
    """
    Generate a list of circuits for calibrating RZ0 phase correction by amplifying the effect.

    Args:
        num_gates_list (list): List of integers specifying the number of CNOT gates in each circuit.
        phase_list (list): List of phase values for the RZ gate.

    Returns:
        list: List of circuits with varied numbers of CNOT gates and RZ phases.
    """
    circ_list = []
    phase = circuit.Parameter("phase")
    custom_cnot = Gate("custom_cnot", 2, [])
    for n in num_gates_list:
        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.barrier()
        for _ in range(n):
            circ.append(custom_cnot, [0, 1])
            circ.rz(phase, 0)
            circ.barrier()
        circ.h(0)
        circ.measure(0, 0)
        circ.measure(1, 1)
        circ_list += [circ.bind_parameters({phase: p}) for p in phase_list]
    return circ_list


def get_rz0_pi_calibration_circuits(phase_list):
    """
    Generate a list of circuits for calibrating RZ0 phase correction with pi rotations. One measures 00 if CX has correct ZI phase, if the phase is pi, one measures 11.

    Args:
        phase_list (list): List of phase values for the RZ gate.

    Returns:
        list: List of circuits for differnet phase.
    """
    phase = circuit.Parameter("phase")
    custom_cnot = Gate("custom_cnot", 2, [])
    circ = QuantumCircuit(2, 2)
    circ.h(0)
    # CX with a phase need to be calibrated
    circ.barrier()
    circ.append(custom_cnot, [0, 1])
    circ.rz(phase, 0)
    circ.barrier()

    circ.rx(pi / 2, 0)
    # CX with a phase need to be calibrated
    circ.barrier()
    circ.append(custom_cnot, [0, 1])
    circ.rz(phase, 0)
    circ.barrier()

    circ.rx(-pi / 2, 1)
    circ.h(0)
    circ.measure(0, 0)
    circ.measure(1, 1)

    return [circ.bind_parameters({phase: p}) for p in phase_list]


def send_rz0_calibration_job(backend, qubits, phase_list, circ_list, session):
    """
    Send a job for calibrating RZ0 phase correction.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        phase_list (list): List of phase values for the RZ gate.
        circ_list (list): List of QuantumCircuit objects to be executed
        session: A Qiskit runtime Session.

    Returns:
        str: Job ID of the calibration job.
    """
    try:
        basis_gates = backend.configuration().basis_gates
    except AttributeError:
        basis_gates = ["measure", "sx", "x", "rz"]
    transpiled_circ_list = transpile(
        circ_list,
        backend=backend,
        basis_gates=basis_gates + ["custom_cnot"],
        initial_layout=qubits,
        optimization_level=1,
    )

    shots = 1024
    if session is not None:
        job = session.run(
            "circuit-runner",
            inputs={
                "circuits": transpiled_circ_list,
                "skip_transpilation": True,
                "shots": shots,
            },
        )
    else:
        job = backend.run(transpiled_circ_list, shots=shots)
    parameters = {
        "name": "rz0 calibration",
        "backend": backend.name,
        "qubits": qubits,
        "phase_list": phase_list,
        "shots": shots,
    }
    logger.info(
        "ZI calibration ID: "
        + job.job_id()
        + "\n"
        + "\n".join([f"{key}: {val}" for key, val in parameters.items()])
        + "\n"
    )
    save_job_data(job, backend=backend, parameters=parameters)
    return job.job_id()


def rough_rz0_correction_calibration(backend, qubits, gate_name, session):
    """
    Perform a rough calibration of the RZ0 phase phase correction on the control qubit.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        gate_name (str): Name of the gate for calibration.
        session: A Qiskit runtime Session.
    """
    logger.info("Rough calibration of the RZ0 phase correction.")
    calibration_data = read_calibration_data(backend, gate_name, qubits)
    calibration_data = deepcopy(calibration_data)
    calibration_data["rz0_correction"] = 0.0
    custom_cnot_schedule = create_direct_cnot_schedule(
        backend, qubits, calibration_data, with_separate_x=False
    )

    phase_list = np.linspace(-pi, pi, 50)
    max_num_gates = 8
    num_gates_list = range(2, max_num_gates + 1, 2)
    circ_list = get_rz0_amplification_circuits(
        num_gates_list, phase_list
    ) + get_rz0_pi_calibration_circuits(phase_list)

    backend = deepcopy(backend)
    backend.target.add_instruction(
        Gate("custom_cnot", 2, []),
        {qubits: InstructionProperties(calibration=custom_cnot_schedule)},
        name="custom_cnot",
    )

    job_id = send_rz0_calibration_job(backend, qubits, phase_list, circ_list, session)
    logger.info("Job complete. Analyzing data...")
    data = load_job_data(job_id)
    result = data["result"]
    shots = data["parameters"]["shots"]

    prob_list = np.array(
        [result.get_counts(i).get("00", 0) / shots for i in range(len(result.results))]
    )
    prob_list = prob_list.reshape(len(result.results) // 50, 50)
    fig, ax = plt.subplots(1)
    im = ax.imshow(prob_list, vmin=0, vmax=1, aspect=3, cmap="PuBu")
    ax.set_xticklabels([0] + [round(p, 2) for p in phase_list[::10]])
    ax.set_yticks([0, 1, 2, 3, 4], [2, 4, 6, 8, "parity"])
    ax.set_ylabel("Number of CNOT")
    cbar = fig.colorbar(im, spacing="proportional", shrink=0.4)
    cbar.set_label(r"Probability of $|00\rangle$")

    estimated_rz0_correction = phase_list[np.argmax(np.sum(prob_list, axis=0))]
    calibration_data["rz0_correction"] = estimated_rz0_correction
    logger.info(f"Estimated RZ0 correction: {estimated_rz0_correction}\n")
    save_calibration_data(backend, gate_name, qubits, calibration_data)


def fine_rz0_correction_calibration(backend, qubits, gate_name, session):
    """
    Perform a fine-tuning calibration of the phase correction on the control qubit.

    Args:
        backend (Backend): The quantum backend.
        qubits (Tuple): Tuple of qubits (control, target).
        gate_name (str): Name of the gate for calibration.
        session: A Qiskit Quantum Session.
    """
    logger.info("Fine calibration of the RZ0 phase correction.")
    calibration_data = read_calibration_data(backend, gate_name, qubits)
    rz0_correction = calibration_data["rz0_correction"]
    calibration_data = deepcopy(calibration_data)
    calibration_data["rz0_correction"] = 0.0
    custom_cnot_schedule = create_direct_cnot_schedule(
        backend, qubits, calibration_data, with_separate_x=False
    )

    narrow_phase_list = rz0_correction + np.linspace(-pi / 10, pi / 10, 50)
    circ_list = get_rz0_amplification_circuits((6,), narrow_phase_list)

    backend = deepcopy(backend)
    backend.target.add_instruction(
        Gate("custom_cnot", 2, []),
        {qubits: InstructionProperties(calibration=custom_cnot_schedule)},
        name="custom_cnot",
    )

    job_id = send_rz0_calibration_job(
        backend, qubits, narrow_phase_list, circ_list, session
    )
    data = load_job_data(job_id)
    result = data["result"]
    shots = data["parameters"]["shots"]

    def _find_closest_extreme_point(f, phi, estimated_x_value):
        """
        Find the extreme point that is closed to the estimated maximum/minimum.
        """
        extreme_points = (phi + np.linspace(-10 * pi, 10 * pi, 21)) / f / 2 / pi
        _diff_estimation = -np.abs(extreme_points - estimated_x_value)
        return extreme_points[np.argmax(_diff_estimation)]

    prob_list = np.array(
        [result.get_counts(i).get("00", 0) / shots for i in range(len(result.results))]
    )
    data_to_fit = -prob_list
    rought_estimated_min_point = narrow_phase_list[np.argmin(data_to_fit)]

    A, B, f, phi = fit_function(
        lambda x, A, B, f, phi: (A * np.cos(2 * pi * f * x - phi) + B),
        narrow_phase_list,
        data_to_fit,
        init_params=[-1.0, 0, 1.0, rought_estimated_min_point],
        show_plot=True,
    )

    rz0_correction = _find_closest_extreme_point(f, phi, rought_estimated_min_point)
    calibration_data["rz0_correction"] = rz0_correction
    logger.info(f"Fine-tuning RZ0 correction: {rz0_correction}")
    save_calibration_data(backend, gate_name, qubits, calibration_data)
