# qiskit-CR-calibration

This repository includes the procedure to calibration a Cross-Resonance gate on fixed-frequency superconducting qubits using `qiskit-pulse`. The notebook uses `qiskit-dynamics` as a simulation backend, but the code also works with the real hardware on IBM Quantum Platform.
Apart from the standard calibration of the echoed Cross-Resonance, it also includes
- the calculation of the recursive DRAG pulse that suppress transition error on the control Transmon and the IY-DRAG pulse to cancel the ZZ interaction.
- the calibration of direct Cross-Resonance gate.

The dependency is documented under `requirement.txt`, with qiskit support till `1.1.1`. The notebooks are tested under the given environment. As qiskit keeps evolving, there is no garantee that it alwasy works on the newest version.
