import math
import random
import statistics as stats
from pathlib import Path
from operator import itemgetter
from functools import partial
from typing import Any
from collections.abc import Callable

from qiskit import QuantumCircuit

from quantum_runner import AerRunner, AzureRunner


# def less_than_oracle(k, n_qubits):
#     '''
#     Defines an oracle which flips the phase of states |x> s.t. x < k.
#     The oracle contains `n_qubits` state qubits, one `compare` qubit which is 1 if the state is less than
#     k and 0 otherwise, and some number of ancillary qubits.
#     '''
#     from qiskit.circuit.library import IntegerComparator

#     # Create the comparator circuit
#     # The IntegerComparator adds ancillas internally as needed
#     # The comparator circuit will flip an ancilla qubit if the condition is met (i.e., register value < k)
#     comparator = IntegerComparator(num_state_qubits=n_qubits, value=k, geq=False)

#     # Compute the inverse of the comparator
#     comparator_inverse = comparator.inverse()

#     # Apply a controlled phase flip based on the ancilla outcome
#     # Flip the phase of states where the comparator's output ancilla is in the |1> state (condition met)
#     comparator.z(n_qubits)  # The output is in the `compare` register in the `n_qubits` position

#     # Uncompute (reset the ancillas)
#     comparator.append(comparator_inverse, comparator.qubits)

#     return comparator


def plot_statevector(qc: QuantumCircuit, figsize: tuple[int] = (17, 6)) -> None:
    '''
    Plot the statevector histogram from a quantum circuit. Works best for < 7 qubits.

    Parameters
    ----------
    qc : QuantumCircuit
    figsize : tuple[int]
        Figure size passed to matplotlib.pyplot.figure
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from qiskit.quantum_info import Statevector

    state_vector = np.asarray(Statevector(qc))

    # Extract amplitudes and phases
    amplitudes = np.abs(state_vector)
    phases = np.angle(state_vector).round(5)
    phases = np.where(phases < 0, 2 * np.pi + phases, phases)  # Map the phases to [0, 2pi)

    # Prepare data for histogram
    states = [bin(i)[2:].rjust(qc.num_qubits, '0') for i in range(2**qc.num_qubits)]
    data = {states[i]: amplitudes[i] for i in range(len(states))}

    # Plot histogram of amplitudes
    plt.figure(figsize=figsize)
    bars = plt.bar(data.keys(), data.values(), color=plt.cm.hsv(phases / (2 * np.pi)))
    plt.xticks(rotation=90)
    plt.xlabel('State')
    plt.ylabel('Amplitude')
    plt.title('Quantum State Amplitudes and Phases')

    # Add phase text on the corresponding bar
    for bar, phase in zip(bars, phases):
        if bar.get_height():
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.02,
                f'{phase:.3f}',
                ha='center',
                va='bottom',
                rotation='vertical',
            )

    plt.tight_layout()
    plt.show()


def durr_hoyer_oracle(T: list[float], k: int, n_qubits: int, barriers: bool = False) -> QuantumCircuit:
    '''
    Defines an oracle which flips the phase of states |j> such that T[j] < T[k].
    This oracle is defined in (at least) O(L*log(L)) time, where L = len(T).

    Parameters
    ----------
    T : list[float]
        The list of values passed tot he QMF algorithm.
    k : int
        The current guess at the index of the minimum.
    n_qubits : int
        The number of qubits in the oracle.
    barriers : bool
        Whether or not to place barriers between each phase-flipping operation.

    Returns
    -------
    QuantumCircuit
        The oracle circuit.
    '''

    assert n_qubits >= math.ceil(math.log2(len(T))), f'{n_qubits} cannot encode {len(T)} indices'

    oracle = QuantumCircuit(n_qubits)

    for j, val in enumerate(T):
        if val >= T[k]:
            continue

        # Convert the index to binary and padd zeros on the left
        j_bin = f'{j:b}'.zfill(n_qubits)

        # Find the indices of all the '0' elements in bit-string.
        # Search the string in reverse to match Qiskit bit-ordering.
        zero_inds = [i for i in range(n_qubits) if j_bin[-i - 1] == '0']

        # Add a multi-controlled phase gate with pre- and post-applied X-gates where the target bit-string has a '0'
        if zero_inds:
            oracle.x(zero_inds)
        oracle.mcp(math.pi, list(range(n_qubits - 1)), n_qubits - 1)  # Equivalent to a multi-controlled z-gate
        if zero_inds:
            oracle.x(zero_inds)

        if barriers:
            oracle.barrier(range(n_qubits))

    return oracle


def grover_diffuser(n_qubits: int) -> QuantumCircuit:
    '''
    Define the n-qubit Grover diffuser. This applies a Hadamard on all qubits, flips the sign
    of the |0> state, and again applies Hadamards.

    Parameters
    ----------
    n_qubits : int
        The number of qubits in the diffuser.

    Returns
    -------
    QuantumCircuit
        The diffuser circuit.
    '''

    diffuser = QuantumCircuit(n_qubits)
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.mcp(math.pi, list(range(n_qubits - 1)), n_qubits - 1)  # Equivalent to a multi-controlled z-gate
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def durr_hoyer_qmf(T: list[float], quantum_runner: Any) -> tuple[int, float]:
    '''
    Implements the quantum minimum finding algorithm introduced by Durr and Hoyer here:
    https://arxiv.org/abs/quant-ph/9607014

    This algorithms uses 22.5 * sqrt(N) + 1.4 * log(N)^2 oracle queries, where N = len(T).
    Assuming the oracle runs in constant time (this is not the case, but is the general
    assumption when computing time complexitites for quantum oracular algorithms), this
    requires a search space of at least 739 values to run "faster" than a classical minimum
    finding algo, where speed is defined in terms of oracle calls.

    Parameters
    ----------
    T : list[float]
        A list of values to compute the minimum of.
    quantum_runner : Any
        An object with a `run` method that accepts a qiskit.QuantumCircuit and returns a
        dictionary of counts. All transpilation/circuit pre-processing should be performed here.

    Returns
    -------
    tuple[int, float]
        The index of the minimum and the minimum element.
    '''

    L = len(T)
    # Handle base cases
    if L == 0:
        raise ValueError('durr_hoyer_qmf() argument is empty')
    if L == 1:
        return 0, T[0]
    if L == 2:
        return min(((0, T[0]), (1, T[1])), key=itemgetter(1))

    n_qubits = math.ceil(math.log2(L))

    # Initial guess at the index of the minimum
    min_index = random.randrange(L)

    # Extend T with inf to consider only the case where len(T) is a power of 2
    T.extend([math.inf] * (2**n_qubits - L))
    N = len(T)  # Power of 2

    runtime = 0  # Runtime refers to the number of calls to the oracle
    max_runtime = 22.5 * math.sqrt(N) + 1.4 * math.log2(N) ** 2
    while True:
        m = 1.0
        lbda = 6.0 / 5.0
        while True:
            # Choose j uniformly at random among the non-negative integers smaller than m
            j = random.randrange(math.ceil(m))

            # Initialize each qubit in the |+> state
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))

            # Implement the amplitude amplification search
            oracle = durr_hoyer_oracle(T, min_index, n_qubits)
            diffuser = grover_diffuser(n_qubits)
            for _ in range(j):
                qc.compose(oracle, inplace=True)
                qc.compose(diffuser, inplace=True)
            qc.measure_all()

            # Increment the runtime
            runtime += j

            # Execute the circuit and obtain a measurement
            counts = quantum_runner.run(qc)
            most_common = max(counts, key=counts.get)
            index = int(most_common, 2)

            # If we found a valid solution, return it
            if T[index] < T[min_index]:
                min_index = index
                break
            if runtime > max_runtime:
                break
            # Otherwise, update m and continue
            m = min(lbda * m, math.sqrt(N))

        if runtime > max_runtime:
            break

    return min_index, T[min_index]


def makhanov_oracle(T: list[float], k: int, n_qubits: int, barriers: bool = False) -> QuantumCircuit:
    '''
    Defines an oracle which flips the phase of states |j> such that T[j] < T[k].
    Identical to the oracle in Durr and Hoyer's algorithm.

    Parameters
    ----------
    T : list[float]
        The list of values passed tot he QMF algorithm.
    k : int
        The current guess at the index of the minimum.
    n_qubits : int
        The number of qubits in the oracle.
    barriers : bool
        Whether or not to place barriers between each phase-flipping operation.

    Returns
    -------
    QuantumCircuit
        The oracle circuit.
    '''

    return durr_hoyer_oracle(T, k, n_qubits, barriers)


def makhanov_qmf_helper(T: list[float], L: int, n_qubits: int, quantum_runner: Any, seed: int):
    '''
    The guts of the Makhanov QMF algorithm. Used as a helper function for `makhanov_qmf`.
    '''

    random.seed(seed)

    # Initial guess at the index of the minimum
    min_index = random.randrange(L)

    while True:
        # Initialize qubits in the |+> state
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits - 1))

        # Implement the amplitude amplification search
        n_marked_states = max(sum(x < T[min_index] for x in T), 1)
        iterations = math.floor(math.pi / 4.0 * math.sqrt(len(T) / n_marked_states))
        oracle = durr_hoyer_oracle(T, min_index, n_qubits)
        diffuser = grover_diffuser(n_qubits)
        for _ in range(iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, inplace=True)
        qc.measure_all()

        # Execute the circuit and obtain a measurement
        run_id = hash((seed, tuple(T), min_index, L, n_qubits))
        counts = quantum_runner.run(qc, run_id=run_id)
        most_common = max(counts, key=counts.get)
        index = int(most_common, 2)

        if T[index] < T[min_index]:
            min_index = index
        else:
            return min_index, T[min_index]


def makhanov_qmf(T: list[float], quantum_runner: Any, repeats: int = 1, processes: int = 1):
    '''
    Implements the quantum minimium finding algorithm described by Makhanov et al. here:
    https://arxiv.org/abs/2304.14445

    Parameters
    ----------
    T : list[float]
        A list of values to compute the minimum of.
    quantum_runner : Any
        An object with a `run` method that accepts a qiskit.QuantumCircuit and an integer ID and
        returns a dictionary of counts. All transpilation/circuit pre-processing should be performed here.
    repeats : int
        The number of times to repeat the algorithm. The most commonly found minimum is returned.
    processes : int
        The number of processes to launch to run the algorithm repetitions in parallel. Setting
        this to -1 will use as many processes as repeats.

    Returns
    -------
    tuple[int, float]
        The index of the minimum and the minimum element.
    '''

    seeds = random.sample(range(1_000, 1_000_000), repeats)

    if processes != 1:
        from multiprocessing import Pool
        import warnings

        warnings.warn('quantum_runner instance attributes will not be accurate when processes != 1')

    if processes == -1:
        processes = repeats

    L = len(T)
    # Handle base cases
    if L == 0:
        raise ValueError('durr_hoyer_qmf() argument is empty')
    if L == 1:
        return 0, T[0]
    if L == 2:
        return min(((0, T[0]), (1, T[1])), key=itemgetter(1))

    n_qubits = math.ceil(math.log2(L))

    T.extend([math.inf] * (2**n_qubits - L))

    helper_args = (T, L, n_qubits, quantum_runner)
    if processes == 1:
        min_indices = [makhanov_qmf_helper(*helper_args, seeds[i])[0] for i in range(repeats)]
    else:
        with Pool(processes) as pool:
            result_tuples = pool.starmap(
                makhanov_qmf_helper, ((*args, seeds[i]) for i, args in enumerate([helper_args] * repeats))
            )
        min_indices = [x[0] for x in result_tuples]

    min_index = stats.mode(min_indices)
    return min_index, T[min_index]


def run_qmf(values, repeats, quantum_runner, qmf_algo):
    '''A helper function for running algorithms in parallel.'''
    return stats.mode(qmf_algo(values, quantum_runner)[1] for _ in range(repeats))


def success_probability(
    qmf_algo: Callable[[list[float], Any], float],
    length: int,
    quantum_runner: Any,
    iterations: int,
    processes: int,
    repeats: int,
    max_val: int = None,
) -> float:
    '''
    Compute the success probability of a QMF algorithm.

    Parameters
    ----------
    length : int
        The length of lists of random numbers to compute the minimum of.
    quantum_runner : Any
        An object with a `run` method that accepts a qiskit.QuantumCircuit and an integer ID and
        returns a dictionary of counts. All transpilation/circuit pre-processing should be performed here.
    iterations : int
        The number of repetitions of the whole quantum minimum finding process to perform to collect
        summary statistics.
    processes : int
        The number of processes to launch to compute runs in parallel.
    repeats : int
        The number of algorithm repetitions to use to compute a minimum.
    max_val : int
        The random lists of values will be sampled from [0, max_val]. If None, `max_val` is set to `length`.

    Returns
    -------
    float
        The success probability.
    '''

    if processes > 1:
        from multiprocessing import Pool

    values = [[random.randint(0, max_val or length) for _ in range(length)] for _ in range(iterations)]
    true_mins = list(map(min, values))

    qmf_partial = partial(run_qmf, repeats=repeats, quantum_runner=quantum_runner, qmf_algo=qmf_algo)
    if processes > 1:
        with Pool(10) as pool:
            mins = pool.map(qmf_partial, values)
    else:
        mins = map(qmf_partial, values)

    successes = sum(t == d for t, d in zip(true_mins, mins))
    prob = successes / iterations
    print(f'Success probability: {prob:.2%}')
    return prob


def sp_by_shots(
    qmf_algo,
    length: int = 10,
    repeats: list[int] = [1, 3, 10, 25, 50, 100],
    log_shots_max: int = 11,
    iterations: int = 10_000,
):
    '''
    Plot the success probability as a function of the number of shots.
    '''
    import matplotlib.pyplot as plt

    all_shots = []
    all_probs = []

    for i, repeat in enumerate(repeats):
        shots = [2**n for n in range(log_shots_max - i)]
        probs = []
        for shot in shots:
            runner = AerRunner(shots=shot)
            success_prob = success_probability(
                qmf_algo, length=length, quantum_runner=runner, iterations=iterations, processes=10, repeats=repeat
            )
            probs.append(success_prob)
            print(repeat, shot, success_prob)
        all_probs.append(probs)
        all_shots.append(shots)

    print(all_shots)
    print(all_probs)

    plt.figure(figsize=(10, 8))
    for i, (shots, probs) in enumerate(zip(all_shots, all_probs)):
        plt.plot(shots, [100 * p for p in probs], label=f'{repeats[i]} repeats', marker='o')

    plt.xlabel('Shots')
    plt.ylabel('Success probability (%)')
    plt.title(f'Success probability of Makhanov QMF on {length}-value list')
    plt.xscale('log')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(Path(__file__).parent / f'len{length}.png', dpi=300)
    # plt.show()

    # Conclusion:
    # Best option is to run QMF 50 times with 8 shots and take the mode.
    # This gives ~99% success probability for lists of length <= 100 and 96.5%
    # success probability for lists of length 1000.


if __name__ == '__main__':
    random.seed(119911)  ## IONQ SEED

    shots = 1
    repeats = 50
    values = [78, 15, 46, 52, 5, 14, 66, 85, 3, 6]

    runner = AzureRunner('ionq.qpu', shots=shots, transpile=True, dryrun=True)
    min_index, min_val = makhanov_qmf(values, runner, repeats, processes=-1)
    # min_index, min_val = durr_hoyer_qmf(values, runner)
    print(f'True min: {min(values)}')
    print(f'QMF min: {min_val}')
    print('Cost:', runner.total_cost, runner.currency)
    print('Runner calls:', runner.n_calls)
    print('Runner qubits:', runner.n_qubits)
