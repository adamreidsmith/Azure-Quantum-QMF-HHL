import os
import warnings
from pathlib import Path
from typing import Optional, Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider, AzureQuantumJob


if os.name == 'nt':
    import msvcrt
elif os.name == 'posix':
    import fcntl
else:
    raise RuntimeError("Unknown operating system")


class AerRunner:
    '''A class to run circuits on a local Aer simulator.'''

    def __init__(self, shots: int = 1):
        self.sim = AerSimulator(method='automatic')
        self.shots = shots
        self.n_calls = 0
        self.n_qubits = 0

    def run(self, qc: QuantumCircuit, run_id: Any) -> dict[str, int]:
        '''
        Simulate a Qiskit quantum circuit and return the dict of counts.
        '''

        qc = transpile(qc, self.sim, optimization_level=3)

        self.n_calls += 1
        self.n_qubits += qc.num_qubits
        counts = self.sim.run(qc, shots=self.shots).result().get_counts()
        return counts

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.shots))


class AzureRunner:
    def __init__(
        self,
        backend_name: str = 'ionq.simulator',
        shots: int = 1,
        transpile: bool = True,
        dryrun: bool = True,
        approximation_degree: float = 1.0,
    ):
        self.workspace = Workspace(
            resource_id="/subscriptions/d570a97a-c638-4dd4-a437-4916bfc1bbe8/resourceGroups/GHGQuantum-rg/providers/Microsoft.Quantum/Workspaces/GHG-Quantum-Workspace",
            location="westus",
        )
        self.provider = AzureQuantumProvider(self.workspace)
        self.backend_name = backend_name
        self.backend = self.provider.get_backend(self.backend_name)

        self.shots = shots
        self.transpile = transpile
        self.dryrun = dryrun
        self.approximation_degree = approximation_degree
        self.sim = AerSimulator() if self.dryrun else None

        self.n_calls = 0
        self.n_qubits = 0
        self.total_cost = 0
        self.currency = None

        self.run_retries = 0

        self.job_id_file = Path(__file__).parent / f'.job_ids_{self.backend_name}.json'

    def list_backends(self) -> None:
        for backend in self.provider.backends():
            print(backend.name())

    def run(self, qc: QuantumCircuit, retries: int = 2, run_id: Optional[int] = None) -> dict[str, int]:
        self.n_calls += 1
        self.n_qubits += qc.num_qubits

        # Transpile the QC
        if self.transpile:
            qc_transpiled = transpile(
                qc, self.backend, optimization_level=3, approximation_degree=self.approximation_degree
            )

        # Save cost estimates, if available
        try:
            cost, currency = self.estimate_cost(qc_transpiled)
        except AttributeError:
            self.total_cost = None
        else:
            if self.currency is None:
                self.currency = currency
            else:
                assert self.currency == currency, f'Currency changed from {self.currency} to {currency}'
            self.total_cost += cost

        # If a dry run, do not run the QPU
        # Instead, we just simulate the results
        if self.dryrun:
            job = self.sim.run(qc, shots=self.shots)
        elif run_id is None:
            job = self.backend.run(qc_transpiled, shots=self.shots)
            print(f'Job {job.id()} submitted')
        else:
            # Get the existing job_ids
            try:
                with open(self.job_id_file, 'r') as f:
                    self.lock_file(f)
                    job_ids_list = f.readlines()
                    self.lock_file(f, unlock=True)
                job_ids = (ids.split(':') for ids in job_ids_list)
                job_ids = {rid: jid[:-1] for rid, jid in job_ids}
            except FileNotFoundError:
                job_ids = {}

            # Determine if the run_id is in the job_file, i.e. the job has already been submitted
            # job_ids is a dictionary mapping run_id to job_id
            job_id = job_ids.get(str(run_id), None)
            if job_id is None:
                # Submit the job
                job = self.backend.run(qc_transpiled, shots=self.shots)
                print(f'Job {job.id()} submitted')

                # Append the run_id and job_id to the file
                with open(self.job_id_file, 'a') as f:
                    self.lock_file(f, unlock=True)
                    f.write(f'{run_id}:{job.id()}\n')
                    self.lock_file(f, unlock=True)
            else:
                # Obtain the job from the workspace if it has already been submitted
                print(f'Job {job_id} exists, reading')
                az_job = self.workspace.get_job(job_id)
                job = AzureQuantumJob(backend=self.backend, azure_job=az_job)

        result = job.result()
        # Check if the circuit ran properly
        if not getattr(result, 'success', False) and self.run_retries < retries:
            self.run_retries += 1
            warnings.warn(f'QC run failed {self.run_retries} time(s). Retrying.')
            return self.run(qc, retries)

        counts = result.get_counts()
        self.run_retries = 0
        return counts

    def estimate_cost(self, qc: QuantumCircuit) -> None:
        cost = self.backend.estimate_cost(qc, shots=self.shots)
        return cost.estimated_total, cost.currency_code

    def remove_job_file(self):
        try:
            os.remove(self.job_id_file)
        except FileNotFoundError:
            warnings.warn('Job file does not exist')

    def __hash__(self) -> int:
        h = (self.__class__.__name__, self.backend_name, self.transpile, self.dryrun)
        return hash(h)

    @staticmethod
    def lock_file(file, unlock=False):
        if os.name == 'nt':
            msvcrt.locking(file.fileno(), msvcrt.LK_ULOCK if unlock else msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(file, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX)


if __name__ == '__main__':
    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(qc.num_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    # runner = AzureRunner(shots=1000, dryrun=True)
    # runner.list_backends()
    # # runner = AzureRunner('rigetti.sim.qvm', shots=1000)
    # # runner = AzureRunner('quantinuum.sim.h1-1e', shots=1000)
    # # runner = AzureRunner('microsoft.estimator', shots=1000)
    # # runner.estimate_cost(qc)

    # for _ in range(10):
    #     runner.run(qc)

    # print(runner.total_cost, runner.currency)

    runner = AzureRunner(shots=1000, dryrun=True)
    runner.list_backends()
