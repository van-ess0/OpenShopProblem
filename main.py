import re, random, math
from operator import attrgetter
import numpy as np

lookup_table = None


class Chromosome(object):
    """
    The chromosome is a representation
    of the genetic chromosome that is used
    to develop a schedule and calculate the
    makespan.
    The chromosome is represented as an array
    that is of size m * n, where m is the number
    of machines, and n is the number of jobs.
    Each operation is assigned a unique identifier
    with its cost. So Job 1 on Machine 1 would have
    an identifier of 1.
    The array represents the order in which operations
    are scheduled.
    A possible Chromosome might look like the following
    if there are 3 jobs and 3 machines
    [7, 5, 9, 3, 1, 8, 6, 2, 4]
    This means that schedule operation 7 first, then
    schedule operation 5, then schedule operation 9, and
    so on.
    """

    def __init__(self, sequence):
        self.sequence = sequence
        self._makespan = None
        self.schedule = np.zeros((len(lookup_table), len(lookup_table[0])))

    @property
    def makespan(self):
        schedule = [[0 for j in lookup_table[0]] for i in lookup_table]
        iter = 1
        for operation in self.sequence:
            job = id_to_job_index(operation)
            machine = id_to_machine_index(operation)
            op_cost = lookup_table[machine][job]

            schedule[machine][job] = op_cost
            self.schedule[machine][job] = iter
            iter += 1
        makespan = max(map(sum, schedule))

        return makespan


class Population(object):
    def __init__(self, size):
        self.size = size
        self._members = []
        self._seed_population()

    def _seed_population(self):
        sequence_size = len(lookup_table) * len(lookup_table[0])
        for i in range(self.size):
            sequence = random.sample(range(1, sequence_size + 1), sequence_size)
            # print('sequence', sequence)
            self._members.append(Chromosome(sequence))

    def evolve_population(self):
        (parent_one, parent_two) = self._selection()
        child = self._crossover(parent_one, parent_two)
        self._members.append(child)
        self.kill_weak()
        for member in self._members:
            if random.random() < MUTATION:
                self._mutate(member)

    def _crossover(self, parent_one, parent_two):
        (start_index, end_index) = random.sample(range(len(parent_one.sequence)), 2)

        child_seq = [None] * len(parent_one.sequence)

        for i in range(len(child_seq)):
            if start_index < end_index and i >= start_index and i <= end_index:
                child_seq[i] = parent_two.sequence[i]
            elif start_index > end_index:
                if not (i <= start_index and i >= end_index):
                    child_seq[i] = parent_one.sequence[i]

        for i in range(len(child_seq)):
            if parent_two.sequence[i] not in child_seq:
                for j in range(len(child_seq)):
                    if child_seq[j] is None:
                        child_seq[j] = parent_two.sequence[i]
                        break

        return Chromosome(child_seq)

    def _mutate(self, member):
        (index_one, index_two) = random.sample(range(len(member.sequence)), 2)

        member.sequence[index_one], member.sequence[index_two] = member.sequence[index_two], member.sequence[index_one]

    def _selection(self):
        num_to_select = math.floor(self.size * (GROUP / 100))
        sample = random.sample(range(self.size), num_to_select)
        sample_members = sorted([self._members[i] for i in sample], key=attrgetter('makespan'))
        return sample_members[:2]

    def fittest(self, size):
        return sorted(self._members, key=attrgetter('makespan'))[:size]

    def kill_weak(self):
        weakest = max(self._members, key=attrgetter('makespan'))
        self._members.remove(weakest)


def parse_file(file_path):
    """
    Parse file returns a tuple of length two
    that contains the processing power of the machines
    """
    with open(file_path) as file:
        contents = file.read()
        machines = list(map(float, re.findall(r'MACHINE: ([0-9. ]*)', contents)[0].split(' ')))
        jobs = list(map(int, re.findall(r'JOBS: ([0-9 ]*)', contents)[0].split(' ')))

        return machines, jobs


def id_to_machine_index(id):
    transposed_id = id - 1
    return transposed_id // len(lookup_table[0])


def id_to_job_index(id):
    transposed_id = id - 1
    return transposed_id % len(lookup_table[0])


def create_lookup_table(machines, jobs):
    """
    Create a lookup table to lookup what each
    job will cost on a specific machine.
    The jobs are the rows, and the machines are the columns
    For example: To find the operation cost to perform
    job 1 on machine 1, you would access the data held
    in the lookup table at index [0][0]. To find the
    operation cost to run job 2 on machine 3, you would
    access the data held in the lookup table [1][2]
    Note: The index is of the machine and job is always
    one less then the ID of the job/machine
    """
    global lookup_table

    lookup_table = [[0 for i in jobs] for i in machines]

    for i in range(0, len(machines)):
        for j in range(0, len(jobs)):
            lookup_table[j][i] = machines[i] * jobs[j]

    # print('lookup_table', lookup_table)


def start_ga(population):
    best = 999999999
    for i in range(GENERATIONS):
        population.evolve_population()
        fittest = population.fittest(1)[0]
        if fittest.makespan < best:
            best = fittest.makespan
    print('best:', best)
    print('schedule:\n', fittest.schedule)


POPULATION = 100
MUTATION = 2.5
GENERATIONS = 10
GROUP = 10
file = "test_data.txt"

machines, jobs = parse_file(file)

print('machines, jobs ', machines, jobs)
create_lookup_table(machines, jobs)
lookup_table = [[10, 7, 3, 1, 12, 6], [6, 9, 8, 2, 7, 6]]
print(np.array(lookup_table))

population = Population(POPULATION)
start_ga(population)
