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

    @property
    def makespan(self):
        schedule = [[0 for j in lookup_table[0]] for i in lookup_table]

        for operation in self.sequence:
            job = id_to_job_index(operation)
            machine = id_to_machine_index(operation)
            op_cost = id_to_lookup(operation)

            column = [row[machine] for row in schedule]
            next_time = max(column)

            if max(schedule[job]) > next_time:
                next_time = max(schedule[job])

            schedule[job][machine]  = next_time + op_cost

        columns = zip(*schedule)
        makespan = 0
        for column in columns:
            cand = max(column)
            if cand > makespan:
                makespan = cand
        return makespan