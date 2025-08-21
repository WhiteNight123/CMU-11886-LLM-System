from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    total_cycles = num_batches + num_partitions - 1
    for clock in range(total_cycles):
        schedule = []
        for batch_idx in range(num_batches):
            partition_idx = clock - batch_idx
            if 0 <= partition_idx < num_partitions:
                schedule.append((batch_idx, partition_idx))
        yield schedule
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        batches = list(torch.split(tensor = x, split_size_or_sections = self.split_size))
        schedules = list(_clock_cycles(len(batches), len(self.partitions)))
        for schedule in schedules:
            # if schedule:  # Skip empty schedules
            self.compute(batches, schedule)
        result = torch.cat([batch.to(self.devices[-1]) for batch in batches], dim=0)
        return result
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        for microbatch_idx, partition_idx in schedule:
            partition = partitions[partition_idx]
            in_queue = self.in_queues[partition_idx]
            out_queue = self.out_queues[partition_idx]
            microbatch = batches[microbatch_idx]
            device = devices[partition_idx]
            def compute_function():
                microbatch_on_device = microbatch.to(device)
                return partition(microbatch_on_device)

            task = Task(compute_function)
            in_queue.put(task)
            success, result = out_queue.get()
            if not success:
                exc_info = result
                raise exc_info[1].with_traceback(exc_info[2])
            _, output = result
            batches[microbatch_idx] = output
        # END ASSIGN5_2_2
