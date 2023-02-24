from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from elixir import gpu_dev

from .memory_pool import MemoryPool, PrivateBlock, PublicBlock, TensorBlock
from .states import ChunkState, TensorState, ts_update_sanity_check


class ChunkFullError(Exception):
    pass


@dataclass
class TensorInfo:
    state: TensorState
    shape: torch.Size
    offset: int
    end: int


class Chunk:
    total_count = 0

    def __init__(
        self,
        rcache: MemoryPool,
        chunk_size: int,
        chunk_dtype: torch.dtype,
        process_group: ProcessGroup,
        temp_device: Optional[torch.device] = None,
        shard_device: Optional[torch.device] = None,
        rcache_fused: bool = False,    # whether this chunk is used in ZeRO2
        cpu_pin_memory: bool = False    # whether this chunk has a permanent copy in cpu
    ) -> None:

        self.chunk_id: int = Chunk.total_count
        Chunk.total_count += 1
        # set rcache
        self.rcache: MemoryPool = rcache

        self.chunk_size: int = chunk_size
        self.chunk_dtype: torch.dtype = chunk_dtype
        self.utilized_size: int = 0

        self.torch_pg: ProcessGroup = process_group
        self.pg_size: int = dist.get_world_size(self.torch_pg)
        self.pg_rank: int = dist.get_rank(self.torch_pg)

        # the chunk size should be divisible by the dp degree
        assert chunk_size % self.pg_size == 0

        self.shard_size: int = chunk_size // self.pg_size
        self.shard_begin: int = self.shard_size * self.pg_rank
        self.shard_end: int = self.shard_begin + self.shard_size
        self.valid_end: int = self.shard_size + 1    # set to an illegal number

        # configure the init device of the shard
        # no-offload default: fp16, fp32 -> CUDA
        # offload default: fp16, fp32 -> CPU
        shard_device: torch.device = shard_device or torch.device('cpu')
        pin_flag: bool = cpu_pin_memory and shard_device.type is 'cpu'
        # chunk.shard is a local chunk
        # it is desinged to exist permanently
        self.shard: torch.Tensor = torch.empty(self.shard_size,
                                               dtype=chunk_dtype,
                                               device=shard_device,
                                               pin_memory=pin_flag)

        # rcache block, the global replicated chunk in R cache
        self.rcb: Optional[TensorBlock] = None
        self.rcache_fused: bool = rcache_fused
        self.is_replica: bool = True

        temp_device: torch.device = temp_device or gpu_dev()
        # chunk_temp is a global chunk, which only exists during building the chunks.
        # keep all elements to zero
        self.chunk_temp: Optional[torch.Tensor] = None
        if rcache_fused:
            self.rcb = rcache.get_private_block(chunk_size, chunk_dtype)
            self.chunk_temp = self.rcb.payload
            torch.zero_(self.chunk_temp)
        else:
            self.chunk_temp = torch.zeros(chunk_size, dtype=chunk_dtype, device=temp_device)

        # calculate the memory occupation of the chunk and the shard
        self.chunk_memo: int = self.chunk_size * self.chunk_temp.element_size()
        self.shard_memo: int = self.chunk_mem // self.pg_size

        # each tensor is associated with a TensorInfo to track its meta info
        # (state, shape, offset, end)
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        # the total number of tensors in the chunk
        self.num_tensors: int = 0

        # Record the number of tensors in different states
        self.tensor_state_cnter: Dict[TensorState, int] = dict()
        for state in TensorState:
            self.tensor_state_cnter[state] = 0

        # we introduce the paired chunk here
        # it refers to another chunk having the same parameters
        # but with different dtype(such as fp16_chunk.paired_chunk -> fp32_chunk
        self.paired_chunk = None
        # if this chunk is synchronized with the optimizer, the flag is True
        self.optim_sync_flag = True

        # whether to record l2 norm for the gradient clipping calculation
        self.l2_norm_flag = False
        self.l2_norm = None
        # whether it overflows after the reduction
        self.overflow_flag = False

    @property
    def is_init(self):
        return self.chunk_temp is not None

    @property
    def shard_device(self):
        return self.shard.device

    @property
    def memory_usage(self) -> Dict[str, int]:
        cuda_memory = 0
        cpu_memory = 0

        if self.is_init:
            # this chunk is not closed
            if self.chunk_temp.device.type == 'cuda':
                cuda_memory += self.chunk_mem
            else:
                cpu_memory += self.chunk_mem
        else:
            if self.rcb is not None:
                cuda_memory += self.rcb.memo_occ

            if self.shard_device.type == 'cuda':
                cuda_memory += self.shard_mem
            elif self.shard_device.type == 'cpu':
                cpu_memory += self.shard_mem
            else:
                raise NotImplementedError

        return dict(cuda=cuda_memory, cpu=cpu_memory)

    @property
    def payload(self) -> torch.Tensor:
        if self.is_init:
            return self.chunk_temp

        if self.rcb is not None:
            return self.rcb.payload
        else:
            return self.shard

    @property
    def shard_move_check(self) -> bool:
        return not self.is_gathered

    @property
    def scatter_check(self) -> bool:
        if self.keep_gathered:
            return False
        return self.tensor_state_cnter[TensorState.HOLD] + \
            self.tensor_state_cnter[TensorState.HOLD_AFTER_BWD] == self.num_tensors

    @property
    def reduce_check(self):
        return self.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == self.num_tensors

    def set_overflow_flag(self, valid_tensor: torch.Tensor) -> None:
        """Check if the chunk has inf or nan values on CUDA.
        """
        assert not self.overflow_flag
        self.overflow_flag = torch.isinf(valid_tensor).any().item() | torch.isnan(valid_tensor).any().item()

    def set_l2_norm(self, valid_tensor: torch.Tensor) -> None:
        """Record l2 norm of this chunks on CUDA.
        """
        assert self.l2_norm is None, 'you are calculating the l2 norm twice'
        chunk_l2_norm = valid_tensor.data.float().norm(2)
        self.l2_norm = chunk_l2_norm.item()**2

    def append_tensor(self, tensor: torch.Tensor):
        """Add a tensor to the chunk.

        Args:
            tensor (torch.Tensor): a tensor to be added to the chunk
        """
        # sanity check
        assert self.is_init
        assert tensor.dtype == self.dtype

        new_utilized_size = self.utilized_size + tensor.numel()
        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.chunk_size:
            raise ChunkFullError

        self.chunk_temp[self.utilized_size:new_utilized_size].copy_(tensor.data.flatten())
        tensor.data = self.chunk_temp[self.utilized_size:new_utilized_size].view(tensor.shape)

        # record all the information about the tensor
        self.num_tensors += 1
        tensor_state = TensorState.HOLD
        self.tensors_info[tensor] = TensorInfo(tensor_state, tensor.shape, self.utilized_size, new_utilized_size)
        self.tensor_state_cnter[tensor_state] += 1
        self.utilized_size = new_utilized_size

    def close_chunk(self):
        """Close the chunk. Any tensor can't be appended to a closed chunk later.
        """
        # sanity check
        assert self.is_init

        # calculate the valid end for each shard
        if self.utilized_size <= self.shard_begin:
            self.valid_end = 0
        elif self.utilized_size < self.shard_end:
            self.valid_end = self.utilized_size - self.shard_begin

        self.__update_shard(self.chunk_temp, self.shard)
        self.is_replica = False

        self.chunk_temp = None

    def replicate(self):
        assert not self.is_replica

        this_shard = self.shard if self.optim_sync_flag else self.__paired_shard_move()
        self.__update_replica(self.rcb.payload, this_shard)
        self.__update_tensors_ptr()
        self.is_replica = True

    def scatter(self):
        assert not self.rcache_fused
        assert self.is_replica

        self.__remove_tensors_ptr()
        if not self.optim_sync_flag:
            self.__update_shard(self.rcb.payload, self.shard)
            self.optim_sync_flag = True
        self.is_replica = False

    def reduce(self):
        assert self.is_replica

        self.__remove_tensors_ptr()
        buffer = self.rcb.payload[self.shard_begin:self.shard_end]
        if self.pg_size > 1:
            input_list = list(torch.chunk(self.rcb.payload, chunks=self.pg_size, dim=0))
            dist.reduce_scatter(buffer, input_list, group=self.torch_pg)
        self.__update_shard(self.rcb.payload, self.shard)

        valid_tensor = buffer[:self.valid_end]
        self.set_overflow_flag(valid_tensor)
        if self.l2_norm_flag:
            self.set_l2_norm(valid_tensor)

        self.is_replica = False

    def access_chunk(self, block: Optional[TensorBlock]):
        # sanity check
        assert not self.is_init
        assert not self.is_replica

        if self.rcache_fused:
            assert block is None
            assert self.rcb.block_type == 'private'
        else:
            assert block.block_type == 'public'
            assert self.rcb is None
            self.rcb = block

        self.replicate()

    def release_chunk(self) -> TensorBlock:
        # sanity check
        assert not self.is_init
        assert self.is_replica

        if self.rcache_fused:
            raise RuntimeError

        self.scatter()
        block = self.rcb
        self.rcb = None
        return block

    def reduce_chunk(self) -> Optional[TensorBlock]:
        """Reduce scatter all the gradients. It's an operation done in CUDA.
        """
        # sanity check
        assert not self.is_init
        assert self.is_replica

        self.reduce()
        self.__update_tensors_state(TensorState.HOLD)

        if self.rcache_fused:
            return None

        block = self.rcb
        self.rcb = None
        return block

    def tensor_trans_state(self, tensor: torch.Tensor, tensor_state: TensorState) -> None:
        """
        Make a transition of the tensor into the next state.

        Args:
            tensor (torch.Tensor): a torch Tensor object.
            tensor_state (TensorState): the target state for transition.
        """

        # As the gradient hook can be triggered either before or after post-backward
        # tensor's state can be compute -> hold_after_bwd -> ready_for_reduce
        # or compute -> ready_for_reduce -> hold_after_bwd
        # the second one is invalid, we just ignore ready_for_reduce -> hold_after_bwd
        # this function only apply valid state transformation
        # invalid calls will be ignored and nothing changes
        if (self.tensors_info[tensor].state, tensor_state) not in STATE_TRANS:
            return
        self.__update_one_tensor_info(self.tensors_info[tensor], tensor_state)

    def copy_tensor_to_chunk_slice(self, tensor: torch.Tensor, data_slice: torch.Tensor) -> None:
        """
        Copy data slice to the memory space indexed by the input tensor in the chunk.

        Args:
            tensor (torch.Tensor): the tensor used to retrive meta information
            data_slice (torch.Tensor): the tensor to be copied to the chunk
        """
        # sanity check
        assert self.is_gathered

        tensor_info = self.tensors_info[tensor]
        self.cuda_global_chunk[tensor_info.offset:tensor_info.end].copy_(data_slice.data.flatten())
        tensor.data = self.cuda_global_chunk[tensor_info.offset:tensor_info.end].view(tensor.shape)

    def init_pair(self, friend_chunk: 'Chunk') -> None:
        """Initialize the paired chunk.
        """
        if self.paired_chunk is None and friend_chunk.paired_chunk is None:
            self.paired_chunk = friend_chunk
            friend_chunk.paired_chunk = self
        else:
            assert self.paired_chunk is friend_chunk
            assert friend_chunk.paired_chunk is self

    def optim_update(self) -> None:
        """Update the fp16 chunks via their fp32 chunks. It's used by the optimizer.
        """
        # sanity check
        assert self.paired_chunk is not None

        friend_chunk = self.paired_chunk
        if self.is_gathered is True:
            assert friend_chunk.is_gathered is True
            self.cuda_global_chunk.copy_(friend_chunk.cuda_global_chunk)
            self.optim_sync_flag = True
        elif friend_chunk.device_type == 'cuda' and self.device_type == 'cuda':
            self.cuda_shard.copy_(friend_chunk.cuda_shard)
            self.optim_sync_flag = True
            self.cpu_vis_flag = False
        else:
            # optim_sync_flag is set to False
            # see shard_move function for more details
            assert friend_chunk.device_type == 'cpu'
            assert self.device_type == 'cpu'
            self.optim_sync_flag = False
            self.cpu_vis_flag = False

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())

    def __update_replica(self, replica: torch.Tensor, shard: torch.Tensor):
        assert self.is_replica
        assert replica.numel() == self.chunk_size
        assert shard.numel() == self.shard_size

        buffer = replica[self.shard_begin:self.shard_end]
        buffer.copy_(shard)

        gather_list = list(torch.chunk(input=replica, chunks=self.pg_size, dim=0))
        dist.all_gather(gather_list, buffer, group=self.torch_pg)

    def __update_shard(self, replica: torch.Tensor, shard: torch.Tensor):
        assert self.is_replica
        assert replica.numel() == self.chunk_size
        assert shard.numel() == self.shard_size

        shard.copy_(replica[self.shard_begin:self.shard_end])

    def __paired_shard_move(self):
        assert self.paired_chunk is not None, 'chunks should be paired before training'
        optim_chunk = self.paired_chunk
        assert self.chunk_size == optim_chunk.chunk_size

        # only be called when optimizer state is in CPU memory
        # the grad and param should be in the same device
        assert self.cuda_shard is None
        temp = optim_chunk.cpu_shard.to(get_current_device())
        # avoid to transform FP32 in CPU
        self.cuda_shard = temp.to(self.dtype)

        if not self.pin_memory:
            self.cpu_shard = None

    def __remove_tensors_ptr(self) -> None:
        empty_tensor = torch.empty(0, device='cuda')
        for tensor in self.tensors_info:
            tensor.data = empty_tensor

    def __update_tensors_ptr(self) -> None:
        # sanity check
        assert self.is_replica
        payload = self.rcb.payload
        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = self.payload[tensor_info.offset:tensor_info.end].view(tensor_info.shape)

    def __update_one_tensor_info(self, tensor_info: TensorInfo, next_state: TensorState):
        self.tensor_state_cnter[tensor_info.state] -= 1
        tensor_info.state = next_state
        self.tensor_state_cnter[tensor_info.state] += 1

    def __update_tensors_state(self, next_state: TensorState, prev_state: Optional[TensorState] = None):
        for tensor_info in self.tensors_info.values():
            if prev_state is None or tensor_info.state == prev_state:
                self.__update_one_tensor_info(tensor_info, next_state)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, __o: object) -> bool:
        return self is __o

    def __repr__(self, detailed: bool = True):
        output = [
            'Chunk Information:\n',
            '\tchunk size: {}, chunk dtype: {}, process group size: {}\n'.format(self.chunk_size, self.dtype,
                                                                                 self.pg_size),
            '\t# of tensors: {}, utilized size: {}, utilized percentage: {:.2f}\n'.format(
                self.num_tensors, self.utilized_size, self.utilized_size / self.chunk_size)
        ]

        def print_tensor(tensor, prefix=''):
            output.append('{}shape: {}, dtype: {}, device: {}\n'.format(prefix, tensor.shape, tensor.dtype,
                                                                        tensor.device))

        if self.chunk_temp is not None:
            output.append('\tchunk temp:\n')
            print_tensor(tensor=self.chunk_temp, prefix='\t\t')

        if self.cuda_global_chunk is not None and self.cuda_global_chunk.storage().size() > 0:
            output.append('\tchunk total:\n')
            print_tensor(tensor=self.cuda_global_chunk, prefix='\t\t')

        if self.cuda_shard is not None:
            output.append('\tcuda shard:\n')
            print_tensor(tensor=self.cuda_shard, prefix='\t\t')

        if self.cpu_shard is not None:
            output.append('\tcpu shard:\n')
            print_tensor(tensor=self.cpu_shard, prefix='\t\t')

        memory_info = self.memory_usage
        output.append('\tmemory usage: cuda {}, cpu {}\n'.format(memory_info['cuda'], memory_info['cpu']))

        if detailed:
            output.append('\ttensor state monitor:\n')
            for st in TensorState:
                output.append('\t\t# of {}: {}\n'.format(st, self.tensor_state_cnter[st]))

        return ''.join(output)
