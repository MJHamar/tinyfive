from time import time
from typing import Union

import numpy as np

from .machine import machine

import logging
logger = logging.getLogger(__name__)

class pseudo_asm_machine(machine):
    """
    Implementation of the machine class without the need for assembling and disassembling the program.
    Fields: 
    - <all fields of the machine class>
    - program -- a list of parameterized uppercase instructions, pupulated by the append_instruction method
    
    Redefines:
    - exe -- instead of disassembling the program, it uses the above program list and selects the next instruction
                based on the program counter.
    """
    rd_opcodes = {'beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'sb', 'sh', 'sw', 'fsw.s'}
    float_opcodes = {'fadd.s', 'fsub.s', 'fmul.s', 'fdiv.s', 'fsqrt.s', 'fmin.s', 'fmax.s',
                    'fmadd.s', 'fmsub.s', 'fnmadd.s', 'fnmsub.s', 'flw.s', 'fsgnj.s',
                    'fsgnjn.s', 'fsgnjx.s', 'fcvt.s.w', 'fcvt.s.wu', 'fmv.w.x'}
    store_opcodes = {'sb', 'sh', 'sw', 'fsw.s'}
    
    def __init__(s, mem_size, initial_state=None):
        super().__init__(mem_size)
        s.program = []
        s.mem_usage = np.zeros(mem_size//4, dtype=np.int8)
        s.x_usage = np.zeros(32, dtype=np.int8)
        s.x_usage[0] = 1 # x0 is always used
        s.f_usage = np.zeros(32, dtype=np.int8)
        s.init_mem = None
        if initial_state is not None:
            # write the initial state to the memory
            s.write_i32_vec(initial_state, 0)
            # update the memory usage
            s.mem_usage[:len(initial_state)//4] = 1
            # cache the initial state
            s.init_mem = (s.mem.copy(), s.mem_usage.copy())
    
    def _update_counters(s, opcode, rd, mem):
        """Update self.x_usage, self.f_usage and self.mem_usage counters."""
        # borrowed from super().dec()
        if opcode.lower() not in s.rd_opcodes:
            assert rd is not None, f"rd is None for opcode {opcode}"
            if opcode.lower() in s.float_opcodes:
                s.f_usage[rd] = 1
            else:
                s.x_usage[rd] = 1
        if opcode.lower() in s.float_opcodes:
            assert mem is not None, f"mem is None for opcode {opcode}"
            s.mem_usage[mem//4] = 1 # NOTE: assumes 4 byte alignment

    def append_instruction(s, opcode, operands):
        # TODO: check if the opcode is valid.
        # TODO: check signature of op_fn and make sure it is correct.
        # TODO: resolve labels in operands -- great quality of life improvement
        s.program.append((opcode, operands))
    
    def exe(s, start=None, end=None, instructions=0, program=None):
        """
        Execute given program or the stored program if unspecified.
        """
        if program is None:
            program = s.program
        # reset the program counter
        if start is None:
            start = 0
        start = s.look_up_label(start)
        s.pc = start
        
        # NOTE: s.pc is always incremented by multiples of 4
        if end is None and not instructions: # execute until the end by default
            end = len(program) * 4
        elif end is None: # stop after a given number of instructions
            end = start + instructions * 4
        else: # stop at the given end label or address
            end = s.look_up_label(end)
        while s.pc < end:
            # get the next instruction
            opcode, operands = program[s.pc // 4]
            # execute the instruction. this also increments the program counter appropriately
            s._update_counters(
                opcode, operands[0], operands[1] if len(operands) > 2 else None) # mem is always at pos 3
            getattr(s, opcode)(*operands)
        # done

    def measure_latency(s):
        """
        Measure the latency of the program.
        """
        start = time()
        s.exe()
        end = time()
        return float(end - start)

    def reset_state(s):
        """
        Reset the machine to its initial state.
        """
        s.pc = 0
        s.clear_cpu()
        s.clear_mem()
        s.x_usage = np.zeros(32, dtype=np.int8)
        s.x_usage[0] = 1
        s.f_usage = np.zeros(32, dtype=np.int8)
        s.mem_usage = np.zeros(s.mem.shape[0]//4, dtype=np.int8)
        if s.init_mem is not None:
            s.mem, s.mem_usage = s.init_mem[0].copy(), s.init_mem[1].copy()
    
    @property
    def registers(self):
        return np.concatenate([self.x, self.f], axis=0)
    @property
    def memory(self):
        return self.read_i32_vec(0, self.mem.shape[0]//4)
    @property
    def register_mask(self):
        return np.concatenate([self.x_usage, self.f_usage], axis=0)
    @property
    def memory_mask(self):
        return self.mem_usage

    def clone(s):
        c = object.__new__(pseudo_asm_machine)
        c.mem =            s.mem.copy()
        c.x =              s.x.copy()
        c.f =              s.f.copy()
        c.pc =             s.pc
        c.label_dict =     s.label_dict.copy()
        c.ops =            s.ops.copy()
        c.x_usage =        s.x_usage.copy()
        c.f_usage =        s.f_usage.copy()
        c.mem_usage =      s.mem_usage.copy()
        c.init_mem =       (s.init_mem[0].copy(), # memory and memory usage
                            s.init_mem[1].copy()) if\
                                s.init_mem is not None else None
        c.program =        s.program.copy() if s.program is not None else None
        return c

class multi_machine(object):
    """
    A collection of machines that share the same program but not the same state (registers, memory, pc).
    """
    def __init__(s, mem_size, num_machines, initial_state=None):
        """
        Initialize the multi-machine with the given number of machines and memory size.
        """
        assert (initial_state is None or
                initial_state.shape[0] == num_machines and
                len(initial_state.shape) == 2), \
            f"Expected initial_state to be of shape ({num_machines}, N), got {initial_state.shape}"
        s.machines = []
        for i in range(num_machines):
            # create a new machine instance
            s_init = None if initial_state is None else initial_state[i]
            s.machines.append(pseudo_asm_machine(mem_size, s_init))
        s.num_machines = num_machines
        s.mem_size = mem_size
        s.program = []
    
    def append_instruction(s, opcode, operands):
        """
        Append an instruction to the program of all machines.
        """
        s.program.append((opcode, operands))
    
    def exe(s, start:int=0, end=None, instructions=0):
        """
        Execute the program on all machines.
        """
        for machine in s.machines:
            machine.exe(start, end, instructions, s.program)
    
    def measure_latency(s):
        """
        Measure the latency of the program on all machines.
        """
        latencies = []
        for machine in s.machines:
            latencies.append(machine.measure_latency())
        return latencies
    
    def reset_state(s):
        """
        Reset all machines to their initial state.
        """
        for machine in s.machines:
            machine.reset_state()

    def set_memory(s, inputs: np.ndarray):
        """
        Set the memory of the machine at index idx to the given inputs.
        """
        s._populate_memory(inputs)

    def _populate_memory(self, inputs: np.ndarray):
        # overflow checks
        # inputs is a 2D array of shape (num_machines, input_size)
        if inputs.shape[0] != self.num_machines:
            raise ValueError(f"Expected {self.num_machines} inputs, got {inputs.shape[0]}")
        for i in range(self.num_machines):
            instance = self.machines[i]
            instance.clear_mem()
            
            inputs_i = inputs[i]
            instance.write_i32_vec(inputs_i, 0)

    @property
    def registers(self):
        return np.stack([m.registers for m in self.machines])
    @property
    def memory(self):
        return np.stack([m.memory for m in self.machines])
    @property
    def register_mask(self):
        return np.stack([m.register_mask for m in self.machines])
    @property
    def memory_mask(self):
        return np.stack([m.memory_mask for m in self.machines])
    @property
    def program_counter(self):
        return np.stack([m.pc // 4 for m in self.machines])
    
    def clone(self):
        clone = object.__new__(multi_machine)
        clone.num_machines = self.num_machines
        clone.mem_size = self.mem_size
        clone.program = self.program.copy()
        clone.machines = [
            m.clone() for m in self.machines
        ]
        return clone
