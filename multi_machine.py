from time import time
from typing import Union

import numpy as np

from .machine import machine

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
    def __init__(s, mem_size):
        super().__init__(mem_size)
        s.program = []
        s.mem_usage = np.zeros(mem_size, dtype=np.int8)
        s.x_usage = np.zeros(32, dtype=np.int8)
        s.x_usage[0] = 1 # x0 is always used
        s.f_usage = np.zeros(32, dtype=np.int8)
    
    @staticmethod
    def insn_wrapper(op_str, operands):
        """
        Wrapper function to call the instruction function with the operands.
        """
        def wrapper(instance):
            instance._update_counters(
                op_str, operands[0], operands[1] if len(operands) > 2 else None) # mem is always at pos 3
            return getattr(instance, op_str)(*operands)
        return wrapper
    
    def _update_counters(s, opcode, rd, mem):
        """Update self.x_usage, self.f_usage and self.mem_usage counters."""
        # borrowed from super().dec()
        if opcode.lower() not in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'sb', 'sh', 'sw', 'fsw.s']:
            assert rd is not None, f"rd is None for opcode {opcode}"
            if opcode.lower() in ['fadd.s', 'fsub.s', 'fmul.s', 'fdiv.s', 'fsqrt.s', 'fmin.s', 'fmax.s',
                    'fmadd.s', 'fmsub.s', 'fnmadd.s', 'fnmsub.s', 'flw.s', 'fsgnj.s',
                    'fsgnjn.s', 'fsgnjx.s', 'fcvt.s.w', 'fcvt.s.wu', 'fmv.w.x']:
                s.f_usage[rd] = 1
            else:
                s.x_usage[rd] = 1
        if opcode.lower() in ['sb', 'sh', 'sw', 'fsw.s']:
            assert mem is not None, f"mem is None for opcode {opcode}"
            s.mem_usage[mem] = 1

    def append_instruction(s, opcode, operands):
        # TODO: check if the opcode is valid.
        # TODO: check signature of op_fn and make sure it is correct.
        # TODO: resolve labels in operands -- great quality of life improvement
        s.program.append(pseudo_asm_machine.insn_wrapper(opcode, operands))
    
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
            instruction = program[s.pc // 4]
            # execute the instruction. this also increments the program counter appropriately
            instruction(s) # pass self.
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
        s.mem_usage = np.zeros(s.mem.shape[0], dtype=np.int8)
    
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

class multi_machine(object):
    """
    A collection of machines that share the same program but not the same state (registers, memory, pc).
    """
    def __init__(s, mem_size, num_machines):
        """
        Initialize the multi-machine with the given number of machines and memory size.
        """
        s.machines = [pseudo_asm_machine(mem_size) for _ in range(num_machines)]
        s.num_machines = num_machines
        s.mem_size = mem_size
        s.program = []
    
    @staticmethod
    def insn_wrapper(op_str, operands):
        return pseudo_asm_machine.insn_wrapper(op_str, operands)
    
    def append_instruction(s, opcode, operands):
        """
        Append an instruction to the program of all machines.
        """
        s.program.append(multi_machine.insn_wrapper(opcode, operands))
    
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

    def set_memory(s, idx:int, inputs: Union[np.ndarray, int, float]):
        """
        Set the memory of the machine at index idx to the given inputs.
        """
        if idx < 0 or idx >= s.num_machines:
            raise ValueError(f"Invalid machine index: {idx}")
        s._populate_memory(idx, inputs)

    def _populate_memory(self, idx:int, inputs: Union[np.ndarray, int, float]):
        # overflow checks
        instance = self.machines[idx]
        instance.clear_mem()
        if isinstance(inputs, (np.ndarray, list)):
            if inputs.dtype == np.int32 or inputs.dtype == np.uint32 or inputs.dtype == np.int64:
                instance.write_i32_vec(inputs, 0)
            elif inputs.dtype == np.float32 or inputs.dtype == np.float64:
                instance.write_f32_vec(inputs, 0)
            else:
                raise ValueError(f"Unsupported dtype: {inputs.dtype}")
        elif isinstance(inputs, int):
            instance.write_i32(inputs, 0)
        elif isinstance(inputs, float):
            instance.write_f32(inputs, 0)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

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
    
    def clone(self):
        # takes care of creating the new machine instances
        clone = multi_machine(self.mem_size, self.num_machines)
        # copy the program
        clone.program = self.program.copy()
        # copy the state of each machine
        for i in range(self.num_machines):
            c = clone.machines[i]
            s = self.machines[i]
            c.mem =            s.mem.copy()
            c.x =              s.x.copy()
            c.f =              s.f.copy()
            c.pc =             s.pc
            c.label_dict =     s.label_dict.copy()
            c.ops =            s.ops.copy()
            c.x_usage =        s.x_usage.copy()
            c.f_usage =        s.f_usage.copy()
            c.mem_usage =      s.mem_usage.copy()
        return clone
