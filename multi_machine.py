import numpy as np
from typing import Union

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
    
    @staticmethod
    def insn_wrapper(op_str, operands):
        """
        Wrapper function to call the instruction function with the operands.
        """
        def wrapper(instance):
            return getattr(instance, op_str)(*operands)
        return wrapper
    
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
            end = len(s.prrogram) * 4
        elif end is None: # stop after a given number of instructions
            end = start + instructions * 4
        else: # stop at the given end label or address
            end = s.look_up_label(end)
        while s.pc < end:
            # get the next instruction
            instruction = program[s.pc // 4]
            # execute the instruction. this also increments the prrogram counter appropriately
            instruction(s) # pass self.
        # done

    def reset_state(s):
        """
        Reset the machine to its initial state.
        """
        s.pc = 0
        s.clear_cpu()
        s.clear_mem()

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
    
    def exe(s, start:int, end=None, instructions=0):
        """
        Execute the program on all machines.
        """
        for machine in s.machines:
            machine.exe(start, end, instructions, s.program)
    
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
