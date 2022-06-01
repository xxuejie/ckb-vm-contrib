use ckb_vm::instructions::{extract_opcode, tagged::TaggedInstruction, InstructionOpcode};
use ckb_vm_definitions::instructions as opcodes;
use core::fmt;

pub struct InstructionPrinter(pub TaggedInstruction);

impl InstructionPrinter {
    pub fn new(t: TaggedInstruction) -> Self {
        Self(t)
    }

    pub fn opcode(&self) -> InstructionOpcode {
        extract_opcode(self.0.clone().into())
    }
}

impl fmt::Display for InstructionPrinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode() {
            opcodes::OP_ECALL => write!(f, "ecall"),
            opcodes::OP_EBREAK => write!(f, "ebreak"),
            _ => self.0.fmt(f),
        }
    }
}
