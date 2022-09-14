use ckb_vm::instructions::{
    extract_opcode, tagged::TaggedInstruction, Instruction, InstructionOpcode, Utype,
};
use ckb_vm_definitions::instructions as opcodes;
use core::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct InstructionPrinter(pub TaggedInstruction);

impl InstructionPrinter {
    pub fn new(t: TaggedInstruction) -> Self {
        Self(t)
    }

    pub fn opcode(&self) -> InstructionOpcode {
        extract_opcode(self.0.clone().into())
    }

    pub fn instruction(&self) -> Instruction {
        match self.0 {
            TaggedInstruction::Rtype(i) => i.0,
            TaggedInstruction::Itype(i) => i.0,
            TaggedInstruction::Stype(i) => i.0,
            TaggedInstruction::Utype(i) => i.0,
            TaggedInstruction::R4type(i) => i.0,
        }
    }
}

impl fmt::Display for InstructionPrinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode() {
            opcodes::OP_LUI | opcodes::OP_AUIPC => {
                if let TaggedInstruction::Utype(i) = self.0 {
                    let shifted = Utype::new(i.op(), i.rd(), i.immediate_u() >> 12);
                    return shifted.fmt(f);
                }
                self.0.fmt(f)
            }
            opcodes::OP_ECALL => write!(f, "ecall"),
            opcodes::OP_EBREAK => write!(f, "ebreak"),
            opcodes::OP_FENCEI => write!(f, "fencei"),
            opcodes::OP_FENCE => {
                if let TaggedInstruction::Rtype(i) = self.0 {
                    if i.rd() == 0 && i.rs1() == 0b1111 && i.rs2() == 0b1111 {
                        return write!(f, "fence");
                    }
                }
                self.0.fmt(f)
            }
            _ => self.0.fmt(f),
        }
    }
}
