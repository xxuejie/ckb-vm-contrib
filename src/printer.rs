use ckb_vm::{
    ckb_vm_definitions::{instructions as opcodes, registers::REGISTER_ABI_NAMES},
    instructions::{
        extract_opcode, tagged::TaggedInstruction, Instruction, InstructionOpcode, Utype,
    },
};
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
            TaggedInstruction::R5type(i) => i.0,
        }
    }
}

impl fmt::Display for InstructionPrinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.opcode(), &self.0) {
            (opcodes::OP_ADDI, TaggedInstruction::Itype(i)) => {
                if i.rd() == 0 && i.rs1() == 0 && i.immediate_u() == 0 {
                    return write!(f, "nop");
                } else if i.rs1() == 0 && i.immediate_u() & 0xFFF == i.immediate_u() {
                    return write!(f, "li {},{}", register_name(i.rd()), i.immediate_s());
                } else if i.immediate_u() == 0 {
                    return write!(f, "mv {},{}", register_name(i.rd()), register_name(i.rs1()));
                }
            }
            (opcodes::OP_XORI, TaggedInstruction::Itype(i)) => {
                if i.immediate_s() == -1 {
                    return write!(
                        f,
                        "not {},{}",
                        register_name(i.rd()),
                        register_name(i.rs1())
                    );
                }
            }
            (opcodes::OP_SUB, TaggedInstruction::Rtype(i)) => {
                if i.rs1() == 0 {
                    return write!(
                        f,
                        "neg {},{}",
                        register_name(i.rd()),
                        register_name(i.rs2()),
                    );
                }
            }
            (opcodes::OP_SUBW, TaggedInstruction::Rtype(i)) => {
                if i.rs1() == 0 {
                    return write!(
                        f,
                        "negw {},{}",
                        register_name(i.rd()),
                        register_name(i.rs2()),
                    );
                }
            }
            (opcodes::OP_ADDIW, TaggedInstruction::Itype(i)) => {
                if i.immediate_u() == 0 {
                    return write!(
                        f,
                        "sext.w {},{}",
                        register_name(i.rd()),
                        register_name(i.rs1()),
                    );
                }
            }
            (opcodes::OP_ANDI, TaggedInstruction::Itype(i)) => {
                if i.immediate_u() == 255 {
                    return write!(
                        f,
                        "zext.b {},{}",
                        register_name(i.rd()),
                        register_name(i.rs1()),
                    );
                }
            }
            (opcodes::OP_SLTIU, TaggedInstruction::Itype(i)) => {
                if i.immediate_u() == 1 {
                    return write!(
                        f,
                        "seqz {},{}",
                        register_name(i.rd()),
                        register_name(i.rs1()),
                    );
                }
            }
            (opcodes::OP_SLTU, TaggedInstruction::Rtype(i)) => {
                if i.rs1() == 0 {
                    return write!(
                        f,
                        "snez {},{}",
                        register_name(i.rd()),
                        register_name(i.rs2()),
                    );
                }
            }
            (opcodes::OP_SLT, TaggedInstruction::Rtype(i)) => {
                if i.rs2() == 0 {
                    return write!(
                        f,
                        "sltz {},{}",
                        register_name(i.rd()),
                        register_name(i.rs1()),
                    );
                } else if i.rs1() == 0 {
                    return write!(
                        f,
                        "sgtz {},{}",
                        register_name(i.rd()),
                        register_name(i.rs2()),
                    );
                }
            }
            (opcodes::OP_BEQ, TaggedInstruction::Stype(i)) => {
                if i.rs2() == 0 && i.immediate_u() & 0x1FFE == i.immediate_u() {
                    return write!(f, "beqz {},{}", register_name(i.rs1()), i.immediate_s(),);
                }
            }
            (opcodes::OP_BNE, TaggedInstruction::Stype(i)) => {
                if i.rs2() == 0 && i.immediate_u() & 0x1FFE == i.immediate_u() {
                    return write!(f, "bnez {},{}", register_name(i.rs1()), i.immediate_s(),);
                }
            }
            (opcodes::OP_BGE, TaggedInstruction::Stype(i)) => {
                if i.immediate_u() & 0x1FFE == i.immediate_u() {
                    if i.rs1() == 0 {
                        return write!(f, "blez {},{}", register_name(i.rs2()), i.immediate_s(),);
                    } else if i.rs2() == 0 {
                        return write!(f, "bgez {},{}", register_name(i.rs1()), i.immediate_s(),);
                    }
                }
            }
            (opcodes::OP_BLT, TaggedInstruction::Stype(i)) => {
                if i.immediate_u() & 0x1FFE == i.immediate_u() {
                    if i.rs1() == 0 {
                        return write!(f, "bgtz {},{}", register_name(i.rs2()), i.immediate_s(),);
                    } else if i.rs2() == 0 {
                        return write!(f, "bltz {},{}", register_name(i.rs1()), i.immediate_s(),);
                    }
                }
            }
            (opcodes::OP_JAL, TaggedInstruction::Utype(i)) => {
                if i.rd() == 0 {
                    return write!(f, "j {}", i.immediate_s());
                } else if i.rd() == 1 {
                    return write!(f, "jal {}", i.immediate_s());
                }
            }
            (opcodes::OP_JALR_VERSION0, TaggedInstruction::Itype(i)) => {
                if i.rd() == 0 && i.rs1() == 1 && i.immediate_u() == 0 {
                    return write!(f, "ret");
                } else if i.rd() == 1 && i.immediate_u() == 0 {
                    return write!(f, "jalr {}", register_name(i.rs1()));
                }
            }
            (opcodes::OP_JALR_VERSION1, TaggedInstruction::Itype(i)) => {
                if i.rd() == 0 && i.rs1() == 1 && i.immediate_u() == 0 {
                    return write!(f, "ret");
                } else if i.rd() == 1 && i.immediate_u() == 0 {
                    return write!(f, "jalr {}", register_name(i.rs1()));
                }
            }
            (opcodes::OP_LUI | opcodes::OP_AUIPC, TaggedInstruction::Utype(i)) => {
                let shifted = Utype::new(i.op(), i.rd(), i.immediate_u() >> 12);
                return shifted.fmt(f);
            }
            (opcodes::OP_ECALL, _) => return write!(f, "ecall"),
            (opcodes::OP_EBREAK, _) => return write!(f, "ebreak"),
            (opcodes::OP_FENCEI, _) => return write!(f, "fencei"),
            (opcodes::OP_FENCE, TaggedInstruction::Rtype(i)) => {
                if i.rd() == 0 && i.rs1() == 0b1111 && i.rs2() == 0b1111 {
                    return write!(f, "fence");
                }
            }
            _ => (),
        };
        self.0.fmt(f)
    }
}

fn register_name(i: usize) -> String {
    if i < REGISTER_ABI_NAMES.len() {
        return REGISTER_ABI_NAMES[i].to_string();
    }
    format!("x{}", i)
}
