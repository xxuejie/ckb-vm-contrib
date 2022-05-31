use super::utils::*;
use ckb_vm::{instructions::tagged::TaggedInstruction, Register};
use ckb_vm_definitions::instructions as opcodes;

pub fn assembler<R: Register>(inst: &TaggedInstruction) -> Option<Vec<u8>> {
    let i = match inst {
        TaggedInstruction::Rtype(i) => i,
        _ => return None,
    };

    match i.op() {
        opcodes::OP_MUL => Some(funct3(0b000) | opcode(0b0110011)),
        opcodes::OP_MULH => Some(funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_MULHSU => Some(funct3(0b010) | opcode(0b0110011)),
        opcodes::OP_MULHU => Some(funct3(0b011) | opcode(0b0110011)),
        opcodes::OP_DIV => Some(funct3(0b100) | opcode(0b0110011)),
        opcodes::OP_DIVU => Some(funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_REM => Some(funct3(0b110) | opcode(0b0110011)),
        opcodes::OP_REMU => Some(funct3(0b111) | opcode(0b0110011)),
        opcodes::OP_MULW => Some(funct3(0b000) | opcode(0b0111011)),
        opcodes::OP_DIVW => Some(funct3(0b100) | opcode(0b0111011)),
        opcodes::OP_DIVUW => Some(funct3(0b101) | opcode(0b0111011)),
        opcodes::OP_REMW => Some(funct3(0b110) | opcode(0b0111011)),
        opcodes::OP_REMUW => Some(funct3(0b111) | opcode(0b0111011)),
        _ => None,
    }
    .map(|packed| packed | funct7(0b0000001) | rs2(i.rs2()) | rs1(i.rs1()) | rd(i.rd()))
    .map(|packed_inst| packed_inst.to_le_bytes().to_vec())
}
