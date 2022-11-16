use super::utils::*;
use ckb_vm::{
    ckb_vm_definitions::instructions as opcodes,
    instructions::{tagged::TaggedInstruction, Itype, Rtype},
    Register,
};

pub fn assembler<R: Register>(inst: &TaggedInstruction) -> Option<Vec<u8>> {
    match inst {
        TaggedInstruction::Itype(i) => assemble_itype::<R>(i),
        TaggedInstruction::Rtype(i) => assemble_rtype::<R>(i),
        _ => None,
    }
    .map(|packed_inst| packed_inst.to_le_bytes().to_vec())
}

fn assemble_itype<R: Register>(i: &Itype) -> Option<u32> {
    match i.op() {
        opcodes::OP_BCLRI => Some(funct7(0b0100100) | funct3(0b001) | opcode(0b0010011)),
        opcodes::OP_BEXTI => Some(funct7(0b0100100) | funct3(0b101) | opcode(0b0010011)),
        opcodes::OP_BINVI => Some(funct7(0b0110100) | funct3(0b001) | opcode(0b0010011)),
        opcodes::OP_BSETI => Some(funct7(0b0010100) | funct3(0b001) | opcode(0b0010011)),
        opcodes::OP_RORI => Some(funct7(0b0110000) | funct3(0b101) | opcode(0b0010011)),
        opcodes::OP_RORIW => Some(funct7(0b0110000) | funct3(0b101) | opcode(0b0011011)),
        opcodes::OP_SLLIUW => Some(funct7(0b0000100) | funct3(0b001) | opcode(0b0011011)),
        _ => None,
    }
    .map(|packed| packed | shamt::<R>(i.immediate_s()) | rs1(i.rs1()) | rd(i.rd()))
}

fn assemble_rtype<R: Register>(i: &Rtype) -> Option<u32> {
    match i.op() {
        opcodes::OP_ADDUW => Some(funct7(0b0000100) | funct3(0b000) | opcode(0b0111011)),
        opcodes::OP_ROLW => Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0111011)),
        opcodes::OP_SH1ADDUW => Some(funct7(0b0010000) | funct3(0b010) | opcode(0b0111011)),
        opcodes::OP_SH2ADDUW => Some(funct7(0b0010000) | funct3(0b100) | opcode(0b0111011)),
        opcodes::OP_SH3ADDUW => Some(funct7(0b0010000) | funct3(0b110) | opcode(0b0111011)),
        opcodes::OP_RORW => Some(funct7(0b0110000) | funct3(0b101) | opcode(0b0111011)),
        opcodes::OP_ANDN => Some(funct7(0b0100000) | funct3(0b111) | opcode(0b0110011)),
        opcodes::OP_ORN => Some(funct7(0b0100000) | funct3(0b110) | opcode(0b0110011)),
        opcodes::OP_XNOR => Some(funct7(0b0100000) | funct3(0b100) | opcode(0b0110011)),
        opcodes::OP_ROL => Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_ROR => Some(funct7(0b0110000) | funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_BINV => Some(funct7(0b0110100) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_BSET => Some(funct7(0b0010100) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_BCLR => Some(funct7(0b0100100) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_BEXT => Some(funct7(0b0100100) | funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_SH1ADD => Some(funct7(0b0010000) | funct3(0b010) | opcode(0b0110011)),
        opcodes::OP_SH2ADD => Some(funct7(0b0010000) | funct3(0b100) | opcode(0b0110011)),
        opcodes::OP_SH3ADD => Some(funct7(0b0010000) | funct3(0b110) | opcode(0b0110011)),
        opcodes::OP_CLMUL => Some(funct7(0b0000101) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_CLMULH => Some(funct7(0b0000101) | funct3(0b011) | opcode(0b0110011)),
        opcodes::OP_CLMULR => Some(funct7(0b0000101) | funct3(0b010) | opcode(0b0110011)),
        opcodes::OP_MIN => Some(funct7(0b0000101) | funct3(0b100) | opcode(0b0110011)),
        opcodes::OP_MINU => Some(funct7(0b0000101) | funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_MAX => Some(funct7(0b0000101) | funct3(0b110) | opcode(0b0110011)),
        opcodes::OP_MAXU => Some(funct7(0b0000101) | funct3(0b111) | opcode(0b0110011)),
        _ => None,
    }
    .map(|packed| packed | rs2(i.rs2()) | rs1(i.rs1()) | rd(i.rd()))
    .or_else(|| {
        match i.op() {
            opcodes::OP_ORCB => {
                Some(funct7(0b0010100) | funct3(0b101) | opcode(0b0010011) | rs2(0b00111))
            }
            opcodes::OP_REV8 => {
                Some(funct7(0b0110101) | funct3(0b101) | opcode(0b0010011) | rs2(0b11000))
            }
            opcodes::OP_CLZ => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0010011) | rs2(0b00000))
            }
            opcodes::OP_CPOP => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0010011) | rs2(0b00010))
            }
            opcodes::OP_CTZ => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0010011) | rs2(0b00001))
            }
            opcodes::OP_SEXTB => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0010011) | rs2(0b00100))
            }
            opcodes::OP_SEXTH => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0010011) | rs2(0b00101))
            }
            opcodes::OP_CLZW => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0011011) | rs2(0b00000))
            }
            opcodes::OP_CPOPW => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0011011) | rs2(0b00010))
            }
            opcodes::OP_CTZW => {
                Some(funct7(0b0110000) | funct3(0b001) | opcode(0b0011011) | rs2(0b00001))
            }
            _ => None,
        }
        .map(|packed| packed | rs1(i.rs1()) | rd(i.rd()))
    })
    .or_else(|| {
        if i.op() == opcodes::OP_ZEXTH && i.rs2() == 0 {
            let c = if R::BITS == 64 { 0b0111011 } else { 0b0110011 };
            return Some(funct7(0b0000100) | funct3(0b100) | opcode(c) | rs1(i.rs1()) | rd(i.rd()));
        }
        None
    })
}
