use super::utils::*;
use ckb_vm::{
    ckb_vm_definitions::instructions as opcodes,
    instructions::{tagged::TaggedInstruction, Itype, Rtype, Stype, Utype},
    Register,
};

pub fn assembler<R: Register>(inst: &TaggedInstruction) -> Option<Vec<u8>> {
    match inst {
        TaggedInstruction::Itype(i) => assemble_itype::<R>(i),
        TaggedInstruction::Rtype(i) => assemble_rtype::<R>(i),
        TaggedInstruction::Utype(i) => assemble_utype::<R>(i),
        TaggedInstruction::Stype(i) => assemble_stype::<R>(i),
        _ => None,
    }
    .map(|packed_inst| packed_inst.to_le_bytes().to_vec())
}

fn assemble_itype<R: Register>(i: &Itype) -> Option<u32> {
    match i.op() {
        opcodes::OP_LB => Some(funct3(0b000) | opcode(0b0000011)),
        opcodes::OP_LH => Some(funct3(0b001) | opcode(0b0000011)),
        opcodes::OP_LW => Some(funct3(0b010) | opcode(0b0000011)),
        opcodes::OP_LD => Some(funct3(0b011) | opcode(0b0000011)),
        opcodes::OP_LBU => Some(funct3(0b100) | opcode(0b0000011)),
        opcodes::OP_LHU => Some(funct3(0b101) | opcode(0b0000011)),
        opcodes::OP_LWU => Some(funct3(0b110) | opcode(0b0000011)),
        opcodes::OP_ADDI => Some(funct3(0b000) | opcode(0b0010011)),
        opcodes::OP_SLTI => Some(funct3(0b010) | opcode(0b0010011)),
        opcodes::OP_SLTIU => Some(funct3(0b011) | opcode(0b0010011)),
        opcodes::OP_XORI => Some(funct3(0b100) | opcode(0b0010011)),
        opcodes::OP_ORI => Some(funct3(0b110) | opcode(0b0010011)),
        opcodes::OP_ANDI => Some(funct3(0b111) | opcode(0b0010011)),
        opcodes::OP_ADDIW => Some(funct3(0b000) | opcode(0b0011011)),
        opcodes::OP_JALR => Some(funct3(0b000) | opcode(0b1100111)),
        _ => None,
    }
    .map(|packed| packed | itype_immediate(i.immediate_s()) | rs1(i.rs1()) | rd(i.rd()))
    .or_else(|| {
        match i.op() {
            opcodes::OP_SLLI => Some(funct3(0b001) | opcode(0b0010011) | funct7(0b0000000)),
            opcodes::OP_SRLI => Some(funct3(0b101) | opcode(0b0010011) | funct7(0b0000000)),
            opcodes::OP_SRAI => Some(funct3(0b101) | opcode(0b0010011) | funct7(0b0100000)),
            _ => None,
        }
        .map(|packed| packed | shamt::<R>(i.immediate_s()) | rs1(i.rs1()) | rd(i.rd()))
    })
    .or_else(|| {
        match i.op() {
            opcodes::OP_SLLIW => Some(funct3(0b001) | opcode(0b0011011) | funct7(0b0000000)),
            opcodes::OP_SRLIW => Some(funct3(0b101) | opcode(0b0011011) | funct7(0b0000000)),
            opcodes::OP_SRAIW => Some(funct3(0b101) | opcode(0b0011011) | funct7(0b0100000)),
            _ => None,
        }
        .map(|packed| packed | shamtw::<R>(i.immediate_s()) | rs1(i.rs1()) | rd(i.rd()))
    })
}

fn assemble_rtype<R: Register>(i: &Rtype) -> Option<u32> {
    match i.op() {
        opcodes::OP_ADD => Some(funct7(0b0000000) | funct3(0b000) | opcode(0b0110011)),
        opcodes::OP_SUB => Some(funct7(0b0100000) | funct3(0b000) | opcode(0b0110011)),
        opcodes::OP_SLL => Some(funct7(0b0000000) | funct3(0b001) | opcode(0b0110011)),
        opcodes::OP_SLT => Some(funct7(0b0000000) | funct3(0b010) | opcode(0b0110011)),
        opcodes::OP_SLTU => Some(funct7(0b0000000) | funct3(0b011) | opcode(0b0110011)),
        opcodes::OP_XOR => Some(funct7(0b0000000) | funct3(0b100) | opcode(0b0110011)),
        opcodes::OP_SRL => Some(funct7(0b0000000) | funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_SRA => Some(funct7(0b0100000) | funct3(0b101) | opcode(0b0110011)),
        opcodes::OP_OR => Some(funct7(0b0000000) | funct3(0b110) | opcode(0b0110011)),
        opcodes::OP_AND => Some(funct7(0b0000000) | funct3(0b111) | opcode(0b0110011)),
        opcodes::OP_ADDW => Some(funct7(0b0000000) | funct3(0b000) | opcode(0b0111011)),
        opcodes::OP_SUBW => Some(funct7(0b0100000) | funct3(0b000) | opcode(0b0111011)),
        opcodes::OP_SLLW => Some(funct7(0b0000000) | funct3(0b001) | opcode(0b0111011)),
        opcodes::OP_SRLW => Some(funct7(0b0000000) | funct3(0b101) | opcode(0b0111011)),
        opcodes::OP_SRAW => Some(funct7(0b0100000) | funct3(0b101) | opcode(0b0111011)),
        opcodes::OP_FENCE => Some(funct3(0b000) | opcode(0b0001111)),
        opcodes::OP_ECALL => return Some(0x73),
        opcodes::OP_EBREAK => return Some(0x100073),
        _ => None,
    }
    .map(|packed| packed | rs2(i.rs2()) | rs1(i.rs1()) | rd(i.rd()))
}

fn assemble_utype<R: Register>(i: &Utype) -> Option<u32> {
    match i.op() {
        opcodes::OP_LUI => Some(utype_immediate(i.immediate_s()) | opcode(0b0110111)),
        opcodes::OP_AUIPC => Some(utype_immediate(i.immediate_s()) | opcode(0b0010111)),
        opcodes::OP_JAL => Some(jtype_immediate(i.immediate_s()) | opcode(0b1101111)),
        _ => None,
    }
    .map(|packed| packed | rd(i.rd()))
}

fn assemble_stype<R: Register>(i: &Stype) -> Option<u32> {
    match i.op() {
        opcodes::OP_BEQ => Some(funct3(0b000) | opcode(0b1100011)),
        opcodes::OP_BNE => Some(funct3(0b001) | opcode(0b1100011)),
        opcodes::OP_BLT => Some(funct3(0b100) | opcode(0b1100011)),
        opcodes::OP_BGE => Some(funct3(0b101) | opcode(0b1100011)),
        opcodes::OP_BLTU => Some(funct3(0b110) | opcode(0b1100011)),
        opcodes::OP_BGEU => Some(funct3(0b111) | opcode(0b1100011)),
        _ => None,
    }
    .map(|packed| packed | btype_immediate(i.immediate_s()) | rs1(i.rs1()) | rs2(i.rs2()))
    .or_else(|| {
        match i.op() {
            opcodes::OP_SB => Some(funct3(0b000) | opcode(0b0100011)),
            opcodes::OP_SH => Some(funct3(0b001) | opcode(0b0100011)),
            opcodes::OP_SW => Some(funct3(0b010) | opcode(0b0100011)),
            opcodes::OP_SD => Some(funct3(0b011) | opcode(0b0100011)),
            _ => None,
        }
        .map(|packed| packed | stype_immediate(i.immediate_s()) | rs1(i.rs1()) | rs2(i.rs2()))
    })
}
