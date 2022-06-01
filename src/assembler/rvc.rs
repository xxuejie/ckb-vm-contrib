use ckb_vm::{
    instructions::{tagged::TaggedInstruction, Itype, Rtype, Stype, Utype},
    Register,
};
use ckb_vm_definitions::instructions as opcodes;

pub fn assembler<R: Register>(inst: &TaggedInstruction) -> Option<Vec<u8>> {
    match inst {
        TaggedInstruction::Itype(i) => assemble_itype::<R>(i),
        TaggedInstruction::Stype(i) => assemble_stype::<R>(i),
        TaggedInstruction::Utype(i) => assemble_utype::<R>(i),
        TaggedInstruction::Rtype(i) => assemble_rtype::<R>(i),
        _ => None,
    }
    .map(|packed_inst| packed_inst.to_le_bytes().to_vec())
}

fn assemble_itype<R: Register>(i: &Itype) -> Option<u16> {
    match i.op() {
        opcodes::OP_ADDI => {
            if i.rd() == i.rs1() && i.rd() != 0 {
                if i.rd() == 2 && in_range(i.immediate_s(), 4, 9, true, false) {
                    // C.ADDI16SP
                    return Some(
                        funct4(0b011)
                            | addi16sp_immediate(i.immediate_s())
                            | full_register(2, 7)
                            | op(0b01),
                    );
                } else if in_range(i.immediate_s(), 0, 5, true, false) {
                    // C.ADDI
                    return Some(
                        funct4(0b000)
                            | addi_immediate(i.immediate_s())
                            | full_register(i.rs1(), 7)
                            | op(0b01),
                    );
                }
            } else if i.rs1() == 2
                && in_range(i.immediate_s(), 2, 9, false, false)
                && is_compact_register(i.rd())
            {
                // C.ADDI4SPN
                return Some(
                    funct4(0b000)
                        | addi4spn_immediate(i.immediate_s())
                        | compact_register(i.rd(), 2)
                        | op(0b00),
                );
            } else if i.rs1() == 0 && i.rd() != 0 && in_range(i.immediate_s(), 0, 5, true, true) {
                // C.LI
                return Some(
                    funct4(0b010)
                        | addi_immediate(i.immediate_s())
                        | full_register(i.rd(), 7)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_ADDIW => {
            if R::BITS > 32
                && i.rd() == i.rs1()
                && i.rd() != 0
                && in_range(i.immediate_s(), 0, 5, true, true)
            {
                // C.ADDIW
                return Some(
                    funct4(0b001)
                        | addi_immediate(i.immediate_s())
                        | full_register(i.rs1(), 7)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_LW => {
            if is_compact_register(i.rd())
                && is_compact_register(i.rs1())
                && in_range(i.immediate_s(), 2, 6, false, true)
            {
                // C.LW
                return Some(
                    funct4(0b010)
                        | lw_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | compact_register(i.rd(), 2)
                        | op(0b00),
                );
            } else if i.rd() != 0 && i.rs1() == 2 && in_range(i.immediate_s(), 2, 7, false, true) {
                // C.LWSP
                return Some(
                    funct4(0b010)
                        | lwsp_immediate(i.immediate_s())
                        | full_register(i.rd(), 7)
                        | op(0b10),
                );
            }
        }
        opcodes::OP_LD => {
            if R::BITS > 32
                && is_compact_register(i.rd())
                && is_compact_register(i.rs1())
                && in_range(i.immediate_s(), 3, 7, false, true)
            {
                // C.LD
                return Some(
                    funct4(0b011)
                        | ld_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | compact_register(i.rd(), 2)
                        | op(0b00),
                );
            } else if R::BITS > 32
                && i.rd() != 0
                && i.rs1() == 2
                && in_range(i.immediate_s(), 3, 8, false, true)
            {
                // C.LDSP
                return Some(
                    funct4(0b011)
                        | ldsp_immediate(i.immediate_s())
                        | full_register(i.rd(), 7)
                        | op(0b10),
                );
            }
        }
        opcodes::OP_ANDI => {
            if i.rs1() == i.rd()
                && is_compact_register(i.rd())
                && in_range(i.immediate_s(), 0, 5, true, true)
            {
                // C.ANDI
                return Some(
                    funct4(0b100)
                        | addi_immediate(i.immediate_s())
                        | (0b10 << 10)
                        | compact_register(i.rd(), 7)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_JALR => {
            if i.rd() == 0 && i.rs1() != 0 && i.immediate_s() == 0 {
                // C.JR
                return Some(funct4(0b100) | full_register(i.rs1(), 7) | op(0b10));
            } else if i.rd() == 1 && i.rs1() != 0 && i.immediate_s() == 0 {
                // C.JALR
                return Some(funct4(0b100) | (0b1 << 12) | full_register(i.rs1(), 7) | op(0b10));
            }
        }
        _ => (),
    };
    None
}

fn assemble_stype<R: Register>(i: &Stype) -> Option<u16> {
    match i.op() {
        opcodes::OP_SW => {
            if is_compact_register(i.rs1())
                && is_compact_register(i.rs2())
                && in_range(i.immediate_s(), 2, 6, false, true)
            {
                // C.SW
                return Some(
                    funct4(0b110)
                        | lw_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | compact_register(i.rs2(), 2)
                        | op(0b00),
                );
            } else if i.rs1() == 2 && in_range(i.immediate_s(), 2, 7, false, true) {
                // C.SWSP
                return Some(
                    funct4(0b110)
                        | swsp_immediate(i.immediate_s())
                        | full_register(i.rs2(), 2)
                        | op(0b10),
                );
            }
        }
        opcodes::OP_SD => {
            if R::BITS > 32
                && is_compact_register(i.rs1())
                && is_compact_register(i.rs2())
                && in_range(i.immediate_s(), 3, 7, false, true)
            {
                // C.SD
                return Some(
                    funct4(0b111)
                        | ld_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | compact_register(i.rs2(), 2)
                        | op(0b00),
                );
            } else if R::BITS > 32 && i.rs1() == 2 && in_range(i.immediate_s(), 3, 8, false, true) {
                // C.SDSP
                return Some(
                    funct4(0b111)
                        | sdsp_immediate(i.immediate_s())
                        | full_register(i.rs2(), 2)
                        | op(0b10),
                );
            }
        }
        opcodes::OP_BEQ => {
            if is_compact_register(i.rs1())
                && i.rs2() == 0
                && in_range(i.immediate_s(), 1, 8, true, true)
            {
                // C.BEQZ
                return Some(
                    funct4(0b110)
                        | beqz_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_BNE => {
            if is_compact_register(i.rs1())
                && i.rs2() == 0
                && in_range(i.immediate_s(), 1, 8, true, true)
            {
                // C.BNEZ
                return Some(
                    funct4(0b111)
                        | beqz_immediate(i.immediate_s())
                        | compact_register(i.rs1(), 7)
                        | op(0b01),
                );
            }
        }
        _ => (),
    };
    None
}

fn assemble_utype<R: Register>(i: &Utype) -> Option<u16> {
    match i.op() {
        opcodes::OP_JAL => {
            if R::BITS == 32 && i.rd() == 1 && in_range(i.immediate_s(), 1, 11, true, true) {
                // C.JAL
                return Some(funct4(0b001) | jal_immediate(i.immediate_s()) | op(0b01));
            } else if i.rd() == 0 && in_range(i.immediate_s(), 1, 11, true, true) {
                // C.J
                return Some(funct4(0b101) | jal_immediate(i.immediate_s()) | op(0b01));
            }
        }
        opcodes::OP_LUI => {
            if i.rd() != 0 && i.rd() != 2 && in_range(i.immediate_s(), 12, 17, true, false) {
                // C.LUI
                return Some(
                    funct4(0b011)
                        | lui_immediate(i.immediate_s())
                        | full_register(i.rd(), 7)
                        | op(0b01),
                );
            }
        }
        _ => (),
    };
    None
}

fn assemble_rtype<R: Register>(i: &Rtype) -> Option<u16> {
    match i.op() {
        opcodes::OP_ADD => {
            if i.rd() != 0 && i.rs2() != 0 {
                if i.rs1() == 0 {
                    // C.MV
                    return Some(
                        funct4(0b100)
                            | full_register(i.rd(), 7)
                            | full_register(i.rs2(), 2)
                            | op(0b10),
                    );
                } else if i.rs1() == i.rd() {
                    // C.ADD
                    return Some(
                        funct4(0b100)
                            | (0b1 << 12)
                            | full_register(i.rd(), 7)
                            | full_register(i.rs2(), 2)
                            | op(0b10),
                    );
                }
            }
        }
        opcodes::OP_SUB => {
            if i.rs1() == i.rd() && is_compact_register(i.rd()) && is_compact_register(i.rs2()) {
                // C.SUB
                return Some(
                    funct4(0b100)
                        | (0b011 << 10)
                        | compact_register(i.rd(), 7)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_XOR => {
            if i.rs1() == i.rd() && is_compact_register(i.rd()) && is_compact_register(i.rs2()) {
                // C.XOR
                return Some(
                    funct4(0b100)
                        | (0b011 << 10)
                        | compact_register(i.rd(), 7)
                        | (0b01 << 5)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_OR => {
            if i.rs1() == i.rd() && is_compact_register(i.rd()) && is_compact_register(i.rs2()) {
                // C.OR
                return Some(
                    funct4(0b100)
                        | (0b011 << 10)
                        | compact_register(i.rd(), 7)
                        | (0b10 << 5)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_AND => {
            if i.rs1() == i.rd() && is_compact_register(i.rd()) && is_compact_register(i.rs2()) {
                // C.AND
                return Some(
                    funct4(0b100)
                        | (0b011 << 10)
                        | compact_register(i.rd(), 7)
                        | (0b11 << 5)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_SUBW => {
            if R::BITS > 32
                && i.rs1() == i.rd()
                && is_compact_register(i.rd())
                && is_compact_register(i.rs2())
            {
                // C.SUBW
                return Some(
                    funct4(0b100)
                        | (0b111 << 10)
                        | compact_register(i.rd(), 7)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_ADDW => {
            if R::BITS > 32
                && i.rs1() == i.rd()
                && is_compact_register(i.rd())
                && is_compact_register(i.rs2())
            {
                // C.ADDW
                return Some(
                    funct4(0b100)
                        | (0b111 << 10)
                        | compact_register(i.rd(), 7)
                        | (0b01 << 5)
                        | compact_register(i.rs2(), 2)
                        | op(0b01),
                );
            }
        }
        opcodes::OP_EBREAK => {
            // C.EBREAK
            return Some(funct4(0b100) | (1 << 12) | op(0b10));
        }
        _ => (),
    };
    None
}

fn in_range(value: i32, lower: usize, higher: usize, signed: bool, can_be_zero: bool) -> bool {
    if (!can_be_zero) && value == 0 {
        return false;
    }
    let value = value as u32;
    let right_mask = (1 << lower) - 1;
    if value & right_mask != 0 {
        return false;
    }
    if signed {
        let left_minus_mask = !((1 << higher) - 1);
        value & left_minus_mask == left_minus_mask || value & left_minus_mask == 0
    } else {
        let left_mask = !((1 << (higher + 1)) - 1);
        value & left_mask == 0 
    }
}

#[inline(always)]
fn addi4spn_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 4) & 0b11) << 11)
        | (((i >> 6) & 0b1111) << 7)
        | (((i >> 2) & 0b1) << 6)
        | (((i >> 3) & 0b1) << 5)
}

#[inline(always)]
fn addi16sp_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 9) & 0b1) << 12)
        | (((i >> 4) & 0b1) << 6)
        | (((i >> 6) & 0b1) << 5)
        | (((i >> 7) & 0b11) << 3)
        | (((i >> 5) & 0b1) << 2)
}

#[inline(always)]
fn addi_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 5) & 0b1) << 12) | ((i & 0b1_1111) << 2)
}

#[inline(always)]
fn lw_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 3) & 0b111) << 10) | (((i >> 2) & 0b1) << 6) | (((i >> 6) & 0b1) << 5)
}

#[inline(always)]
fn ld_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 3) & 0b111) << 10) | (((i >> 6) & 0b11) << 5)
}

#[inline(always)]
fn lwsp_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 5) & 0b1) << 12) | (((i >> 2) & 0b111) << 4) | (((i >> 6) & 0b11) << 2)
}

#[inline(always)]
fn ldsp_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 5) & 0b1) << 12) | (((i >> 3) & 0b11) << 5) | (((i >> 6) & 0b111) << 2)
}

#[inline(always)]
fn swsp_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 2) & 0b1111) << 9) | (((i >> 6) & 0b11) << 7)
}

#[inline(always)]
fn sdsp_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 3) & 0b111) << 10) | (((i >> 6) & 0b111) << 7)
}

#[inline(always)]
fn beqz_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 8) & 0b1) << 12)
        | (((i >> 3) & 0b11) << 10)
        | (((i >> 6) & 0b11) << 5)
        | (((i >> 1) & 0b11) << 3)
        | (((i >> 5) & 0b1) << 2)
}

#[inline(always)]
fn jal_immediate(immediate: i32) -> u16 {
    let i = immediate as u32 as u16;
    (((i >> 11) & 0b1) << 12)
        | (((i >> 4) & 0b1) << 11)
        | (((i >> 8) & 0b11) << 9)
        | (((i >> 10) & 0b1) << 8)
        | (((i >> 6) & 0b1) << 7)
        | (((i >> 7) & 0b1) << 6)
        | (((i >> 1) & 0b111) << 3)
        | (((i >> 5) & 0b1) << 2)
}

#[inline(always)]
fn lui_immediate(immediate: i32) -> u16 {
    let i = immediate as u32;
    let v = (((i >> 17) & 0b1) << 12) | (((i >> 12) & 0b1_1111) << 2);
    v as u16
}

#[inline(always)]
fn compact_register(reg: usize, shift: usize) -> u16 {
    ((reg as u16 - 8) & 0b111) << shift
}

#[inline(always)]
fn full_register(reg: usize, shift: usize) -> u16 {
    ((reg as u16) & 0b1_1111) << shift
}

#[inline(always)]
fn is_compact_register(reg: usize) -> bool {
    (8..=15).contains(&reg)
}

#[inline(always)]
pub fn funct4(funct: u16) -> u16 {
    (funct & 0b111) << 13
}

#[inline(always)]
pub fn op(op: u16) -> u16 {
    op & 0b11
}
