use ckb_vm::Register;

#[inline(always)]
pub fn opcode(opcode: u32) -> u32 {
    opcode & 0b111_1111
}

#[inline(always)]
pub fn funct3(funct: u32) -> u32 {
    (funct & 0b111) << 12
}

#[inline(always)]
pub fn funct7(funct: u32) -> u32 {
    (funct & 0b111_1111) << 25
}

#[inline(always)]
pub fn shamt<R: Register>(immediate: i32) -> u32 {
    ((immediate as u32) & R::SHIFT_MASK as u32) << 20
}

#[inline(always)]
pub fn rd(reg: usize) -> u32 {
    ((reg as u32) & 0b1_1111) << 7
}

#[inline(always)]
pub fn rs1(reg: usize) -> u32 {
    ((reg as u32) & 0b1_1111) << 15
}

#[inline(always)]
pub fn rs2(reg: usize) -> u32 {
    ((reg as u32) & 0b1_1111) << 20
}

#[inline(always)]
pub fn itype_immediate(immediate: i32) -> u32 {
    ((immediate as u32) & 0xFFF) << 20
}

#[inline(always)]
pub fn jtype_immediate(immediate: i32) -> u32 {
    let i = immediate as u32;
    let v = (((i >> 20) & 0b1) << 19)
        | (((i >> 1) & 0b11_1111_1111) << 9)
        | (((i >> 11) & 0b1) << 8)
        | (i >> 12) & 0b1111_1111;
    v << 12
}

#[inline(always)]
pub fn utype_immediate(immediate: i32) -> u32 {
    (immediate as u32) & 0xFFFFF000
}

#[inline(always)]
pub fn stype_immediate(immediate: i32) -> u32 {
    let i = immediate as u32;
    (((i >> 5) & 0b111_1111) << 25) | ((i & 0b1_1111) << 6)
}

#[inline(always)]
pub fn btype_immediate(immediate: i32) -> u32 {
    let i = immediate as u32;
    (((i >> 12) & 0b1) << 31)
        | (((i >> 5) & 0b11_1111) << 25)
        | (((i >> 1) & 0b1111) << 8)
        | (((i >> 11) & 0b1) << 7)
}
