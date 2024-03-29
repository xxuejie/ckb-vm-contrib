mod utils;

use ckb_vm::{
    ckb_vm_definitions::instructions as opcodes,
    instructions::{tagged::TaggedInstruction, Itype, Rtype, Stype, Utype},
};
use ckb_vm_contrib::{
    assembler::{assemble, parse},
    printer::InstructionPrinter,
};
use proptest::prelude::*;
use utils::*;

#[test]
fn test_auipc_shifts() {
    let text = "auipc   a0,0x7ffff";

    let insts = parse::<u64>(text).expect("parsing");
    let binary = assemble::<u64>(&insts).expect("assemble");

    assert_eq!(binary, [0x17, 0xf5, 0xff, 0x7f]);
}

#[test]
fn test_lui_shifts() {
    let text = "lui   gp,0x7ffff";

    let insts = parse::<u64>(text).expect("parsing");
    let binary = assemble::<u64>(&insts).expect("assemble");

    assert_eq!(binary, [0xb7, 0xf1, 0xff, 0x7f]);
}

#[test]
fn test_beqz_parsing() {
    let text = "beqz s6,2168";

    let insts = parse::<u64>(text).expect("parsing");
    let binary = assemble::<u64>(&insts).expect("assemble");

    assert_eq!(binary, [0xe3, 0x0c, 0x0b, 0x06]);
}

fn t<T: Into<TaggedInstruction>>(i: T) {
    let i: TaggedInstruction = i.into();
    let text = format!("{}", InstructionPrinter::new(i.clone()));

    let parse_result = parse::<u64>(&text);
    assert!(
        parse_result.is_ok(),
        "Parser error: {:?}",
        parse_result.unwrap_err()
    );
    let insts = parse_result.unwrap();
    assert_eq!(1, insts.len());

    assert_same_tagged(&i, &insts[0]);
}

proptest! {
    #[test]
    fn parse_rtype_instruction(
        op in rtype_op(),
        rd in 0usize..32usize,
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize)
    {
        t(Rtype::new(op, rd, rs1, rs2));
    }

    #[test]
    fn parse_stype_branch_instruction(
        op in stype_branch_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Stype::new_s(op, imm << 1, rs1, rs2));
    }

    #[test]
    fn parse_stype_store_instruction(
        op in stype_store_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Stype::new_s(op, imm, rs1, rs2));
    }

    #[test]
    fn parse_itype_normal_instruction(
        op in itype_normal_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Itype::new_s(op, rs1, rs2, imm));
    }

    #[test]
    fn parse_itype_shift_instruction(
        op in itype_shift_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..64u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }

    #[test]
    fn parse_itype_shiftw_instruction(
        op in itype_shiftw_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..32u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }

    #[test]
    fn parse_utype_instruction(
        op in prop::sample::select(vec![opcodes::OP_LUI, opcodes::OP_AUIPC]),
        rd in 0usize..32usize,
        uimm in 0u32..1048576u32)
    {
        t(Utype::new(op, rd, uimm << 12));
    }


    #[test]
    fn parse_jal_instruction(
        rd in 0usize..32usize,
        imm in -524288i32..524288i32)
    {
        t(Utype::new_s(opcodes::OP_JAL, rd, imm << 1));
    }

    #[test]
    fn parse_system_instruction(
        op in prop::sample::select(vec![opcodes::OP_ECALL, opcodes::OP_EBREAK]),
    ) {
        t(Rtype::new(op, 0, 0, 0));
    }
}
