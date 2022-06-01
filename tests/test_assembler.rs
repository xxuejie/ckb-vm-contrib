mod utils;

use ckb_vm::{
    decoder::build_decoder,
    instructions::{tagged::TaggedInstruction, Itype, Rtype, Stype},
    machine::VERSION1,
    ISA_IMC,
};
use ckb_vm_contrib::assembler::assemble;
use proptest::prelude::*;
use utils::*;

fn t<T: Into<TaggedInstruction>>(i: T) {
    let i: TaggedInstruction = i.into();
    let result = assemble::<u64>(&[i.clone()]);
    assert!(result.is_ok(), "Assemble error: {:?}", result.unwrap_err());
    let assemble_result = result.unwrap();
    let mut mem = VecMemory::<u64>::new(assemble_result.clone());

    let mut decoder = build_decoder::<u64>(ISA_IMC, VERSION1);
    let decode_result = decoder.decode(&mut mem, 0);
    assert!(
        decode_result.is_ok(),
        "Decoder error: {:?}",
        decode_result.unwrap_err()
    );
    let i2 = decode_result.unwrap();
    let i2: TaggedInstruction = i2.try_into().unwrap();

    assert_same_tagged(&i, &i2);
}

proptest! {
    #[test]
    fn assemble_rtype_instruction(
        op in rtype_op(),
        rd in 0usize..32usize,
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize)
    {
        t(Rtype::new(op, rd, rs1, rs2));
    }

    #[test]
    fn assemble_stype_branch_instruction(
        op in stype_branch_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Stype::new_s(op, imm << 1, rs1, rs2));
    }

    #[test]
    fn assemble_stype_store_instruction(
        op in stype_store_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Stype::new_s(op, imm, rs1, rs2));
    }

    #[test]
    fn assemble_itype_normal_instruction(
        op in itype_normal_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        imm in -2048i32..2048i32)
    {
        t(Itype::new_s(op, rs1, rs2, imm));
    }

    #[test]
    fn assemble_itype_shift_instruction(
        op in itype_shift_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..64u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }

    #[test]
    fn assemble_itype_shift_instruction_u32(
        op in itype_shift_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..32u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }

    #[test]
    fn assemble_itype_shiftw_instruction(
        op in itype_shiftw_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..32u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }
}
