mod utils;

use ckb_vm::{
    ckb_vm_definitions::instructions as opcodes,
    decoder::build_decoder,
    instructions::{tagged::TaggedInstruction, Itype, Rtype, Stype, Utype},
    machine::VERSION1,
    ISA_IMC,
};
use ckb_vm_contrib::assembler::{assemble, i, m, Assembler};
use proptest::prelude::*;
use utils::*;

fn t<T: Into<TaggedInstruction>>(i: T) {
    let i: TaggedInstruction = i.into();

    t_normal_assemble(&i);
    t_norvc_assemble(&i);
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
    fn assemble_itype_shiftw_instruction(
        op in itype_shiftw_op(),
        rs1 in 0usize..32usize,
        rs2 in 0usize..32usize,
        uimm in 0u32..32u32)
    {
        t(Itype::new_u(op, rs1, rs2, uimm));
    }

    #[test]
    fn assemble_utype_instruction(
        op in prop::sample::select(vec![opcodes::OP_LUI, opcodes::OP_AUIPC]),
        rd in 0usize..32usize,
        uimm in 0u32..1048576u32)
    {
        t(Utype::new(op, rd, uimm << 12));
    }


    #[test]
    fn assemble_jal_instruction(
        rd in 0usize..32usize,
        imm in -524288i32..524288i32)
    {
        t(Utype::new_s(opcodes::OP_JAL, rd, imm << 1));
    }

    #[test]
    fn assemble_system_instruction(
        op in prop::sample::select(vec![opcodes::OP_ECALL, opcodes::OP_EBREAK]),
    ) {
        t(Rtype::new(op, 0, 0, 0));
    }
}

fn t_normal_assemble(i: &TaggedInstruction) {
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

fn t_norvc_assemble(i: &TaggedInstruction) {
    let mut assembler = Assembler::new();
    assembler.add_assembler_factory(i::assembler::<u64>);
    assembler.add_assembler_factory(m::assembler::<u64>);

    let result = assembler.assemble(i);
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
