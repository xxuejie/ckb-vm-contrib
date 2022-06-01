mod utils;

use ckb_vm::instructions::{Rtype, Stype};
use ckb_vm_contrib::assembler::parse;
use proptest::prelude::*;
use utils::{assert_same_tagged, rtype_op, stype_branch_op, stype_store_op};

proptest! {
    #[test]
    fn parse_rtype_instruction(op in rtype_op(), rd in 0usize..32usize, rs1 in 0usize..32usize, rs2 in 0usize..32usize) {
        let i = Rtype::new(op, rd, rs1, rs2);
        let text = format!("{}", i);

        let parse_result = parse(&text);
        assert!(parse_result.is_ok(), "Parser error: {:?}", parse_result.unwrap_err());
        let insts = parse_result.unwrap();
        assert_eq!(1, insts.len());

        assert_same_tagged(&i.into(), &insts[0]);
    }

    #[test]
    fn parse_stype_branch_instruction(op in stype_branch_op(), rs1 in 0usize..32usize, rs2 in 0usize..32usize, imm in -2048i32..2048i32) {
        let i = Stype::new_s(op, imm << 1, rs1, rs2);
        let text = format!("{}", i);

        let parse_result = parse(&text);
        assert!(parse_result.is_ok(), "Parser error: {:?}", parse_result.unwrap_err());
        let insts = parse_result.unwrap();
        assert_eq!(1, insts.len());

        assert_same_tagged(&i.into(), &insts[0]);
    }

    #[test]
    fn parse_stype_store_instruction(op in stype_store_op(), rs1 in 0usize..32usize, rs2 in 0usize..32usize, imm in -2048i32..2048i32) {
        let i = Stype::new_s(op, imm, rs1, rs2);
        let text = format!("{}", i);

        let parse_result = parse(&text);
        assert!(parse_result.is_ok(), "Parser error: {:?}", parse_result.unwrap_err());
        let insts = parse_result.unwrap();
        assert_eq!(1, insts.len());

        assert_same_tagged(&i.into(), &insts[0]);
    }
}
