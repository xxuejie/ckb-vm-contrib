mod utils;

use ckb_vm::instructions::Rtype;
use ckb_vm_contrib::assembler::parse;
use ckb_vm_definitions::instructions as opcodes;
use proptest::prelude::*;
use utils::assert_same_tagged;

proptest! {
    #[test]
    fn parse_rtype_instruction(rd in 0usize..32usize, rs1 in 0usize..32usize, rs2 in 0usize..32usize) {
        let i = Rtype::new(opcodes::OP_ADD, rd, rs1, rs2);
        let text = format!("{}", i);

        let parse_result = parse(&text);
        assert!(parse_result.is_ok(), "Parser error: {:?}", parse_result.unwrap_err());
        let insts = parse_result.unwrap();
        assert_eq!(1, insts.len());

        assert_same_tagged(&i.into(), &insts[0]);
    }
}
