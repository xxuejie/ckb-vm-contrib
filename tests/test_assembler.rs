mod utils;

use ckb_vm::{decoder::build_decoder, instructions::Rtype, machine::VERSION1, ISA_IMC};
use ckb_vm_contrib::assembler::assemble;
use proptest::prelude::*;
use utils::{assert_same_inst, rtype_op, VecMemory};

proptest! {
    #[test]
    fn assemble_rtype_instruction(op in rtype_op(), rd in 0usize..32usize, rs1 in 0usize..32usize, rs2 in 0usize..32usize) {
        let i = Rtype::new(op, rd, rs1, rs2);
        let result = assemble::<u64>(&[i.into()]);
        assert!(result.is_ok(), "Assemble error: {:?}", result.unwrap_err());
        let assemble_result = result.unwrap();
        let mut mem = VecMemory::<u64>::new(assemble_result.clone());

        let mut decoder = build_decoder::<u64>(ISA_IMC, VERSION1);
        let decode_result = decoder.decode(&mut mem, 0);
        assert!(decode_result.is_ok(), "Decoder error: {:?}", decode_result.unwrap_err());
        let i2 = decode_result.unwrap();

        assert_same_inst(&i.0, &i2);
    }
}
