use ckb_vm::{
    decoder::Decoder,
    instructions::{
        extract_opcode, instruction_length, set_instruction_length_n, Instruction, Itype, Utype,
    },
    Error, Memory,
};
use ckb_vm_definitions::instructions as insts;

/// This wraps on top of ckb-vm's decoder, providing more mops & opcode
/// rewriting that can further optimize the VM.
pub struct AuxDecoder {
    inner: Decoder,
}

impl AuxDecoder {
    pub fn new(inner: Decoder) -> Self {
        Self { inner }
    }

    pub fn decode<M: Memory>(&mut self, memory: &mut M, pc: u64) -> Result<Instruction, Error> {
        let head_inst = self.inner.decode(memory, pc)?;
        match extract_opcode(head_inst) {
            insts::OP_AUIPC => {
                let i = Utype(head_inst);
                let head_len = instruction_length(head_inst);
                let mut rule_auipc = || -> Result<Option<Instruction>, Error> {
                    let next_instruction = self.inner.decode(memory, pc + head_len as u64)?;
                    let next_opcode = extract_opcode(next_instruction);
                    match next_opcode {
                        insts::OP_ADDI => {
                            let ni = Itype(next_instruction);
                            if i.rd() == ni.rd() && ni.rd() == ni.rs1() {
                                let value = pc
                                    .wrapping_add(i64::from(i.immediate_s()) as u64)
                                    .wrapping_add(i64::from(ni.immediate_s()) as u64);
                                if let Ok(value) = value.try_into() {
                                    return Ok(Some(set_instruction_length_n(
                                        Utype::new(insts::OP_CUSTOM_LOAD_IMM, i.rd(), value).0,
                                        head_len + instruction_length(next_instruction),
                                    )));
                                }
                            }
                        }
                        _ => (),
                    };
                    Ok(None)
                };
                if let Ok(Some(i)) = rule_auipc() {
                    return Ok(i);
                } else {
                    let value = pc.wrapping_add(i64::from(i.immediate_s()) as u64);
                    if let Ok(value) = value.try_into() {
                        return Ok(set_instruction_length_n(
                            Utype::new(insts::OP_CUSTOM_LOAD_UIMM, i.rd(), value).0,
                            head_len,
                        ));
                    }
                }
            }
            _ => (),
        };

        Ok(head_inst)
    }
}
