#![allow(dead_code)]

use ckb_vm::{
    instructions::{tagged::TaggedInstruction, Instruction, InstructionOpcode},
    Bytes, Error, Memory, Register, RISCV_PAGESIZE,
};
use ckb_vm_definitions::instructions as opcodes;
use proptest::prelude::*;
use std::marker::PhantomData;

pub fn rtype_op() -> impl Strategy<Value = InstructionOpcode> {
    prop::sample::select(vec![opcodes::OP_ADD, opcodes::OP_AND])
}

pub fn assert_same_tagged(i: &TaggedInstruction, i2: &TaggedInstruction) {
    match (i, i2) {
        (TaggedInstruction::Rtype(i), TaggedInstruction::Rtype(i2)) => {
            assert_same_inst(&i.0, &i2.0)
        }
        (TaggedInstruction::Itype(i), TaggedInstruction::Itype(i2)) => {
            assert_same_inst(&i.0, &i2.0)
        }
        (TaggedInstruction::Stype(i), TaggedInstruction::Stype(i2)) => {
            assert_same_inst(&i.0, &i2.0)
        }
        (TaggedInstruction::Utype(i), TaggedInstruction::Utype(i2)) => {
            assert_same_inst(&i.0, &i2.0)
        }
        _ => panic!("Unmatched instructions: \"{}\" <> \"{}\"", i, i2),
    }
}

pub fn assert_same_inst(i: &Instruction, i2: &Instruction) {
    // Due to ckb-vm's design, internal decoder would also generate a special
    // +flag+ for faster execution. When comparing 2 instructions, we need to
    // strip this flag out. See the following URL for more details:
    //
    // https://github.com/nervosnetwork/ckb-vm/blob/3d50ec6e419c66311708b093b991d94bfac987a5/definitions/src/instructions.rs#L1-L20
    assert_eq!(
        i & 0xFFFFFFFF00FFFFFF,
        i2 & 0xFFFFFFFF00FFFFFF,
        "Instructions do not match: original {:016x} <> decoded {:016x}",
        i,
        i2,
    );
}

pub struct VecMemory<R> {
    data: Vec<u8>,
    _inner: PhantomData<R>,
}

impl<R> VecMemory<R> {
    pub fn new(mut data: Vec<u8>) -> Self {
        // Fill in a page at least for enough decoder padding
        data.resize(RISCV_PAGESIZE, 0);
        Self {
            data,
            _inner: PhantomData,
        }
    }
}

impl<R: Register> Memory for VecMemory<R> {
    type REG = R;

    fn init_pages(
        &mut self,
        _addr: u64,
        _size: u64,
        _flags: u8,
        _source: Option<Bytes>,
        _offset_from_addr: u64,
    ) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn fetch_flag(&mut self, _page: u64) -> Result<u8, Error> {
        Err(Error::Unimplemented)
    }

    fn set_flag(&mut self, _page: u64, _flag: u8) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn clear_flag(&mut self, _page: u64, _flag: u8) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn execute_load16(&mut self, addr: u64) -> Result<u16, Error> {
        let addr = addr as usize;
        if addr + 2 > self.data.len() {
            return Err(Error::MemOutOfBound);
        }
        let mut data = [0u8; 2];
        data.copy_from_slice(&self.data[addr..addr + 2]);
        Ok(u16::from_le_bytes(data))
    }

    fn execute_load32(&mut self, addr: u64) -> Result<u32, Error> {
        let addr = addr as usize;
        if addr + 4 > self.data.len() {
            return Err(Error::MemOutOfBound);
        }
        let mut data = [0u8; 4];
        data.copy_from_slice(&self.data[addr..addr + 4]);
        Ok(u32::from_le_bytes(data))
    }

    fn load8(&mut self, _addr: &Self::REG) -> Result<Self::REG, Error> {
        Err(Error::Unimplemented)
    }

    fn load16(&mut self, _addr: &Self::REG) -> Result<Self::REG, Error> {
        Err(Error::Unimplemented)
    }

    fn load32(&mut self, _addr: &Self::REG) -> Result<Self::REG, Error> {
        Err(Error::Unimplemented)
    }

    fn load64(&mut self, _addr: &Self::REG) -> Result<Self::REG, Error> {
        Err(Error::Unimplemented)
    }

    fn store8(&mut self, _addr: &Self::REG, _value: &Self::REG) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn store16(&mut self, _addr: &Self::REG, _value: &Self::REG) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn store32(&mut self, _addr: &Self::REG, _value: &Self::REG) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn store64(&mut self, _addr: &Self::REG, _value: &Self::REG) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn store_bytes(&mut self, _addr: u64, _value: &[u8]) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }

    fn store_byte(&mut self, _addr: u64, _size: u64, _value: u8) -> Result<(), Error> {
        Err(Error::Unimplemented)
    }
}
