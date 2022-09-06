use super::{
    symbols::{AotSymbols, EntryFunctionType},
    AOT_ISA, AOT_VERSION,
};
use ckb_vm::{
    decoder::{build_decoder, Decoder},
    instructions::execute,
    machine::{DefaultMachine, DefaultMachineBuilder},
    memory::{fill_page_data, memset, round_page_down, round_page_up, FLAG_EXECUTABLE},
    Bytes, CoreMachine, Error, Machine, Memory, SupportMachine,
};
use ckb_vm_definitions::{RISCV_GENERAL_REGISTER_NUMBER, RISCV_PAGESIZE};
use region::{self, alloc, query_range, Allocation, Protection, Region};
use std::collections::HashMap;
use std::ptr;
use std::slice::from_raw_parts_mut;

const RUNTIME_FLAG_RUNNING: u8 = 1;

pub const EXIT_REASON_UNKNOWN_ADDRESS: u8 = 1;
pub const EXIT_REASON_ECALL: u8 = 2;
pub const EXIT_REASON_EBREAK: u8 = 3;

pub struct LlvmAotMachine {
    pub machine: DefaultMachine<LlvmAotCoreMachine>,

    pub code_hash: [u8; 32],
    // Mapping from RISC-V function entry to host function entry
    pub function_mapping: HashMap<u64, u64>,
    pub entry_function: EntryFunctionType,
}

impl LlvmAotMachine {
    pub fn new(memory_size: usize, aot_symbols: &AotSymbols) -> Result<Self, Error> {
        let core_machine = LlvmAotCoreMachine::new(memory_size)?;
        let machine = DefaultMachineBuilder::new(core_machine).build();
        Self::new_with_machine(machine, aot_symbols)
    }

    pub fn new_with_machine(
        machine: DefaultMachine<LlvmAotCoreMachine>,
        aot_symbols: &AotSymbols,
    ) -> Result<Self, Error> {
        if aot_symbols.code_hash.len() != 32 {
            return Err(Error::External(
                "code hash must be a slice of 32 bytes!".to_string(),
            ));
        }
        let mut code_hash = [0u8; 32];
        code_hash.copy_from_slice(aot_symbols.code_hash);

        let function_mapping = aot_symbols
            .address_table
            .iter()
            .map(|table| (table.riscv_addr, table.host_addr))
            .collect();

        Ok(Self {
            machine,
            code_hash,
            function_mapping,
            entry_function: aot_symbols.entry_function,
        })
    }

    pub fn load_program(&mut self, program: &Bytes, args: &[Bytes]) -> Result<u64, Error> {
        let program_hash: [u8; 32] = blake3::hash(program).into();
        if program_hash != self.code_hash {
            return Err(Error::External(format!(
                "Loaded program hash: {:x} does not match expected AOT code hash: {:x}",
                Bytes::from(program_hash.to_vec()),
                Bytes::from(self.code_hash.to_vec())
            )));
        }
        self.machine.load_program(program, args)
    }

    pub fn run(&mut self) -> Result<i8, Error> {
        let mut decoder = build_decoder::<u64>(self.machine.isa(), self.machine.version());
        self.machine.set_running(true);
        while self.machine.running() {
            let pc = *self.machine.pc();
            if let Some(host_function) = self.function_mapping.get(&pc) {
                // Clear previous last_ra value, which should now be invalid.
                self.machine.inner_mut().data.last_ra = 0;
                let result = unsafe {
                    (self.entry_function)(&mut self.machine.inner_mut().data, *host_function)
                };
                match result {
                    EXIT_REASON_UNKNOWN_ADDRESS => (),
                    EXIT_REASON_ECALL => self.machine.ecall()?,
                    EXIT_REASON_EBREAK => self.machine.ebreak()?,
                    _ => return Err(Error::Asm(result)),
                }
            } else {
                self.step(&mut decoder)?;
            }
        }
        Ok(self.machine.exit_code())
    }

    pub fn step(&mut self, decoder: &mut Decoder) -> Result<(), Error> {
        let pc = *self.machine.pc();
        let instruction = decoder.decode(self.machine.memory_mut(), pc)?;
        execute(instruction, &mut self.machine)
    }
}

#[repr(C)]
pub struct LlvmAotCoreMachineData {
    pub pc: u64,
    pub memory: *mut u8,
    pub registers: [u64; RISCV_GENERAL_REGISTER_NUMBER],
    pub last_ra: u64,
    pub next_pc: u64,
    pub runtime_flags: u8,
    pub exit_aot_reason: u8,
    pub jmpbuf: [u64; 5],
}

pub struct LlvmAotCoreMachine {
    pub data: LlvmAotCoreMachineData,

    pub allocation: Allocation,
    pub cached_region: Option<Region>,
}

impl LlvmAotCoreMachine {
    pub fn new(memory_size: usize) -> Result<Self, Error> {
        if memory_size % RISCV_PAGESIZE != 0 {
            return Err(Error::External(
                "Memory size must be a multiple of 4KB!".to_string(),
            ));
        }

        let mut memory = alloc(memory_size, Protection::READ_WRITE)
            .map_err(|e| Error::External(format!("region alloc error: {}", e)))?;

        let data = LlvmAotCoreMachineData {
            pc: 0,
            memory: memory.as_mut_ptr(),
            registers: [0u64; RISCV_GENERAL_REGISTER_NUMBER],
            last_ra: 0,
            next_pc: 0,
            runtime_flags: 0,
            exit_aot_reason: 0,
            jmpbuf: [0u64; 5],
        };

        Ok(Self {
            data,
            allocation: memory,
            cached_region: None,
        })
    }

    fn execute_check(&mut self, addr: u64, len: usize) -> Result<*const u8, Error> {
        let host_addr = self.data.memory.wrapping_offset(addr as isize);
        if let Some(region) = self.cached_region {
            let r = region.as_range();
            if host_addr as usize >= r.start && host_addr as usize + len <= r.end {
                return Ok(host_addr);
            }
        }
        let ranges: Vec<Region> = query_range(host_addr, len)
            .map_err(e)?
            .try_fold(vec![], |mut acc, r| match r {
                Ok(r) => {
                    acc.push(r);
                    Ok(acc)
                }
                Err(e) => Err(e),
            })
            .map_err(e)?;
        if ranges.iter().any(|r| !r.is_executable()) {
            return Err(Error::External(format!(
                "addr: {:x} executing on non executable pages!",
                addr
            )));
        }
        if ranges.len() == 1 {
            self.cached_region = Some(ranges[0]);
        }
        Ok(host_addr)
    }
}

impl CoreMachine for LlvmAotCoreMachine {
    type REG = u64;
    type MEM = Self;

    fn pc(&self) -> &Self::REG {
        &self.data.pc
    }

    fn update_pc(&mut self, pc: Self::REG) {
        self.data.next_pc = pc;
    }

    fn commit_pc(&mut self) {
        self.data.pc = self.data.next_pc;
    }

    fn memory(&self) -> &Self::MEM {
        self
    }

    fn memory_mut(&mut self) -> &mut Self::MEM {
        self
    }

    fn registers(&self) -> &[Self::REG] {
        &self.data.registers
    }

    fn set_register(&mut self, idx: usize, value: Self::REG) {
        self.data.registers[idx] = value;
    }

    fn version(&self) -> u32 {
        AOT_VERSION
    }

    fn isa(&self) -> u8 {
        AOT_ISA
    }
}

impl SupportMachine for LlvmAotCoreMachine {
    fn cycles(&self) -> u64 {
        0
    }

    fn set_cycles(&mut self, _cycles: u64) {}

    fn max_cycles(&self) -> u64 {
        0
    }

    fn running(&self) -> bool {
        self.data.runtime_flags & RUNTIME_FLAG_RUNNING != 0
    }

    fn set_running(&mut self, running: bool) {
        self.data.runtime_flags &= !RUNTIME_FLAG_RUNNING;
        self.data.runtime_flags |= if running { RUNTIME_FLAG_RUNNING } else { 0 };
    }

    // Erase all the states of the virtual machine.
    fn reset(&mut self, _max_cycles: u64) {
        unimplemented!()
    }

    fn reset_signal(&mut self) -> bool {
        false
    }
}

impl Memory for LlvmAotCoreMachine {
    type REG = u64;

    fn init_pages(
        &mut self,
        addr: u64,
        size: u64,
        flags: u8,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error> {
        if round_page_down(addr) != addr || round_page_up(size) != size {
            return Err(Error::MemPageUnalignedAccess);
        }
        let memory_size = self.allocation.len();
        if addr > memory_size as u64
            || size > memory_size as u64
            || addr + size > memory_size as u64
            || offset_from_addr > size
        {
            return Err(Error::MemOutOfBound);
        }
        let host_addr = self.data.memory.wrapping_offset(addr as isize);
        let mut ranges = query_range(host_addr, size as usize).map_err(e)?;
        if ranges.any(|r| r.is_ok() && r.unwrap().is_executable()) {
            return Err(Error::MemWriteOnFreezedPage);
        }
        fill_page_data(self, addr, size, source, offset_from_addr)?;
        let protection = if flags & FLAG_EXECUTABLE != 0 {
            Protection::READ_EXECUTE
        } else {
            Protection::READ_WRITE
        };
        unsafe { region::protect(host_addr, size as usize, protection) }.map_err(e)?;
        self.cached_region = None;
        Ok(())
    }

    // TODO: right now we leverage host OS to mark the pages as executable,
    // is it secure to mark RISC-V code executable on a x86/arm CPU?
    fn execute_load16(&mut self, addr: u64) -> Result<u16, Error> {
        let host_addr = self.execute_check(addr, 2)?;
        Ok(unsafe { ptr::read(host_addr as *const u16) })
    }

    fn execute_load32(&mut self, addr: u64) -> Result<u32, Error> {
        let host_addr = self.execute_check(addr, 4)?;
        Ok(unsafe { ptr::read(host_addr as *const u32) })
    }

    fn load8(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        let value: u8 = unsafe { ptr::read(host_addr) };
        Ok(u64::from(value))
    }

    fn load16(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        let value: u16 = unsafe { ptr::read(host_addr as *const u16) };
        Ok(u64::from(value))
    }

    fn load32(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        let value: u32 = unsafe { ptr::read(host_addr as *const u32) };
        Ok(u64::from(value))
    }

    fn load64(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        let value: u64 = unsafe { ptr::read(host_addr as *const u64) };
        Ok(value)
    }

    // TODO: dirty tracking, or maybe we should rethink on how to implement
    // suspend/resume in such a design?
    fn store_byte(&mut self, addr: u64, size: u64, value: u8) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(addr as isize);
        let mut dst = unsafe { from_raw_parts_mut(host_addr, size as usize) };
        memset(&mut dst, value);
        Ok(())
    }

    fn store_bytes(&mut self, addr: u64, value: &[u8]) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(addr as isize);
        let dst = unsafe { from_raw_parts_mut(host_addr, value.len()) };
        dst.copy_from_slice(value);
        Ok(())
    }

    fn store8(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        unsafe {
            ptr::write(host_addr, *value as u8);
        }
        Ok(())
    }

    fn store16(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        unsafe {
            ptr::write(host_addr as *mut u16, *value as u16);
        }
        Ok(())
    }

    fn store32(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        unsafe {
            ptr::write(host_addr as *mut u32, *value as u32);
        }
        Ok(())
    }

    fn store64(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        let host_addr = self.data.memory.wrapping_offset(*addr as isize);
        unsafe {
            ptr::write(host_addr as *mut u64, *value as u64);
        }
        Ok(())
    }

    fn fetch_flag(&mut self, _page: u64) -> Result<u8, Error> {
        unreachable!()
    }

    fn set_flag(&mut self, _page: u64, _flag: u8) -> Result<(), Error> {
        unreachable!()
    }

    fn clear_flag(&mut self, _page: u64, _flag: u8) -> Result<(), Error> {
        unreachable!()
    }
}

fn e(error: region::Error) -> Error {
    Error::External(format!("region error: {}", error))
}
