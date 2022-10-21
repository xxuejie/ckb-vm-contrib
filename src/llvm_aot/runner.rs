use super::{
    symbols::{AotSymbols, EntryFunctionType},
    AOT_ISA, AOT_VERSION,
};
use ckb_vm::{
    decoder::{build_decoder, Decoder},
    instructions::{execute, is_basic_block_end_instruction},
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

pub const EXIT_REASON_ECALL_UNREACHABLE: u8 = 1;
pub const EXIT_REASON_EBREAK_UNREACHABLE: u8 = 2;
pub const EXIT_REASON_UNKNOWN_BLOCK: u8 = 3;
pub const EXIT_REASON_UNKNOWN_PC_VALUE: u8 = 4;
pub const EXIT_REASON_UNKNOWN_RESUME_ADDRESS: u8 = 5;
pub const EXIT_REASON_MALFORMED_RETURN: u8 = 6;
pub const EXIT_REASON_MALFORMED_INDIRECT_CALL: u8 = 7;
pub const EXIT_REASON_BARE_CALL_EXIT: u8 = 101;
pub const EXIT_REASON_MAX_CYCLES_EXCEEDED: u8 = 102;
pub const EXIT_REASON_CYCLES_OVERFLOW: u8 = 103;

pub struct LlvmAotMachine<'a> {
    pub machine: DefaultMachine<'a, LlvmAotCoreMachine<'a>>,

    pub code_hash: [u8; 32],
    // Mapping from RISC-V function entry to host function entry
    pub function_mapping: HashMap<u64, u64>,
    pub entry_function: EntryFunctionType,

    pub decoder: Decoder,
    pub bare_call_error: Option<Error>,
}

#[repr(C)]
pub struct LlvmAotMachineEnv<'a> {
    pub data: *mut LlvmAotMachine<'a>,
    pub ecall: unsafe extern "C" fn(m: *mut LlvmAotMachine) -> u64,
    pub ebreak: unsafe extern "C" fn(m: *mut LlvmAotMachine) -> u64,
    pub query_function: unsafe extern "C" fn(m: *mut LlvmAotMachine, riscv_addr: u64) -> u64,
    pub interpret: unsafe extern "C" fn(m: *mut LlvmAotMachine) -> u64,
}

unsafe extern "C" fn bare_ecall(m: *mut LlvmAotMachine) -> u64 {
    let m = &mut *m;
    match m.machine.ecall() {
        Ok(()) => {
            if m.machine.running() {
                1
            } else {
                0
            }
        }
        Err(e) => {
            m.bare_call_error = Some(e);
            0
        }
    }
}

unsafe extern "C" fn bare_ebreak(m: *mut LlvmAotMachine) -> u64 {
    let m = &mut *m;
    match m.machine.ebreak() {
        Ok(()) => {
            if m.machine.running() {
                1
            } else {
                0
            }
        }
        Err(e) => {
            m.bare_call_error = Some(e);
            0
        }
    }
}

unsafe extern "C" fn bare_query_function(m: *mut LlvmAotMachine, riscv_addr: u64) -> u64 {
    let m = &*m;
    *m.function_mapping
        .get(&riscv_addr)
        .unwrap_or(&u64::max_value())
}

unsafe extern "C" fn bare_interpret(m: *mut LlvmAotMachine) -> u64 {
    let mut m = &mut *m;
    match inner_interpret(&mut m) {
        Ok(block_end) => block_end,
        Err(e) => {
            m.bare_call_error = Some(e);
            0
        }
    }
}

fn inner_interpret(m: &mut LlvmAotMachine) -> Result<u64, Error> {
    while m.machine.running() {
        let pc = *m.machine.pc();
        let instruction = m.decoder.decode(m.machine.memory_mut(), pc)?;
        execute(instruction, &mut m.machine)?;

        if is_basic_block_end_instruction(instruction) {
            return Ok(*m.machine.pc());
        }
    }
    // Terminating
    Ok(0)
}

impl<'a> LlvmAotMachine<'a> {
    pub fn new(memory_size: usize, aot_symbols: &AotSymbols) -> Result<Self, Error> {
        let core_machine = LlvmAotCoreMachine::new(memory_size)?;
        let machine = DefaultMachineBuilder::new(core_machine).build();
        Self::new_with_machine(machine, aot_symbols)
    }

    pub fn new_with_machine(
        machine: DefaultMachine<'a, LlvmAotCoreMachine<'a>>,
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

        let decoder = build_decoder::<u64>(machine.isa(), machine.version());

        Ok(Self {
            machine,
            code_hash,
            function_mapping,
            entry_function: aot_symbols.entry_function,
            bare_call_error: None,
            decoder,
        })
    }

    pub fn set_max_cycles(&mut self, cycles: u64) {
        self.machine.inner_mut().data.max_cycles = cycles;
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

    pub fn run(&mut self, fast_mode: bool) -> Result<i8, Error> {
        self.machine.set_running(true);
        while self.machine.running() {
            let pc = *self.machine.pc();
            if let Some(host_function) = self.function_mapping.get(&pc) {
                let result = {
                    // Clear previous last_ra value, which should now be invalid.
                    self.machine.inner_mut().data.last_ra = 0;
                    let env = self.env();
                    self.machine.inner_mut().data.env = &env;
                    unsafe {
                        (self.entry_function)(&mut self.machine.inner_mut().data, *host_function)
                    }
                };
                if let Some(e) = &self.bare_call_error {
                    return Err(e.clone());
                }
                match result {
                    EXIT_REASON_ECALL_UNREACHABLE
                    | EXIT_REASON_EBREAK_UNREACHABLE
                    | EXIT_REASON_UNKNOWN_BLOCK
                    | EXIT_REASON_UNKNOWN_PC_VALUE
                    | EXIT_REASON_UNKNOWN_RESUME_ADDRESS
                    | EXIT_REASON_MALFORMED_RETURN
                    | EXIT_REASON_MALFORMED_INDIRECT_CALL => {
                        if fast_mode && self.machine.running() {
                            return Err(Error::External(format!(
                                "Fast mode exit with: {}",
                                result
                            )));
                        }
                    }
                    EXIT_REASON_BARE_CALL_EXIT => {
                        if self.machine.running() {
                            return Err(Error::External(
                                "Machine is still running but bare call triggers exiting!"
                                    .to_string(),
                            ));
                        }
                    }
                    EXIT_REASON_MAX_CYCLES_EXCEEDED => {
                        return Err(Error::CyclesExceeded);
                    }
                    EXIT_REASON_CYCLES_OVERFLOW => {
                        return Err(Error::CyclesOverflow);
                    }
                    _ => return Err(Error::Asm(result)),
                }
            } else {
                self.step()?;
            }
        }
        Ok(self.machine.exit_code())
    }

    pub fn step(&mut self) -> Result<(), Error> {
        let pc = *self.machine.pc();
        let instruction = self.decoder.decode(self.machine.memory_mut(), pc)?;
        execute(instruction, &mut self.machine)
    }

    fn env<'b>(&self) -> LlvmAotMachineEnv<'b> {
        LlvmAotMachineEnv {
            data: self as *const LlvmAotMachine as *mut LlvmAotMachine,
            ecall: bare_ecall,
            ebreak: bare_ebreak,
            query_function: bare_query_function,
            interpret: bare_interpret,
        }
    }
}

#[repr(C)]
pub struct LlvmAotCoreMachineData<'a> {
    pub pc: u64,
    pub memory: *mut u8,
    pub cycles: u64,
    pub max_cycles: u64,
    pub registers: [u64; RISCV_GENERAL_REGISTER_NUMBER],
    pub last_ra: u64,
    pub next_pc: u64,
    pub runtime_flags: u8,
    pub exit_aot_reason: u8,
    pub jmpbuf: [u64; 5],

    pub env: *const LlvmAotMachineEnv<'a>,
}

pub struct LlvmAotCoreMachine<'a> {
    pub data: LlvmAotCoreMachineData<'a>,

    pub allocation: Allocation,
    pub cached_region: Option<Region>,
}

impl<'a> LlvmAotCoreMachine<'a> {
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
            cycles: 0,
            max_cycles: 0,
            registers: [0u64; RISCV_GENERAL_REGISTER_NUMBER],
            last_ra: 0,
            next_pc: 0,
            runtime_flags: 0,
            exit_aot_reason: 0,
            jmpbuf: [0u64; 5],
            env: ptr::null(),
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

impl<'a> CoreMachine for LlvmAotCoreMachine<'a> {
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

impl<'a> SupportMachine for LlvmAotCoreMachine<'a> {
    fn cycles(&self) -> u64 {
        self.data.cycles
    }

    fn set_cycles(&mut self, cycles: u64) {
        self.data.cycles = cycles;
    }

    fn max_cycles(&self) -> u64 {
        self.data.max_cycles
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

impl<'a> Memory for LlvmAotCoreMachine<'a> {
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
