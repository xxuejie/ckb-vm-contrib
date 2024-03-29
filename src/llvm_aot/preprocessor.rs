use super::{
    super::printer::InstructionPrinter,
    ast::{simplify, simplify_with_writes, AstMachine, Control, ExternalType, Write},
    AOT_ISA, AOT_VERSION,
};
use crate::decoder::AuxDecoder;
use ckb_vm::{
    decoder::build_decoder,
    instructions::{
        ast::{ActionOp2, Value},
        execute, extract_opcode, instruction_length, insts, is_basic_block_end_instruction, Utype,
    },
    machine::InstructionCycleFunc,
    registers::RA,
    Bytes, CoreMachine, DefaultCoreMachine, Error, Memory, Register, SparseMemory, SupportMachine,
};
use core::ops::Range;
use goblin::elf::{program_header::PT_LOAD, section_header::SHF_EXECINSTR, sym::STT_FUNC, Elf};
use log::debug;
use rust_lapper::{Interval, Lapper};
use rustc_demangle::demangle;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub range: Range<u64>,
    pub write_batches: Vec<Vec<Write>>,
    pub control: Control,
    pub insts: usize,
    pub cycles: u64,

    pub debug_lines: Vec<String>,
}

impl BasicBlock {
    pub fn possible_targets(&self) -> Vec<u64> {
        let mut r = Vec::new();
        match self.control.pc() {
            Value::Imm(v) => r.push(v),
            Value::Cond(_, v1, v2) => {
                if let Value::Imm(v) = &*v1 {
                    r.push(*v);
                }
                if let Value::Imm(v) = &*v2 {
                    r.push(*v);
                }
            }
            _ => (),
        };
        if let Some(resume_address) = self.control.call_resume_address() {
            r.push(resume_address);
        }
        r
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub range: Range<u64>,
    pub name: Option<String>,
    pub basic_blocks: Vec<BasicBlock>,
}

impl Func {
    pub fn force_name(&self, demangle_func: bool) -> String {
        self.name
            .as_ref()
            .map(|s| {
                if demangle_func {
                    demangle(s).as_str()
                } else {
                    s
                }
                .to_string()
            })
            .unwrap_or_else(|| format!("function_{:x}", self.range.start))
    }
}

/// Given a RISC-V binary, perform the initial preprocessing & analysis:
/// * Infer all functions and basic blocks within each functions
/// * Infer all function calls and merge basic blocks separated by calls
/// * Execute RISC-V instructions on AST machine, generating simple writes
/// * Simple optimizations are also perform to simplify writes
/// The generated data structure after this function will be ready for LLVM
/// code emitting.
pub fn preprocess(
    code: &Bytes,
    instruction_cycle_func: &InstructionCycleFunc,
) -> Result<Vec<Func>, Error> {
    let (mut memory, mut decoder): (SparseMemory<u64>, _) = {
        let isa = AOT_ISA;
        let version = AOT_VERSION;
        let mut dummy = DefaultCoreMachine::new(isa, version, 0);
        dummy.load_elf(&code, false)?;
        (
            dummy.take_memory(),
            AuxDecoder::new(build_decoder::<u64>(isa, version)),
        )
    };

    let mut funcs = extract_funcs(code, &mut memory, &mut decoder)?;
    let func_addresses = funcs.iter().map(|f| f.range.start).collect();

    for func in &mut funcs {
        let mut blocks = extract_basic_blocks(
            func,
            &mut memory,
            &mut decoder,
            &func_addresses,
            instruction_cycle_func,
        )?;
        for block in &mut blocks {
            build_memory_hints(block);
        }
        func.basic_blocks = blocks;
    }

    // Exclude empty functions
    funcs = funcs
        .into_iter()
        .filter(|f| !f.basic_blocks.is_empty())
        .collect();

    Ok(funcs)
}

// Extract possible functions from an ELF binary. Notice in the case that debugging
// information is missing, we can only infer the start & end of functions here, meaning
// some inferred function might actually contain more than one function.
fn extract_funcs<M: Memory>(
    code: &Bytes,
    memory: &mut M,
    decoder: &mut AuxDecoder,
) -> Result<Vec<Func>, Error> {
    let object = Elf::parse(&code)?;
    let mut sections: Vec<Range<u64>> = object
        .section_headers
        .iter()
        .filter_map(|section_header| {
            if section_header.sh_flags & u64::from(SHF_EXECINSTR) != 0 {
                Some(Range {
                    start: section_header.sh_addr,
                    end: section_header.sh_addr.wrapping_add(section_header.sh_size),
                })
            } else {
                None
            }
        })
        .rev()
        .collect();
    sections.sort_by_key(|section| section.start);

    // Remove sections that do not fall into executable sections as indicated by
    // program headers. This way we assert that when a RISC-V address maps to an LLVM
    // generated function, it will be executable without needing for checking.
    // What's remaining is just fallback which can test against executable
    // regions manually.
    let executable_headers: Vec<Range<u64>> = object
        .program_headers
        .iter()
        .filter_map(|ph| {
            if ph.p_type == PT_LOAD {
                Some(Range {
                    start: ph.p_vaddr,
                    end: ph.p_vaddr.wrapping_add(ph.p_memsz),
                })
            } else {
                None
            }
        })
        .collect();
    sections = sections
        .into_iter()
        .filter(|s| {
            executable_headers
                .iter()
                .any(|eh| s.start >= eh.start && s.end <= eh.end)
        })
        .collect();

    // Test there's no empty section
    if sections.iter().any(Range::is_empty) {
        return Err(Error::External("An empty section is found!".to_string()));
    }
    // Test no section overlaps with one another. We first sort section
    // list by start, then we test if each end is equal or less than
    // the next start.
    if sections.windows(2).any(|w| w[0].end > w[1].start) {
        return Err(Error::External(
            "A section overlaps with another one!".to_string(),
        ));
    }

    let mut funcs: Vec<Func> = sections
        .iter()
        .map(|r| Func {
            range: r.clone(),
            name: None,
            basic_blocks: vec![],
        })
        .collect();

    // The entrypoint is naturally the start of a function.
    split_func(&mut funcs, object.header.e_entry, None);

    // If a symbol table is available, use it to identify functions first.
    for i in 0..object.syms.len() {
        if let Some(sym) = object.syms.get(i) {
            if sym.st_type() == STT_FUNC {
                let name = object
                    .strtab
                    .get(sym.st_name)
                    .and_then(|r| r.map(|s| s.to_string()).ok());
                split_func(&mut funcs, sym.st_value, name);
            }
        }
    }

    // For all JALs, we can infer the callee function entry.
    for section in &sections {
        let mut pc = section.start;
        while pc < section.end {
            match decoder.decode(memory, pc) {
                Ok(instruction) => {
                    match extract_opcode(instruction) {
                        insts::OP_JAL => {
                            let i = Utype(instruction);
                            if i.rd() == RA {
                                let target_pc = pc.wrapping_add(i.immediate_s() as i64 as u64);

                                split_func(&mut funcs, target_pc, None);
                            }
                        }
                        insts::OP_FAR_JUMP_REL => {
                            let i = Utype(instruction);
                            let target_pc = pc.wrapping_add(i.immediate_s() as i64 as u64) & (!1);

                            split_func(&mut funcs, target_pc, None);
                        }
                        insts::OP_FAR_JUMP_ABS => {
                            let i = Utype(instruction);
                            let target_pc = (i.immediate_s() as i64 as u64) & (!1);

                            split_func(&mut funcs, target_pc, None);
                        }
                        _ => (),
                    }
                    pc += instruction_length(instruction) as u64;
                }
                Err(Error::InvalidInstruction {
                    pc: _,
                    instruction: i,
                }) => {
                    if i == 0 {
                        // Skip alignment data
                        let mut dummy_end = pc + 2;
                        while dummy_end < section.end && memory.execute_load16(dummy_end)? == 0 {
                            dummy_end += 2;
                        }
                        pc = dummy_end;
                    } else {
                        // Skip invalid instruction as well
                        let len = if i & 0x3 == 0x3 { 4 } else { 2 };
                        pc += len;
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    Ok(funcs)
}

// Given a known function starting address, search for existing inferred functions,
// finding a function A where the given address lies within the inferred function A.
// In such a case, the inferred function A is actually the combination of more than
// one function, we can split such a function A into two.
fn split_func(funcs: &mut Vec<Func>, addr: u64, name: Option<String>) {
    if let Some(idx) = funcs
        .iter()
        .position(|r| addr > r.range.start && addr < r.range.end)
    {
        funcs.splice(
            idx..idx + 1,
            [
                Func {
                    range: Range {
                        start: funcs[idx].range.start,
                        end: addr,
                    },
                    name: funcs[idx].name.clone(),
                    basic_blocks: vec![],
                },
                Func {
                    range: Range {
                        start: addr,
                        end: funcs[idx].range.end,
                    },
                    name,
                    basic_blocks: vec![],
                },
            ],
        );
    } else if let (Some(name), Some(idx)) = (name, funcs.iter().position(|r| addr == r.range.start))
    {
        funcs[idx].name = Some(name);
    }
}

// For each given function, infer and extract the basic blocks within those functions.
// Notice a side effect when inferring basic blocks, is that we also get to *execute*
// the RISC-V instructions, simplifying RISC-V functions into a simplified AST for code
// generation. After this step, we will only need to deal with simple AST, minimal RISC-V
// knowledge will be needed in the code generation phase.
fn extract_basic_blocks<M: Memory>(
    func: &Func,
    memory: &mut M,
    decoder: &mut AuxDecoder,
    func_addresses: &HashSet<u64>,
    instruction_cycle_func: &InstructionCycleFunc,
) -> Result<Vec<BasicBlock>, Error> {
    let mut blocks = vec![];
    let mut targets: HashSet<u64> = HashSet::default();
    let mut pc = func.range.start;
    while pc < func.range.end {
        let (block, block_end) = parse_basic_block(
            memory,
            decoder,
            pc,
            func.range.end,
            func_addresses,
            instruction_cycle_func,
        )?;
        pc = block_end;

        if let Some(block) = block {
            // println!("Block: {:x}", block.range.start);
            for target in block.possible_targets() {
                // println!("Target: {:x}", target);
                targets.insert(target);
            }
            blocks.push(block);
        }
    }
    // Some optimizations here:
    // 1. Gather all block's possible targets in the above step, if a target
    // lies in the middle of basic block A, split basic block A into 2 blocks
    // separated by the target.
    for target in targets.iter() {
        if let Some(i) = blocks
            .iter()
            .position(|b| *target > b.range.start && *target < b.range.end)
        {
            debug!(
                "Splitting block 0x{:x}-0x{:x} at range 0x{:x}",
                blocks[i].range.start, blocks[i].range.end, *target
            );
            let splitted_blocks = {
                let block = &blocks[i];

                let (block1, _) = parse_basic_block(
                    memory,
                    decoder,
                    block.range.start,
                    *target,
                    func_addresses,
                    instruction_cycle_func,
                )?;
                let (block2, _) = parse_basic_block(
                    memory,
                    decoder,
                    *target,
                    block.range.end,
                    func_addresses,
                    instruction_cycle_func,
                )?;

                let mut splitted_blocks = vec![];
                if let Some(block) = block1 {
                    splitted_blocks.push(block);
                }
                if let Some(block) = block2 {
                    splitted_blocks.push(block);
                }
                splitted_blocks
            };

            blocks.splice(i..(i + 1), splitted_blocks);
        }
    }
    blocks.sort_by_key(|b| b.range.start);
    Ok(blocks)
}

// Given a starting address and a maximum ending address(in case of branches,
// the generated basic block will not end at the provided maximum ending address),
// decode the RISC-V instructions from memory, and *execute* them for generated AST.
// This is seperated from *extract_basic_blocks*, in that we will need to execute this
// function on the same address for more than once due to the way some basic blocks are
// inferred. See *extract_basic_blocks* for more details.
fn parse_basic_block<M: Memory>(
    memory: &mut M,
    decoder: &mut AuxDecoder,
    block_start: u64,
    block_maximum_end: u64,
    func_addresses: &HashSet<u64>,
    instruction_cycle_func: &InstructionCycleFunc,
) -> Result<(Option<BasicBlock>, u64), Error> {
    let mut insts = vec![];
    let mut debug_lines = vec![];
    let mut invalid = false;
    let mut pc = block_start;
    while pc < block_maximum_end {
        match decoder.decode(memory, pc) {
            Ok(instruction) => {
                insts.push(instruction);
                debug_lines.push(format!(
                    "0x{:x}: {}",
                    pc,
                    InstructionPrinter::new(instruction.try_into()?)
                ));
                pc += instruction_length(instruction) as u64;
                if is_basic_block_end_instruction(instruction) {
                    break;
                }
            }
            Err(Error::InvalidInstruction {
                pc: _,
                instruction: i,
            }) if i == 0 => {
                // Terminate previous basic block first.
                if pc != block_start {
                    break;
                }
                // Skip alignment data
                let mut dummy_end = pc + 2;
                while dummy_end < block_maximum_end
                    && memory.execute_load16(dummy_end).expect("load16") == 0
                {
                    dummy_end += 2;
                }
                pc = dummy_end;
                debug!("Skipping alignment data at 0x{:x}-0x{:x}", block_start, pc);
                return Ok((None, pc));
            }
            Err(Error::InvalidInstruction {
                pc: _,
                instruction: i,
            }) => {
                // Terminate previous valid basic block first.
                if !invalid && pc != block_start {
                    break;
                }
                // Invalid instruction met, we will still skip them following
                // RISC-V encoding rules. This ensures the AOT engine works with
                // slowpath instructions.
                let len = if i & 0x3 == 0x3 { 4 } else { 2 };
                pc += len;
                invalid = true;
            }
            Err(e) => return Err(e),
        }
    }

    assert!(
        !insts.is_empty(),
        "Parsed basic block is empty from 0x{:x}-0x{:x}",
        block_start,
        pc
    );

    if invalid {
        debug!(
            "Skipping basic block containing unknown instructions at 0x{:x}-0x{:x}",
            block_start, pc
        );
        // In case of invalid instructions, skip the basic block. Examples
        // are slowpath instructions (V), they will not be AOTed for now,
        // the included interpreter will take care of them.
        return Ok((None, pc));
    }

    let block_end = pc;
    let mut ast_machine = AstMachine::default();
    ast_machine.update_pc(Value::from_u64(block_start));
    ast_machine.commit_pc();
    let len = insts.len();
    let mut cycles = 0;
    let mut write_batches = Vec::with_capacity(len - 1);
    // The above code has asserted that insts cannot be empty
    let last_inst = insts.pop().unwrap();
    for inst in insts {
        cycles += instruction_cycle_func(inst);
        execute(inst, &mut ast_machine)?;
        write_batches.push(ast_machine.take_writes());
        ast_machine.reset_registers();
    }

    cycles += instruction_cycle_func(last_inst);
    execute(last_inst, &mut ast_machine)?;
    let last_pc = simplify_with_writes(
        ast_machine.pc(),
        &write_batches.iter().flatten().collect::<Vec<&Write>>(),
    );
    let last_writes = ast_machine.take_writes();
    let control = if is_basic_block_end_instruction(last_inst) {
        match ast_machine.take_control() {
            Some(Control::Ecall { .. }) => Control::Ecall { pc: last_pc },
            Some(Control::Ebreak { .. }) => Control::Ebreak { pc: last_pc },
            _ => {
                if let Some(callee) =
                    end_with_call(block_end, &last_pc, &last_writes, func_addresses)
                {
                    Control::Call {
                        address: callee,
                        writes: last_writes,
                    }
                } else if end_with_possible_indirect_call(block_end, &last_pc, &last_writes) {
                    Control::IndirectCall {
                        pc: last_pc,
                        writes: last_writes,
                    }
                } else if let Some(callee) =
                    end_with_tail_call(&last_pc, &last_writes, func_addresses)
                {
                    Control::Tailcall { address: callee }
                } else if is_possible_return(&last_pc) {
                    Control::Return {
                        pc: last_pc,
                        writes: last_writes,
                    }
                } else {
                    Control::Jump {
                        pc: last_pc,
                        writes: last_writes,
                    }
                }
            }
        }
    } else {
        write_batches.push(last_writes);
        Control::Jump {
            pc: last_pc,
            writes: vec![],
        }
    };

    let block = BasicBlock {
        range: Range {
            start: block_start,
            end: block_end,
        },
        write_batches,
        control,
        insts: len,
        cycles,
        debug_lines,
    };

    Ok((Some(block), pc))
}

fn is_possible_return(pc: &Value) -> bool {
    if let Value::Op2(ActionOp2::Bitand, v1, v2) = pc {
        if let (Value::Register(r), Value::Imm(0xfffffffffffffffe)) = (&**v1, &**v2) {
            if *r == RA {
                return true;
            }
        }
    }
    false
}

fn end_with_call(
    block_end: u64,
    last_pc: &Value,
    last_writes: &[Write],
    func_addresses: &HashSet<u64>,
) -> Option<u64> {
    if let Some(last_write) = last_writes.last() {
        if last_write.is_simple_register_write(RA, block_end) {
            if let Value::Imm(target) = last_pc {
                if func_addresses.contains(target) {
                    return Some(*target);
                }
            }
        }
    }
    None
}

fn end_with_possible_indirect_call(block_end: u64, last_pc: &Value, last_writes: &[Write]) -> bool {
    if let Some(last_write) = last_writes.last() {
        if last_write.is_simple_register_write(RA, block_end) {
            if let Value::Op2(ActionOp2::Bitand, _v1, v2) = last_pc {
                if let Value::Imm(0xfffffffffffffffe) = &**v2 {
                    return true;
                }
            }
        }
    }
    false
}

fn end_with_tail_call(
    last_pc: &Value,
    last_writes: &[Write],
    func_addresses: &HashSet<u64>,
) -> Option<u64> {
    if last_writes.is_empty() {
        if let Value::Imm(target) = last_pc {
            if func_addresses.contains(target) {
                return Some(*target);
            }
        }
    }
    None
}

struct MemoryRange {
    lapper: Lapper<u128, ()>,
    batch_index: usize,
    register: usize,
}

impl MemoryRange {
    fn new(batch_index: usize, register: usize) -> Self {
        Self {
            lapper: Lapper::new(vec![]),
            batch_index,
            register,
        }
    }

    fn insert(&mut self, offset: u128, size: u64) {
        self.lapper.insert(Interval {
            start: offset,
            stop: offset + size as u128,
            val: (),
        });
    }

    fn iter(&self) -> impl Iterator<Item = (u64, u64)> {
        self.lapper
            .iter()
            .map(|interval| {
                (
                    interval.start as u64,
                    (interval.stop - interval.start) as u64,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[derive(Default)]
struct MemoryMerger {
    // (reg, batch_start) -> MemoryRange
    ranges: HashMap<(usize, usize), MemoryRange>,
    valid_batch_start: HashMap<usize, usize>,
    // When we have writes like the following:
    //
    // * Register[ a1 ] = (0x8050 + Reg(sp))
    //
    // We want hints using a1 as base register can be transformed
    // into hints using sp directly. Virtual register are used to handle
    // this in a series of register writes.
    // Note that one register might also shadow itself:
    //
    // * Register[ sp ] = (Reg(sp) + 0xffffffffffffff20)
    //
    // One can think of batch_index as a sort of timestamp amongst register writes,
    // the immediate preceeding batch_start, can be viewed as the "version" for a
    // particular real register value.
    //
    // virtual_reg -> (real_reg, batch_start, additional_offset)
    virtual_regs: HashMap<usize, (usize, usize, u128)>,
}

impl MemoryMerger {
    fn insert(&mut self, batch_index: usize, reg: usize, offset: u64, size: u64) {
        let offset = offset as u128;
        let (real_reg, real_batch_start, additional_offset) = match self.virtual_regs.get(&reg) {
            Some((reg, batch_start, offset)) => (*reg, *batch_start, *offset),
            None => (
                reg,
                self.valid_batch_start.get(&reg).cloned().unwrap_or(0),
                0,
            ),
        };
        self.ranges
            .entry((real_reg, real_batch_start))
            .or_insert_with(|| {
                MemoryRange::new(std::cmp::min(batch_index, real_batch_start), real_reg)
            })
            .insert(offset + additional_offset, size);
    }

    fn seal(&mut self, batch_index: usize, writes: &HashMap<usize, Value>) {
        for (reg, value) in writes {
            if let Some((real_reg, offset)) = extract_register_n_offset(&simplify(value)) {
                let offset = offset as u128;
                // Nested simplifying, since writes are iteratively processed,
                // loops are not needed here.
                let (real_reg, real_batch_start, offset) = if let Some((
                    a_reg,
                    a_batch_start,
                    a_offset,
                )) = self.virtual_regs.get(&real_reg)
                {
                    (*a_reg, *a_batch_start, a_offset + offset)
                } else {
                    let real_batch_start =
                        self.valid_batch_start.get(&real_reg).cloned().unwrap_or(0);
                    (real_reg, real_batch_start, offset)
                };
                self.virtual_regs
                    .insert(*reg, (real_reg, real_batch_start, offset));
            }
        }
        // Make sure each write batch is processed atomically
        for (reg, _) in writes {
            self.valid_batch_start.insert(*reg, batch_index + 1);
        }
    }

    fn finish(mut self) -> Vec<MemoryRange> {
        self.ranges
            .drain()
            .map(|(_, mut r)| {
                r.lapper.merge_overlaps();
                r
            })
            .collect()
    }
}

fn extract_register_n_offset(value: &Value) -> Option<(usize, u64)> {
    if let Value::Op2(ActionOp2::Add, lhs, rhs) = value {
        match (&**lhs, &**rhs) {
            (Value::Register(reg), Value::Imm(imm)) => {
                return Some((*reg, *imm));
            }
            (Value::Imm(imm), Value::Register(reg)) => {
                return Some((*reg, *imm));
            }
            _ => (),
        }
    }
    None
}

fn recursively_extracting_load(value: &mut Value, results: &mut Vec<(usize, u64, u64)>) -> bool {
    match value {
        Value::Load(v, size) => {
            if let Value::Register(reg) = &**v {
                results.push((*reg, 0, *size as u64));
                false
            } else if let Value::Imm(imm) = &**v {
                results.push((0, *imm, *size as u64));
                false
            } else if let Some((reg, offset)) = extract_register_n_offset(&**v) {
                results.push((reg, offset, *size as u64));
                false
            } else {
                // When a load cannot be covered by hint, wrap it with UnhintedLoad
                // so as to inform the emitter
                *value = Value::External(value.clone().into(), ExternalType::UnhintedLoad as u64);
                true
            }
        }
        Value::Op1(op, v) => {
            let mut v2 = (&**v).clone();
            let changed = recursively_extracting_load(&mut v2, results);
            if changed {
                *value = Value::Op1(*op, v2.into());
            }
            changed
        }
        Value::Op2(op, lhs, rhs) => {
            let mut lhs2 = (&**lhs).clone();
            let lhs_changed = recursively_extracting_load(&mut lhs2, results);
            let mut rhs2 = (&**rhs).clone();
            let rhs_changed = recursively_extracting_load(&mut rhs2, results);
            let changed = lhs_changed || rhs_changed;
            if changed {
                *value = Value::Op2(*op, lhs2.into(), rhs2.into());
            }
            changed
        }
        Value::SignOp2(op, lhs, rhs, sign) => {
            let mut lhs2 = (&**lhs).clone();
            let lhs_changed = recursively_extracting_load(&mut lhs2, results);
            let mut rhs2 = (&**rhs).clone();
            let rhs_changed = recursively_extracting_load(&mut rhs2, results);
            let changed = lhs_changed || rhs_changed;
            if changed {
                *value = Value::SignOp2(*op, lhs2.into(), rhs2.into(), *sign);
            }
            changed
        }
        Value::Cond(c, t, f) => {
            let mut c2 = (&**c).clone();
            let c_changed = recursively_extracting_load(&mut c2, results);
            let mut t2 = (&**t).clone();
            let t_changed = recursively_extracting_load(&mut t2, results);
            let mut f2 = (&**f).clone();
            let f_changed = recursively_extracting_load(&mut f2, results);
            let changed = c_changed || t_changed || f_changed;
            if changed {
                *value = Value::Cond(c2.into(), t2.into(), f2.into());
            }
            changed
        }
        _ => false,
    }
}

fn extract_memory_loads(writes: &mut [Write]) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    for write in writes {
        match write {
            Write::Lr { value } => {
                recursively_extracting_load(value, &mut result);
            }
            Write::Memory { value, .. } => {
                recursively_extracting_load(value, &mut result);
            }
            Write::Register { value, .. } => {
                recursively_extracting_load(value, &mut result);
            }
            _ => (),
        }
    }
    result
}

fn extract_memory_writes(writes: &mut [Write]) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    for write in writes {
        match write {
            Write::Memory {
                address,
                size,
                hinted,
                ..
            } => {
                if let Some((reg, offset)) = extract_register_n_offset(address) {
                    result.push((reg, offset, *size as u64));
                    *hinted = true;
                } else if let Value::Imm(imm) = address {
                    result.push((0, *imm, *size as u64));
                    *hinted = true;
                }
            }
            _ => (),
        }
    }
    result
}

fn extract_register_writes(writes: &[Write]) -> HashMap<usize, Value> {
    let mut result = HashMap::default();
    for write in writes {
        match write {
            Write::Register { index, value } => {
                result.insert(*index, value.clone());
            }
            _ => (),
        };
    }
    result
}

fn build_memory_hints(basic_block: &mut BasicBlock) {
    let mut read_merger = MemoryMerger::default();
    let mut write_merger = MemoryMerger::default();

    for (i, batch) in basic_block.write_batches.iter_mut().enumerate() {
        for (reg, offset, size) in extract_memory_loads(batch) {
            read_merger.insert(i, reg, offset, size);
        }
        for (reg, offset, size) in extract_memory_writes(batch) {
            write_merger.insert(i, reg, offset, size);
        }
        // Since each write batch is considered to be atomic, we will
        // only seal ranges after all register reads are performed.
        let register_writes = extract_register_writes(batch);
        read_merger.seal(i, &register_writes);
        write_merger.seal(i, &register_writes);
    }

    for range in read_merger.finish() {
        for (offset, size) in range.iter() {
            basic_block.write_batches[range.batch_index].insert(
                0,
                Write::Hint {
                    reg: range.register,
                    offset,
                    size,
                    write: false,
                },
            );
        }
    }
    for range in write_merger.finish() {
        for (offset, size) in range.iter() {
            basic_block.write_batches[range.batch_index].insert(
                0,
                Write::Hint {
                    reg: range.register,
                    offset,
                    size,
                    write: true,
                },
            );
        }
    }
}
