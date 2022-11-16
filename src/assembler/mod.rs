pub mod b;
pub mod i;
pub mod m;
pub mod rvc;
mod utils;

use ckb_vm::{
    ckb_vm_definitions::{
        instructions::{self as opcodes, instruction_opcode_name, InstructionOpcode},
        registers::REGISTER_ABI_NAMES,
    },
    instructions::{
        blank_instruction, tagged::TaggedInstruction, Itype, RegisterIndex, Rtype, Stype, Utype,
    },
    Error, Register,
};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::str::FromStr;

/// Parse RISC-V instructions in text format into ckb-vm instruction formats
pub fn parse<R: Register>(input: &str) -> Result<Vec<TaggedInstruction>, Error> {
    let mut insts = Vec::new();

    // First, ensure multiple lines are processed correctly
    let lines = input.trim().split('\n');
    for line in lines {
        let mut stream = InstrStream::create(line);
        let opcode_name = stream.next_token()?.to_lowercase();
        if let Some((opcode, f)) = PARSER_FUNCS.get(&opcode_name.replace(".", "")) {
            let inst = f(*opcode, &mut stream)?;
            insts.push(inst);
        } else if let Some(parsed_insts) = parse_pseudoinstructions::<R>(&opcode_name, &mut stream)?
        {
            insts.extend(parsed_insts);
        } else {
            return Err(Error::External(format!(
                "Invalid instruction {}!",
                opcode_name
            )));
        }
        if stream.has_token() {
            return Err(Error::External(format!(
                "Trailing token exists for \"{}\"",
                line
            )));
        }
    }

    Ok(insts)
}

pub type InstructionAssembler = fn(inst: &TaggedInstruction) -> Option<Vec<u8>>;

#[derive(Default)]
pub struct Assembler {
    factories: Vec<InstructionAssembler>,
}

impl Assembler {
    pub fn new() -> Assembler {
        Assembler {
            factories: Vec::new(),
        }
    }

    pub fn add_assembler_factory(&mut self, factory: InstructionAssembler) {
        self.factories.push(factory);
    }

    pub fn assemble(&self, inst: &TaggedInstruction) -> Result<Vec<u8>, Error> {
        for factory in &self.factories {
            if let Some(binary) = factory(inst) {
                return Ok(binary);
            }
        }
        Err(Error::External(format!("Invalid instruction {}", inst)))
    }
}

/// Assemble ckb-vm instructions into binary bytes
pub fn assemble<R: Register>(insts: &[TaggedInstruction]) -> Result<Vec<u8>, Error> {
    let mut assembler = Assembler::new();
    // Order here matters, rvc should be invoked first
    assembler.add_assembler_factory(rvc::assembler::<R>);
    assembler.add_assembler_factory(i::assembler::<R>);
    assembler.add_assembler_factory(m::assembler::<R>);
    assembler.add_assembler_factory(b::assembler::<R>);

    let mut content = Vec::new();
    for inst in insts {
        content.extend(assembler.assemble(inst)?);
    }

    Ok(content)
}

lazy_static! {
    static ref TOKENS_RE: Regex = Regex::new(r#"[^\s,\(\)]+"#).expect("regex");
    static ref COMMENT_RE: Regex = Regex::new("#.*$").expect("regex");
    static ref REG_NAME_RE: Regex = Regex::new("^x([0-9]{1,2})$").expect("regex");
    static ref PARSER_FUNCS: HashMap<String, (InstructionOpcode, ParserFunc)> = {
        let mut m = HashMap::new();
        // noarg types
        for op in [opcodes::OP_ECALL, opcodes::OP_EBREAK, opcodes::OP_FENCEI] {
            m.insert(instruction_opcode_name(op).to_lowercase().to_string(), (op, parse_noarg_rtype as ParserFunc));
        }
        // B types
        for op in [opcodes::OP_BEQ, opcodes::OP_BNE, opcodes::OP_BLT,
            opcodes::OP_BGE, opcodes::OP_BLTU, opcodes::OP_BGEU]
        {
            m.insert(instruction_opcode_name(op).to_lowercase().to_string(), (op, parse_btype as ParserFunc));
        }

        for op in opcodes::OP_UNLOADED..=opcodes::OP_CUSTOM_TRACE_END {
            let name = instruction_opcode_name(op).to_lowercase().to_string();
            if !m.contains_key(&name) {
                let f = match TaggedInstruction::try_from(blank_instruction(op)) {
                    Ok(TaggedInstruction::Rtype(_)) => Some(parse_rtype as ParserFunc),
                    Ok(TaggedInstruction::Itype(_)) => Some(parse_itype as ParserFunc),
                    Ok(TaggedInstruction::Stype(_)) => Some(parse_stype as ParserFunc),
                    Ok(TaggedInstruction::Utype(_)) => Some(parse_utype as ParserFunc),
                    _ => None,
                };
                if let Some(f) = f {
                    m.insert(name, (op, f));
                }
            }
        }
        m
    };
}

type ParserFunc = fn(InstructionOpcode, &mut InstrStream) -> Result<TaggedInstruction, Error>;

fn parse_itype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    // https://github.com/riscv-non-isa/riscv-asm-manual/blob/a9945e1db585abaed594d55ff84e87dd93e21723/riscv-asm.md#-a-listing-of-standard-risc-v-pseudoinstructions
    if opcode == opcodes::OP_JALR && stream.remaining() == 1 {
        let rs1 = stream.next_register()?;
        return Ok(Itype::new_u(opcode, 1, rs1, 0).into());
    }
    let rd = stream.next_register()?;
    let (rs1, immediate) = match (stream.peek_register(), stream.peek_number()) {
        (Ok(rs1), _) => {
            stream.next_token()?;
            (rs1, stream.next_number()?)
        }
        (_, Ok(immediate)) => {
            stream.next_token()?;
            (stream.next_register()?, immediate)
        }
        _ => {
            return Err(Error::External(format!(
                "Invalid token: {}",
                stream.peek_token()?
            )))
        }
    };
    Ok(Itype::new_u(opcode, rd, rs1, immediate as u32).into())
}

fn parse_rtype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    if opcode == opcodes::OP_FENCE && stream.remaining() == 0 {
        return Ok(Rtype::new(opcode, 0, 0b1111, 0b1111).into());
    }
    let rd = stream.next_register()?;
    let rs1 = stream.next_register()?;
    let rs2 = stream.next_register()?;
    Ok(Rtype::new(opcode, rd, rs1, rs2).into())
}

fn parse_btype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    let rs1 = stream.next_register()?;
    let (rs2, immediate) = match (stream.peek_register(), stream.peek_number()) {
        (Ok(rs2), _) => {
            stream.next_token()?;
            (rs2, stream.next_number()?)
        }
        (_, Ok(immediate)) => {
            stream.next_token()?;
            (stream.next_register()?, immediate)
        }
        _ => {
            return Err(Error::External(format!(
                "Invalid token: {}",
                stream.peek_token()?
            )))
        }
    };
    Ok(Stype::new_u(opcode, immediate as u32, rs1, rs2).into())
}

fn parse_stype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    let rs1 = stream.next_register()?;
    let (rd, immediate) = match (stream.peek_register(), stream.peek_number()) {
        (Ok(rd), _) => {
            stream.next_token()?;
            (rd, stream.next_number()?)
        }
        (_, Ok(immediate)) => {
            stream.next_token()?;
            (stream.next_register()?, immediate)
        }
        _ => {
            return Err(Error::External(format!(
                "Invalid token: {}",
                stream.peek_token()?
            )))
        }
    };
    Ok(Stype::new_u(opcode, immediate as u32, rd, rs1).into())
}

fn parse_utype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    // https://github.com/riscv-non-isa/riscv-asm-manual/blob/a9945e1db585abaed594d55ff84e87dd93e21723/riscv-asm.md#-a-listing-of-standard-risc-v-pseudoinstructions
    if opcode == opcodes::OP_JAL && stream.remaining() == 1 {
        let immediate = stream.next_number()?;
        return Ok(Utype::new(opcode, 1, immediate as u32).into());
    }
    let rd = stream.next_register()?;
    let mut immediate = stream.next_number()?;
    if opcode == opcodes::OP_LUI || opcode == opcodes::OP_AUIPC {
        immediate = immediate << 12;
    }
    Ok(Utype::new(opcode, rd, immediate as u32).into())
}

fn parse_noarg_rtype(
    opcode: InstructionOpcode,
    _stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
    Ok(Rtype::new(opcode, 0, 0, 0).into())
}

fn parse_pseudoinstructions<R: Register>(
    opcode_name: &str,
    stream: &mut InstrStream,
) -> Result<Option<Vec<TaggedInstruction>>, Error> {
    match opcode_name {
        "lla" => {
            let rd = stream.next_register()?;
            let offset = stream.next_number()? as u32;
            Ok(Some(vec![
                Utype::new(opcodes::OP_AUIPC, rd, offset >> 12).into(),
                Itype::new_u(opcodes::OP_ADDI, rd, rd, offset & 0xFFF).into(),
            ]))
        }
        "nop" => Ok(Some(vec![Itype::new_u(opcodes::OP_ADDI, 0, 0, 0).into()])),
        "li" => {
            let rd = stream.next_register()?;
            let imm = stream.next_number_with_mask(0xFFF)?;
            Ok(Some(vec![Itype::new_u(
                opcodes::OP_ADDI,
                rd,
                0,
                imm as u32,
            )
            .into()]))
        }
        "mv" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Itype::new_u(opcodes::OP_ADDI, rd, rs, 0).into()]))
        }
        "not" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(
                vec![Itype::new_s(opcodes::OP_XORI, rd, rs, -1).into()],
            ))
        }
        "neg" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Rtype::new(opcodes::OP_SUB, rd, 0, rs).into()]))
        }
        "negw" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Rtype::new(opcodes::OP_SUBW, rd, 0, rs).into()]))
        }
        "sext.b" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![
                Itype::new_u(opcodes::OP_SLLI, rd, rs, R::BITS as u32 - 8).into(),
                Itype::new_u(opcodes::OP_SRAI, rd, rd, R::BITS as u32 - 8).into(),
            ]))
        }
        "sext.h" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![
                Itype::new_u(opcodes::OP_SLLI, rd, rs, R::BITS as u32 - 16).into(),
                Itype::new_u(opcodes::OP_SRAI, rd, rd, R::BITS as u32 - 16).into(),
            ]))
        }
        "sext.w" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(
                vec![Itype::new_u(opcodes::OP_ADDIW, rd, rs, 0).into()],
            ))
        }
        "zext.b" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![
                Itype::new_u(opcodes::OP_ANDI, rd, rs, 255).into()
            ]))
        }
        "zext.h" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![
                Itype::new_u(opcodes::OP_SLLI, rd, rs, R::BITS as u32 - 16).into(),
                Itype::new_u(opcodes::OP_SRLI, rd, rd, R::BITS as u32 - 16).into(),
            ]))
        }
        "zext.w" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![
                Itype::new_u(opcodes::OP_SLLI, rd, rs, R::BITS as u32 - 32).into(),
                Itype::new_u(opcodes::OP_SRLI, rd, rd, R::BITS as u32 - 32).into(),
            ]))
        }
        "seqz" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(
                vec![Itype::new_u(opcodes::OP_SLTIU, rd, rs, 1).into()],
            ))
        }
        "snez" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Rtype::new(opcodes::OP_SLTU, rd, 0, rs).into()]))
        }
        "sltz" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Rtype::new(opcodes::OP_SLT, rd, rs, 0).into()]))
        }
        "sgtz" => {
            let rd = stream.next_register()?;
            let rs = stream.next_register()?;
            Ok(Some(vec![Rtype::new(opcodes::OP_SLT, rd, 0, rs).into()]))
        }
        "beqz" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BEQ,
                offset as u32,
                rs,
                0,
            )
            .into()]))
        }
        "bnez" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BNE,
                offset as u32,
                rs,
                0,
            )
            .into()]))
        }
        "blez" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BGE,
                offset as u32,
                0,
                rs,
            )
            .into()]))
        }
        "bgez" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BGE,
                offset as u32,
                rs,
                0,
            )
            .into()]))
        }
        "bltz" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BLT,
                offset as u32,
                rs,
                0,
            )
            .into()]))
        }
        "bgtz" => {
            let rs = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BLT,
                offset as u32,
                0,
                rs,
            )
            .into()]))
        }
        "bgt" => {
            let rs = stream.next_register()?;
            let rt = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BLT,
                offset as u32,
                rt,
                rs,
            )
            .into()]))
        }
        "ble" => {
            let rs = stream.next_register()?;
            let rt = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BGE,
                offset as u32,
                rt,
                rs,
            )
            .into()]))
        }
        "bgtu" => {
            let rs = stream.next_register()?;
            let rt = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BLTU,
                offset as u32,
                rt,
                rs,
            )
            .into()]))
        }
        "bleu" => {
            let rs = stream.next_register()?;
            let rt = stream.next_register()?;
            let offset = stream.next_number_with_mask(0x1FFE)?;
            Ok(Some(vec![Stype::new_u(
                opcodes::OP_BGEU,
                offset as u32,
                rt,
                rs,
            )
            .into()]))
        }
        "j" => Ok(Some(vec![Utype::new(
            opcodes::OP_JAL,
            0,
            stream.next_number()? as u32,
        )
        .into()])),
        "jr" => Ok(Some(vec![Itype::new_u(
            opcodes::OP_JALR,
            0,
            stream.next_register()?,
            0,
        )
        .into()])),
        "ret" => Ok(Some(vec![Itype::new_u(opcodes::OP_JALR, 0, 1, 0).into()])),
        "call" => {
            let offset = stream.next_number()? as u32;
            Ok(Some(vec![
                Utype::new(opcodes::OP_AUIPC, 6, offset >> 12).into(),
                Itype::new_u(opcodes::OP_JALR, 1, 6, offset & 0xFFF).into(),
            ]))
        }
        "tail" => {
            let offset = stream.next_number()? as u32;
            Ok(Some(vec![
                Utype::new(opcodes::OP_AUIPC, 6, offset >> 12).into(),
                Itype::new_u(opcodes::OP_JALR, 0, 6, offset & 0xFFF).into(),
            ]))
        }
        _ => Ok(None),
    }
}

struct InstrStream<'a> {
    current: usize,
    tokens: Vec<&'a str>,
}

impl<'a> InstrStream<'a> {
    fn create(input: &'a str) -> Self {
        let tokens = input
            .trim()
            .split('\n')
            // For each line, # starts a comment till the end of line
            .map(|l| {
                if let Some(m) = COMMENT_RE.find(l) {
                    &l[0..m.start()]
                } else {
                    l
                }
            })
            // Now we exact all valuable terms
            .flat_map(|l| {
                TOKENS_RE
                    .captures_iter(l)
                    .map(|c| c.get(0))
                    .filter(|c| c.is_some())
                    .map(|c| c.unwrap().as_str())
            })
            .collect();

        Self { tokens, current: 0 }
    }

    fn remaining(&self) -> usize {
        self.tokens.len() - self.current
    }

    fn has_token(&self) -> bool {
        self.remaining() > 0
    }

    fn peek_token(&self) -> Result<&'a str, Error> {
        if self.current >= self.tokens.len() {
            return Err(Error::External("No available tokens!".to_string()));
        }
        Ok(self.tokens[self.current])
    }

    fn next_token(&mut self) -> Result<&'a str, Error> {
        let token = self.peek_token()?;
        self.current += 1;
        Ok(token)
    }

    fn peek_register(&self) -> Result<RegisterIndex, Error> {
        let token = self.peek_token()?;
        if let Some(m) = REG_NAME_RE.captures(token) {
            if let Some(i) = m.get(1) {
                let i = RegisterIndex::from_str(i.as_str())
                    .map_err(|e| Error::External(format!("Number parsing error: {}", e)))?;
                if i < REGISTER_ABI_NAMES.len() {
                    return Ok(i);
                }
            }
        }
        if let Some(p) = REGISTER_ABI_NAMES.iter().position(|n| n == &token) {
            return Ok(p);
        }
        Err(Error::External(format!("Invalid register {}", token)))
    }

    fn next_register(&mut self) -> Result<RegisterIndex, Error> {
        let register = self.peek_register()?;
        self.current += 1;
        Ok(register)
    }

    fn peek_number(&self) -> Result<u64, Error> {
        let token = self.peek_token()?;
        if let Some(n) = token.strip_prefix("0x") {
            u64::from_str_radix(n, 16)
        } else if let Some(n) = token.strip_prefix("0b") {
            u64::from_str_radix(n, 2)
        } else if let Some(n) = token.strip_prefix("0o") {
            u64::from_str_radix(n, 8)
        } else if token.starts_with('-') {
            i64::from_str(token).map(|n| n as u64)
        } else {
            u64::from_str(token)
        }
        .map_err(|e| Error::External(format!("Number parsing error: {}", e)))
    }

    fn next_number(&mut self) -> Result<u64, Error> {
        let number = self.peek_number()?;
        self.current += 1;
        Ok(number)
    }

    fn next_number_with_mask(&mut self, mask: u64) -> Result<u64, Error> {
        let number = self.peek_number()?;
        if number & (!mask) != 0 {
            return Err(Error::External(format!("Cannot encode number: {}", number)));
        }
        self.current += 1;
        Ok(number)
    }
}

#[cfg(test)]
mod tests {
    use ckb_vm::ckb_vm_definitions::instructions::{self as opcodes, instruction_opcode_name};
    use std::collections::HashSet;

    #[test]
    fn test_all_opcode_names_are_unique() {
        let mut count = 0;
        let mut distinct_names: HashSet<&str> = HashSet::default();
        for i in opcodes::OP_UNLOADED..=opcodes::OP_CUSTOM_TRACE_END {
            count += 1;
            distinct_names.insert(instruction_opcode_name(i));
        }

        assert_eq!(count, distinct_names.len());
    }
}
