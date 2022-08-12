pub mod i;
pub mod m;
pub mod rvc;
mod utils;

use ckb_vm::{
    instructions::{tagged::TaggedInstruction, Itype, RegisterIndex, Rtype, Stype, Utype},
    Error, Register,
};
use ckb_vm_definitions::{
    instructions::{self as opcodes, InstructionOpcode},
    registers::REGISTER_ABI_NAMES,
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
        let opcode_name = stream.next_token()?;
        if let Some((opcode, f)) = PARSER_FUNCS.get(opcode_name) {
            let inst = f(*opcode, &mut stream)?;
            insts.push(inst);
        } else if let Some(parsed_insts) = parse_pseudoinstructions::<R>(opcode_name, &mut stream)?
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
    static ref PARSER_FUNCS: HashMap<&'static str, (InstructionOpcode, ParserFunc)> = {
        let mut m = HashMap::new();
        // I type
        m.insert("lb", (opcodes::OP_LB, parse_itype as ParserFunc));
        m.insert("lh", (opcodes::OP_LH, parse_itype as ParserFunc));
        m.insert("lw", (opcodes::OP_LW, parse_itype as ParserFunc));
        m.insert("ld", (opcodes::OP_LD, parse_itype as ParserFunc));
        m.insert("lbu", (opcodes::OP_LBU, parse_itype as ParserFunc));
        m.insert("lhu", (opcodes::OP_LHU, parse_itype as ParserFunc));
        m.insert("lwu", (opcodes::OP_LWU, parse_itype as ParserFunc));
        m.insert("addi", (opcodes::OP_ADDI, parse_itype as ParserFunc));
        m.insert("slti", (opcodes::OP_SLTI, parse_itype as ParserFunc));
        m.insert("sltiu", (opcodes::OP_SLTIU, parse_itype as ParserFunc));
        m.insert("xori", (opcodes::OP_XORI, parse_itype as ParserFunc));
        m.insert("ori", (opcodes::OP_ORI, parse_itype as ParserFunc));
        m.insert("andi", (opcodes::OP_ANDI, parse_itype as ParserFunc));
        m.insert("slli", (opcodes::OP_SLLI, parse_itype as ParserFunc));
        m.insert("srli", (opcodes::OP_SRLI, parse_itype as ParserFunc));
        m.insert("srai", (opcodes::OP_SRAI, parse_itype as ParserFunc));
        m.insert("addiw", (opcodes::OP_ADDIW, parse_itype as ParserFunc));
        m.insert("slliw", (opcodes::OP_SLLIW, parse_itype as ParserFunc));
        m.insert("srliw", (opcodes::OP_SRLIW, parse_itype as ParserFunc));
        m.insert("sraiw", (opcodes::OP_SRAIW, parse_itype as ParserFunc));
        m.insert("jalr", (opcodes::OP_JALR, parse_itype as ParserFunc));
        // R type
        m.insert("add", (opcodes::OP_ADD, parse_rtype as ParserFunc));
        m.insert("sub", (opcodes::OP_SUB, parse_rtype as ParserFunc));
        m.insert("sll", (opcodes::OP_SLL, parse_rtype as ParserFunc));
        m.insert("slt", (opcodes::OP_SLT, parse_rtype as ParserFunc));
        m.insert("sltu", (opcodes::OP_SLTU, parse_rtype as ParserFunc));
        m.insert("xor", (opcodes::OP_XOR, parse_rtype as ParserFunc));
        m.insert("srl", (opcodes::OP_SRL, parse_rtype as ParserFunc));
        m.insert("sra", (opcodes::OP_SRA, parse_rtype as ParserFunc));
        m.insert("or", (opcodes::OP_OR, parse_rtype as ParserFunc));
        m.insert("and", (opcodes::OP_AND, parse_rtype as ParserFunc));
        m.insert("addw", (opcodes::OP_ADDW, parse_rtype as ParserFunc));
        m.insert("subw", (opcodes::OP_SUBW, parse_rtype as ParserFunc));
        m.insert("sllw", (opcodes::OP_SLLW, parse_rtype as ParserFunc));
        m.insert("srlw", (opcodes::OP_SRLW, parse_rtype as ParserFunc));
        m.insert("sraw", (opcodes::OP_SRAW, parse_rtype as ParserFunc));
        m.insert("mul", (opcodes::OP_MUL, parse_rtype as ParserFunc));
        m.insert("mulw", (opcodes::OP_MULW, parse_rtype as ParserFunc));
        m.insert("mulh", (opcodes::OP_MULH, parse_rtype as ParserFunc));
        m.insert("mulhsu", (opcodes::OP_MULHSU, parse_rtype as ParserFunc));
        m.insert("mulhu", (opcodes::OP_MULHU, parse_rtype as ParserFunc));
        m.insert("div", (opcodes::OP_DIV, parse_rtype as ParserFunc));
        m.insert("divw", (opcodes::OP_DIVW, parse_rtype as ParserFunc));
        m.insert("divu", (opcodes::OP_DIVU, parse_rtype as ParserFunc));
        m.insert("divuw", (opcodes::OP_DIVUW, parse_rtype as ParserFunc));
        m.insert("rem", (opcodes::OP_REM, parse_rtype as ParserFunc));
        m.insert("remw", (opcodes::OP_REMW, parse_rtype as ParserFunc));
        m.insert("remu", (opcodes::OP_REMU, parse_rtype as ParserFunc));
        m.insert("remuw", (opcodes::OP_REMUW, parse_rtype as ParserFunc));
        // S type
        m.insert("beq", (opcodes::OP_BEQ, parse_stype as ParserFunc));
        m.insert("bne", (opcodes::OP_BNE, parse_stype as ParserFunc));
        m.insert("blt", (opcodes::OP_BLT, parse_stype as ParserFunc));
        m.insert("bge", (opcodes::OP_BGE, parse_stype as ParserFunc));
        m.insert("bltu", (opcodes::OP_BLTU, parse_stype as ParserFunc));
        m.insert("bgeu", (opcodes::OP_BGEU, parse_stype as ParserFunc));
        m.insert("sb", (opcodes::OP_SB, parse_stype as ParserFunc));
        m.insert("sh", (opcodes::OP_SH, parse_stype as ParserFunc));
        m.insert("sw", (opcodes::OP_SW, parse_stype as ParserFunc));
        m.insert("sd", (opcodes::OP_SD, parse_stype as ParserFunc));
        // U type
        m.insert("lui", (opcodes::OP_LUI, parse_utype as ParserFunc));
        m.insert("auipc", (opcodes::OP_AUIPC, parse_utype as ParserFunc));
        m.insert("jal", (opcodes::OP_JAL, parse_utype as ParserFunc));
        // noarg type
        m.insert("ecall", (opcodes::OP_ECALL, parse_noarg_rtype as ParserFunc));
        m.insert("ebreak", (opcodes::OP_EBREAK, parse_noarg_rtype as ParserFunc));
        m.insert("fence", (opcodes::OP_FENCE, parse_noarg_rtype as ParserFunc));
        m.insert("fencei", (opcodes::OP_FENCEI, parse_noarg_rtype as ParserFunc));
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
    let rd = stream.next_register()?;
    let rs1 = stream.next_register()?;
    let rs2 = stream.next_register()?;
    Ok(Rtype::new(opcode, rd, rs1, rs2).into())
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
    let immediate = stream.next_number()?;
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
                offset as u32 >> 1,
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
