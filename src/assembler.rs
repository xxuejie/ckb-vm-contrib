use ckb_vm::{
    instructions::{tagged::TaggedInstruction, Itype, RegisterIndex, Rtype, Stype, Utype},
    Error,
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
pub fn parse(input: &str) -> Result<Vec<TaggedInstruction>, Error> {
    let mut stream = InstrStream::create(input);
    let mut insts = Vec::new();

    while stream.has_token() {
        let opcode_name = stream.next_token()?;
        if let Some((opcode, f)) = PARSER_FUNCS.get(opcode_name) {
            insts.push(f(*opcode, &mut stream)?);
        } else {
            return Err(Error::External(format!(
                "Invalid instruction {}!",
                opcode_name
            )));
        }
    }

    Ok(insts)
}

/// Assemble ckb-vm instructions into binary bytes
pub fn assemble(_insts: &[TaggedInstruction]) -> Result<Vec<u8>, Error> {
    // Note: like decoders, RISC-V extension separation will be introduced here.
    unimplemented!()
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
        m.insert("subw", (opcodes::OP_ADDW, parse_rtype as ParserFunc));
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
    let rd = stream.next_register()?;
    let rs1 = stream.next_register()?;
    let immediate = stream.next_number()?;
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

fn parse_utype(
    opcode: InstructionOpcode,
    stream: &mut InstrStream,
) -> Result<TaggedInstruction, Error> {
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

struct InstrStream<'a> {
    current: usize,
    tokens: Vec<&'a str>,
}

impl<'a> InstrStream<'a> {
    fn create(input: &'a str) -> Self {
        let tokens = input
            // First, ensure multiple lines are processed correctly
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
            .map(|l| {
                TOKENS_RE
                    .captures_iter(l)
                    .map(|c| c.get(0))
                    .filter(|c| c.is_some())
                    .map(|c| c.unwrap().as_str())
            })
            .flatten()
            .collect();

        Self { tokens, current: 0 }
    }

    fn has_token(&self) -> bool {
        self.current < self.tokens.len()
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
}
