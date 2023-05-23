use super::{AOT_ISA, AOT_VERSION};
use ckb_vm::{
    ckb_vm_definitions::registers::REGISTER_ABI_NAMES,
    instructions::ast::{ActionOp2, Value},
    registers::RA,
    Bytes, CoreMachine, Error, Machine, Memory, Register,
};
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Write {
    Hint {
        reg: usize,
        offset: u64,
        size: u64,
        write: bool,
    },
    Lr {
        value: Value,
    },
    Memory {
        address: Value,
        size: u8,
        value: Value,
    },
    Register {
        index: usize,
        value: Value,
    },
}

impl Write {
    pub fn is_simple_register_write(&self, index: usize, value: u64) -> bool {
        match self {
            Write::Register {
                index: i,
                value: Value::Imm(v),
            } if *i == index && *v == value => true,
            _ => false,
        }
    }
}

impl fmt::Display for Write {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Write::Hint {
                reg,
                offset,
                size,
                write,
            } => {
                let prefix = if *write { "Write" } else { "Read" };
                write!(
                    f,
                    "{}Hint: Reg({} + 0x{:x}), size: 0x{:x}",
                    prefix,
                    register_names(*reg),
                    offset,
                    size
                )
            }
            Write::Lr { value } => write!(f, "Lr = {}", PrettyValue::new(value),),
            Write::Memory {
                address,
                size,
                value,
            } => write!(
                f,
                "Memory[ {} ]@{} = {}",
                PrettyValue::new(address),
                size,
                PrettyValue::new(value),
            ),
            Write::Register { index, value } => write!(
                f,
                "Register[ {} ] = {}",
                register_names(*index),
                PrettyValue::new(value),
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Control {
    Jump { pc: Value, writes: Vec<Write> },
    Call { address: u64, writes: Vec<Write> },
    IndirectCall { pc: Value, writes: Vec<Write> },
    Tailcall { address: u64 },
    Return { pc: Value, writes: Vec<Write> },
    Ecall { pc: Value },
    Ebreak { pc: Value },
}

impl Control {
    pub fn pc(&self) -> Value {
        match self {
            Control::Jump { pc, .. } => pc.clone(),
            Control::Call { address, .. } => Value::Imm(*address),
            Control::IndirectCall { pc, .. } => pc.clone(),
            Control::Tailcall { address } => Value::Imm(*address),
            Control::Return { pc, .. } => pc.clone(),
            Control::Ecall { pc } => pc.clone(),
            Control::Ebreak { pc } => pc.clone(),
        }
    }

    pub fn writes(&self) -> Option<&[Write]> {
        match self {
            Control::Jump { writes, .. } => Some(writes),
            Control::Call { writes, .. } => Some(writes),
            Control::IndirectCall { writes, .. } => Some(writes),
            Control::Return { writes, .. } => Some(writes),
            Control::Tailcall { .. } => None,
            Control::Ecall { .. } => None,
            Control::Ebreak { .. } => None,
        }
    }

    // For calls, there should be a Write setting RA to a specific value,
    // this method checks Control structure and return such a value if exists.
    pub fn call_resume_address(&self) -> Option<u64> {
        let writes = match self {
            Control::Call { writes, .. } => Some(writes),
            Control::IndirectCall { writes, .. } => Some(writes),
            _ => None,
        };
        writes.and_then(|writes| {
            writes.iter().find_map(|write| {
                if let Write::Register { index, value } = write {
                    if *index == RA {
                        if let Value::Imm(imm) = value {
                            return Some(*imm);
                        }
                    }
                }
                None
            })
        })
    }
}

impl fmt::Display for Control {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = if self
            .writes()
            .map(|writes| writes.len() > 0)
            .unwrap_or(false)
        {
            " (with writes)"
        } else {
            ""
        };
        match self {
            Control::Jump { pc, .. } => write!(f, "Jump to {}{}", PrettyValue::new(pc), suffix),
            Control::Call { address, .. } => write!(f, "Call to 0x{:x}{}", address, suffix),
            Control::IndirectCall { pc, .. } => {
                write!(f, "Indirect call to {}{}", PrettyValue::new(pc), suffix)
            }
            Control::Tailcall { address, .. } => write!(f, "Tailcall to 0x{:x}{}", address, suffix),
            Control::Return { .. } => write!(f, "Return{}", suffix),
            Control::Ecall { .. } => write!(f, "Ecall{}", suffix),
            Control::Ebreak { .. } => write!(f, "Ebreak{}", suffix),
        }
    }
}

pub struct AstMachine {
    registers: [Value; 32],
    pc: Value,
    next_pc: Value,

    control: Option<Control>,
    memory_writes: Vec<Write>,
    register_writes: HashMap<usize, Value>,
}

impl Default for AstMachine {
    fn default() -> Self {
        Self {
            registers: init_registers(),
            pc: Value::Imm(0),
            next_pc: Value::from_u64(0),
            control: None,
            memory_writes: vec![],
            register_writes: HashMap::new(),
        }
    }
}

impl AstMachine {
    pub fn take_writes(&mut self) -> Vec<Write> {
        let mut writes = mem::replace(&mut self.memory_writes, vec![]);
        let register_writes: Vec<Write> = mem::replace(&mut self.register_writes, HashMap::new())
            .into_iter()
            .map(|(index, value)| Write::Register { index, value })
            .collect();
        writes.extend(register_writes);
        writes
    }

    pub fn take_control(&mut self) -> Option<Control> {
        mem::replace(&mut self.control, None)
    }

    pub fn reset_registers(&mut self) {
        self.registers = init_registers();
    }
}

impl CoreMachine for AstMachine {
    type REG = Value;
    type MEM = Self;

    fn pc(&self) -> &Value {
        &self.pc
    }

    fn update_pc(&mut self, pc: Self::REG) {
        self.next_pc = pc;
    }

    fn commit_pc(&mut self) {
        self.pc = simplify(&self.next_pc);
    }

    fn memory(&self) -> &Self {
        &self
    }

    fn memory_mut(&mut self) -> &mut Self {
        self
    }

    fn registers(&self) -> &[Value] {
        &self.registers
    }

    fn set_register(&mut self, index: usize, value: Value) {
        let simplified_value = simplify(&value);
        self.registers[index] = simplified_value.clone();
        self.register_writes.insert(index, simplified_value);
    }

    fn version(&self) -> u32 {
        AOT_VERSION
    }

    fn isa(&self) -> u8 {
        AOT_ISA
    }
}

impl Machine for AstMachine {
    fn ecall(&mut self) -> Result<(), Error> {
        self.control = Some(Control::Ecall {
            pc: self.next_pc.clone(),
        });
        Ok(())
    }

    fn ebreak(&mut self) -> Result<(), Error> {
        self.control = Some(Control::Ebreak {
            pc: self.next_pc.clone(),
        });
        Ok(())
    }
}

impl Memory for AstMachine {
    type REG = Value;

    fn new() -> Self {
        unreachable!()
    }

    fn new_with_memory(_memory_size: usize) -> Self {
        unreachable!()
    }

    fn init_pages(
        &mut self,
        _addr: u64,
        _size: u64,
        _flags: u8,
        _source: Option<Bytes>,
        _offset_from_addr: u64,
    ) -> Result<(), Error> {
        unreachable!()
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

    fn memory_size(&self) -> usize {
        unreachable!()
    }

    fn store_byte(&mut self, _addr: u64, _size: u64, _value: u8) -> Result<(), Error> {
        unreachable!()
    }

    fn store_bytes(&mut self, _addr: u64, _value: &[u8]) -> Result<(), Error> {
        unreachable!()
    }

    fn load_bytes(&mut self, _addr: u64, _size: u64) -> Result<Bytes, Error> {
        unreachable!()
    }

    fn execute_load16(&mut self, _addr: u64) -> Result<u16, Error> {
        unreachable!()
    }

    fn execute_load32(&mut self, _addr: u64) -> Result<u32, Error> {
        unreachable!()
    }

    fn load8(&mut self, addr: &Value) -> Result<Value, Error> {
        Ok(Value::Load(Rc::new(addr.clone()), 1))
    }

    fn load16(&mut self, addr: &Value) -> Result<Value, Error> {
        Ok(Value::Load(Rc::new(addr.clone()), 2))
    }

    fn load32(&mut self, addr: &Value) -> Result<Value, Error> {
        Ok(Value::Load(Rc::new(addr.clone()), 4))
    }

    fn load64(&mut self, addr: &Value) -> Result<Value, Error> {
        Ok(Value::Load(Rc::new(addr.clone()), 8))
    }

    fn store8(&mut self, addr: &Value, value: &Value) -> Result<(), Error> {
        self.memory_writes.push(Write::Memory {
            address: addr.clone(),
            size: 1,
            value: value.clone(),
        });
        Ok(())
    }

    fn store16(&mut self, addr: &Value, value: &Value) -> Result<(), Error> {
        self.memory_writes.push(Write::Memory {
            address: addr.clone(),
            size: 2,
            value: value.clone(),
        });
        Ok(())
    }

    fn store32(&mut self, addr: &Value, value: &Value) -> Result<(), Error> {
        self.memory_writes.push(Write::Memory {
            address: addr.clone(),
            size: 4,
            value: value.clone(),
        });
        Ok(())
    }

    fn store64(&mut self, addr: &Value, value: &Value) -> Result<(), Error> {
        self.memory_writes.push(Write::Memory {
            address: addr.clone(),
            size: 8,
            value: value.clone(),
        });
        Ok(())
    }

    fn lr(&self) -> &Value {
        &Value::Lr
    }

    fn set_lr(&mut self, value: &Value) {
        self.memory_writes.push(Write::Lr {
            value: value.clone(),
        });
    }
}

#[derive(Clone, Debug)]
pub struct PrettyValue(pub Rc<Value>);

impl PrettyValue {
    pub fn new(value: &Value) -> Self {
        Self(Rc::new(value.clone()))
    }
}

fn infix_action_op2(op: &ActionOp2) -> Option<&'static str> {
    match op {
        ActionOp2::Add => Some("+"),
        ActionOp2::Sub => Some("-"),
        ActionOp2::Mul => Some("*"),
        ActionOp2::Bitand => Some("&"),
        ActionOp2::Bitor => Some("|"),
        ActionOp2::Bitxor => Some("^"),
        ActionOp2::Eq => Some("=="),
        ActionOp2::Shl => Some("<<"),
        _ => None,
    }
}

impl fmt::Display for PrettyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            Value::Lr => write!(f, "Lr"),
            Value::Imm(i) => write!(f, "0x{:x}", i),
            Value::Register(r) => write!(f, "Reg({})", register_names(*r)),
            Value::Op1(op, v) => write!(f, "Op1({:?}, {})", op, PrettyValue(v.clone())),
            Value::Op2(op, v1, v2) => {
                if let Some(s) = infix_action_op2(op) {
                    write!(
                        f,
                        "({} {} {})",
                        PrettyValue(v1.clone()),
                        s,
                        PrettyValue(v2.clone())
                    )
                } else {
                    write!(
                        f,
                        "Op2({:?}, {}, {})",
                        op,
                        PrettyValue(v1.clone()),
                        PrettyValue(v2.clone())
                    )
                }
            }
            Value::SignOp2(op, v1, v2, b) => write!(
                f,
                "SignOp2({:?}, {}, {}, {})",
                op,
                PrettyValue(v1.clone()),
                PrettyValue(v2.clone()),
                b
            ),
            Value::Cond(c, v1, v2) => write!(
                f,
                "Cond({}, {}, {})",
                PrettyValue(c.clone()),
                PrettyValue(v1.clone()),
                PrettyValue(v2.clone()),
            ),
            Value::Load(v, s) => write!(f, "Load({}, {})", PrettyValue(v.clone()), s,),
        }
    }
}

// Notice LLVM would perform a lot of simplification work, the 2 simplify
// functions provided here, are more to simplify control structures. One
// example is to simplify possible targets for basic block, for better
// preprocessing reasons. Other computation related simplifications will
// be deferred to LLVM's optimization engine.
pub fn simplify_with_writes(value: &Value, writes: &[&Write]) -> Value {
    match value {
        Value::Op2(ActionOp2::Bitand, lhs, rhs) => {
            if let (Value::Imm(lhs), Value::Imm(rhs)) = (
                &simplify_with_writes(lhs, writes),
                &simplify_with_writes(rhs, writes),
            ) {
                return Value::Imm((*lhs) & (*rhs));
            }
        }
        Value::Register(r) => {
            if let Some(imm) = writes.iter().rev().find_map(|write| {
                match write {
                    Write::Register { index: r2, value } if *r == *r2 => {
                        if let Value::Imm(imm) = &value {
                            return Some(*imm);
                        }
                    }
                    _ => (),
                };
                None
            }) {
                return Value::Imm(imm);
            }
        }
        _ => (),
    };
    value.clone()
}

pub fn simplify(value: &Value) -> Value {
    match value {
        Value::Op2(ActionOp2::Add, a, b) => {
            match (&simplify(a), &simplify(b)) {
                (Value::Op2(ActionOp2::Add, r, c), Value::Imm(j)) => {
                    if let Value::Imm(i) = &**c {
                        return Value::Op2(ActionOp2::Add, r.clone(), Rc::new(Value::Imm(i + j)));
                    }
                }
                (Value::Imm(j), Value::Op2(ActionOp2::Add, r, c)) => {
                    if let Value::Imm(i) = &**c {
                        return Value::Op2(ActionOp2::Add, r.clone(), Rc::new(Value::Imm(i + j)));
                    }
                }
                (Value::Imm(0), r) => return r.clone(),
                (r, Value::Imm(0)) => return r.clone(),
                _ => (),
            };
            value.clone()
        }
        Value::Op1(op, a) => Value::Op1(*op, Rc::new(simplify(a))),
        Value::Op2(op, a, b) => Value::Op2(*op, Rc::new(simplify(a)), Rc::new(simplify(b))),
        Value::SignOp2(op, a, b, c) => {
            Value::SignOp2(*op, Rc::new(simplify(a)), Rc::new(simplify(b)), *c)
        }
        Value::Cond(c, t, f) => Value::Cond(
            Rc::new(simplify(c)),
            Rc::new(simplify(t)),
            Rc::new(simplify(f)),
        ),
        Value::Load(r, s) => Value::Load(Rc::new(simplify(r)), *s),
        _ => value.clone(),
    }
}

pub fn init_registers() -> [Value; 32] {
    [
        Value::Imm(0),
        Value::Register(1),
        Value::Register(2),
        Value::Register(3),
        Value::Register(4),
        Value::Register(5),
        Value::Register(6),
        Value::Register(7),
        Value::Register(8),
        Value::Register(9),
        Value::Register(10),
        Value::Register(11),
        Value::Register(12),
        Value::Register(13),
        Value::Register(14),
        Value::Register(15),
        Value::Register(16),
        Value::Register(17),
        Value::Register(18),
        Value::Register(19),
        Value::Register(20),
        Value::Register(21),
        Value::Register(22),
        Value::Register(23),
        Value::Register(24),
        Value::Register(25),
        Value::Register(26),
        Value::Register(27),
        Value::Register(28),
        Value::Register(29),
        Value::Register(30),
        Value::Register(31),
    ]
}

pub fn register_names(index: usize) -> &'static str {
    if index < REGISTER_ABI_NAMES.len() {
        REGISTER_ABI_NAMES[index]
    } else {
        panic!("Invalid register index: {}!", index)
    }
}
