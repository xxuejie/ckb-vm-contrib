use ckb_vm::{
    instructions::ast::{ActionOp1, ActionOp2, SignActionOp2, Value},
    Error, Machine, Memory, Register, RISCV_GENERAL_REGISTER_NUMBER,
};

pub const PC_INDEX: usize = 0xFFFF;

/// An interpreter function for CKB-VM's AST value, which might be super
/// helpful in terms of debugging.
pub fn interpret<Mac: Machine>(value: &Value, machine: &mut Mac) -> Result<Mac::REG, Error> {
    match value {
        Value::Lr => Ok(machine.memory_mut().lr().clone()),
        Value::Imm(imm) => Ok(Mac::REG::from_u64(*imm)),
        Value::Register(index) => {
            if *index < RISCV_GENERAL_REGISTER_NUMBER {
                Ok(machine.registers()[*index].clone())
            } else if *index == PC_INDEX {
                Ok(machine.pc().clone())
            } else {
                Err(Error::External(format!(
                    "Invalid register index: {}",
                    index
                )))
            }
        }
        Value::Op1(op, v) => interpret_op1::<Mac>(op, interpret(v, machine)?),
        Value::Op2(op, lhs, rhs) => {
            interpret_op2::<Mac>(op, interpret(lhs, machine)?, interpret(rhs, machine)?)
        }
        Value::SignOp2(op, lhs, rhs, signed) => interpret_sign_op2::<Mac>(
            op,
            interpret(lhs, machine)?,
            interpret(rhs, machine)?,
            *signed,
        ),
        Value::Cond(c, t, f) => {
            let c = interpret(c, machine)?;
            if c.to_u64() == Mac::REG::one().to_u64() {
                interpret(t, machine)
            } else {
                interpret(f, machine)
            }
        }
        Value::Load(addr, size) => {
            let addr = interpret(addr, machine)?;
            match size {
                1 => machine.memory_mut().load8(&addr),
                2 => machine.memory_mut().load16(&addr),
                4 => machine.memory_mut().load32(&addr),
                8 => machine.memory_mut().load64(&addr),
                _ => Err(Error::External(format!("Invalid load size: {}", size))),
            }
        }
        Value::External(value, _) => interpret(value, machine),
    }
}

fn interpret_op1<Mac: Machine>(op: &ActionOp1, value: Mac::REG) -> Result<Mac::REG, Error> {
    match op {
        ActionOp1::Not => Ok(!value),
        ActionOp1::LogicalNot => Ok(value.logical_not()),
        ActionOp1::Clz => Ok(value.clz()),
        ActionOp1::Ctz => Ok(value.ctz()),
        ActionOp1::Cpop => Ok(value.cpop()),
        ActionOp1::Orcb => Ok(value.orcb()),
        ActionOp1::Rev8 => Ok(value.rev8()),
    }
}

fn interpret_op2<Mac: Machine>(
    op: &ActionOp2,
    lhs: Mac::REG,
    rhs: Mac::REG,
) -> Result<Mac::REG, Error> {
    match op {
        ActionOp2::Add => Ok(lhs.overflowing_add(&rhs)),
        ActionOp2::Sub => Ok(lhs.overflowing_sub(&rhs)),
        ActionOp2::Mul => Ok(lhs.overflowing_mul(&rhs)),
        ActionOp2::Mulhsu => Ok(lhs.overflowing_mul_high_signed_unsigned(&rhs)),
        ActionOp2::Bitand => Ok(lhs & rhs),
        ActionOp2::Bitor => Ok(lhs | rhs),
        ActionOp2::Bitxor => Ok(lhs ^ rhs),
        ActionOp2::Shl => Ok(lhs << rhs),
        ActionOp2::Eq => Ok(lhs.eq(&rhs)),
        ActionOp2::Clmul => Ok(lhs.clmul(&rhs)),
        ActionOp2::Clmulh => Ok(lhs.clmulh(&rhs)),
        ActionOp2::Clmulr => Ok(lhs.clmulr(&rhs)),
        ActionOp2::Rol => Ok(lhs.rol(&rhs)),
        ActionOp2::Ror => Ok(lhs.ror(&rhs)),
    }
}

fn interpret_sign_op2<Mac: Machine>(
    op: &SignActionOp2,
    lhs: Mac::REG,
    rhs: Mac::REG,
    signed: bool,
) -> Result<Mac::REG, Error> {
    match op {
        SignActionOp2::Mulh => {
            if signed {
                Ok(lhs.overflowing_mul_high_signed(&rhs))
            } else {
                Ok(lhs.overflowing_mul_high_unsigned(&rhs))
            }
        }
        SignActionOp2::Div => {
            if signed {
                Ok(lhs.overflowing_div_signed(&rhs))
            } else {
                Ok(lhs.overflowing_div(&rhs))
            }
        }
        SignActionOp2::Rem => {
            if signed {
                Ok(lhs.overflowing_rem_signed(&rhs))
            } else {
                Ok(lhs.overflowing_rem(&rhs))
            }
        }
        SignActionOp2::Shr => {
            if signed {
                Ok(lhs.signed_shr(&rhs))
            } else {
                Ok(lhs >> rhs)
            }
        }
        SignActionOp2::Lt => {
            if signed {
                Ok(lhs.lt_s(&rhs))
            } else {
                Ok(lhs.lt(&rhs))
            }
        }
        SignActionOp2::Extend => {
            if signed {
                Ok(lhs.sign_extend(&rhs))
            } else {
                Ok(lhs.zero_extend(&rhs))
            }
        }
    }
}
