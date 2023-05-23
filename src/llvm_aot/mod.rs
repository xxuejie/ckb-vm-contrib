pub mod ast;
mod compiler;
mod emitter;
mod preprocessor;
mod runner;
mod symbols;
mod utils;

pub use compiler::LlvmCompilingMachine;
pub use preprocessor::{preprocess, BasicBlock, Func};
pub use runner::{LlvmAotCoreMachine, LlvmAotCoreMachineData, LlvmAotMachine};
pub use symbols::{
    derive_aot_symbols_from_static_globals, AddressTableEntry, AotSymbols, DlSymbols,
    EntryFunctionType,
};

use ckb_vm::{machine::VERSION1, ISA_A, ISA_B, ISA_IMC, ISA_MOP};

pub const AOT_VERSION: u32 = VERSION1;
pub const AOT_ISA: u8 = ISA_A | ISA_B | ISA_IMC | ISA_MOP;

// So derive_aot_symbols_from_static_globals macro can work
pub use paste::paste;
