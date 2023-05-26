use super::emitter;
use ckb_vm::{machine::InstructionCycleFunc, Bytes, Error};
use inkwell::context::Context;

pub struct LlvmCompilingMachine {
    code: emitter::Code,
}

impl LlvmCompilingMachine {
    pub fn initialize() -> Result<(), Error> {
        emitter::initialize()
    }

    pub fn load(
        output_path: &str,
        code: &Bytes,
        symbol_prefix: &str,
        instruction_cycle_func: &InstructionCycleFunc,
        generate_debug_info: bool,
        check_memory_bounds: bool,
    ) -> Result<Self, Error> {
        let code = emitter::load(
            output_path,
            code,
            symbol_prefix,
            instruction_cycle_func,
            generate_debug_info,
            check_memory_bounds,
        )?;

        Ok(Self { code })
    }

    pub fn bitcode(self, optimize: bool) -> Result<Bytes, Error> {
        let context = Context::create();
        let (mut emit_data, mut debug_data) = emitter::build_llvm_data(&context, self.code)?;

        emitter::bitcode(&context, &mut emit_data, &mut debug_data, optimize)
    }

    pub fn aot(self, optimize: bool) -> Result<Bytes, Error> {
        let context = Context::create();
        let (mut emit_data, mut debug_data) = emitter::build_llvm_data(&context, self.code)?;

        emitter::aot(&context, &mut emit_data, &mut debug_data, optimize)
    }
}
