use super::{
    ast::{register_names, Control, Write},
    preprocessor::{preprocess, Func},
    runner::{
        LlvmAotCoreMachineData, LlvmAotMachineEnv, BARE_FUNC_ERROR_OR_TERMINATED,
        BARE_FUNC_MISSING, BARE_FUNC_RETURN, EXIT_REASON_BARE_CALL_EXIT,
        EXIT_REASON_CYCLES_OVERFLOW, EXIT_REASON_MAX_CYCLES_EXCEEDED,
        EXIT_REASON_REVERT_TO_INTERPRETER,
    },
};
use ckb_vm::{
    ckb_vm_definitions::registers,
    instructions::ast::{ActionOp1, ActionOp2, SignActionOp2, Value},
    machine::InstructionCycleFunc,
    Bytes, Error, Register,
};
use either::Either;
use inkwell::{
    attributes::{Attribute, AttributeLoc},
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    debug_info::{
        AsDIScope, DICompileUnit, DIFile, DILocation, DIScope, DWARFEmissionKind,
        DWARFSourceLanguage, DebugInfoBuilder,
    },
    intrinsics::Intrinsic,
    memory_buffer::MemoryBuffer,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{FunctionType, IntType, PointerType},
    values::{
        AnyValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue,
        PointerValue, StructValue,
    },
    AddressSpace, IntPredicate, OptimizationLevel,
};
use lazy_static::lazy_static;
use log::debug;
use memoffset::offset_of;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write as StdWrite;
use std::path::Path;

const HHVM_CALL_CONV: u32 = llvm_sys::LLVMCallConv::LLVMHHVMCallConv as u32;

pub fn initialize() -> Result<(), Error> {
    Target::initialize_all(&InitializationConfig {
        base: true,
        info: true,
        asm_printer: true,
        asm_parser: true,
        ..Default::default()
    });
    assert_eq!(REGISTER_MAPPINGS.len(), 15);
    assert_eq!(REGISTER_MAPPINGS[1], Mapping::Pointer);
    assert!(REGISTER_MAPPINGS.iter().any(|m| match m {
        Mapping::Pc => true,
        _ => false,
    }));
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Code {
    funcs: Vec<Func>,
    code_hash: [u8; 32],
    symbol_prefix: String,
    name: String,
    directory: String,
    generate_debug_info: bool,
}

pub struct EmitData<'a> {
    code: Code,

    builder: Builder<'a>,
    module: Module<'a>,
    pass: PassManager<Module<'a>>,

    // RISC-V function address -> generated LLVM function reference
    emitted_funcs: HashMap<u64, FunctionValue<'a>>,
    // Entry function to AOT code
    entry_func: FunctionValue<'a>,
    // RISC-V early exit function
    exit_func: FunctionValue<'a>,
    // Wrapper to call into Rust FFI functions
    ffi_wrapper_func: FunctionValue<'a>,
}

impl<'a> EmitData<'a> {
    fn hosted_setjmp_name(&self) -> String {
        format!("{}____hosted_setjmp____", self.code.symbol_prefix)
    }

    fn hosted_longjmp_name(&self) -> String {
        format!("{}____hosted_longjmp____", self.code.symbol_prefix)
    }

    /// As of LLVM 14, the intrinsics llvm.eh.sjlj.setjmp/llvm.eh.sjlj.longjmp have
    /// bugs that rsp is not properly saved. Hence we are providing our own setjmp/longjmp
    /// pairs for now. Considering the design of the AOT module here, setjmp/longjmp is
    /// simpler and most likely faster than the alternative stack unwinding setup.
    fn module_inline_asm(&self) -> String {
        format!(
            r#"
.text
{}:
    mov %rbp, (%rdi)
    mov (%rsp), %rcx   # (%rsp) contains the return address
    mov %rcx, 8(%rdi)
    mov %rbx, 24(%rdi)
    mov %r12, (%rsp)   # short for `add $8, %rsp; push %r12`
    push %r13
    push %r14
    push %r15
    mov %rsp, 16(%rdi)
    xor %eax, %eax
    jmp *%rcx
{}:
    mov (%rdi), %rbp
    mov 16(%rdi), %rsp
    mov 24(%rdi), %rbx
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    mov $1, %eax
    jmp *8(%rdi)
        "#,
            self.hosted_setjmp_name(),
            self.hosted_longjmp_name()
        )
        .trim()
        .to_string()
    }

    fn basic_block_name(&self, addr: u64) -> String {
        format!("basic_block_0x{:x}", addr)
    }
}

pub struct DebugData<'a> {
    di_builder: DebugInfoBuilder<'a>,
    compile_file: DIFile<'a>,
    current_scope: DIScope<'a>,
    file: File,
    line: u32,
}

impl<'a> DebugData<'a> {
    fn new(
        di_builder: DebugInfoBuilder<'a>,
        compile_file: DIFile<'a>,
        compile_unit: DICompileUnit<'a>,
        filename: &str,
    ) -> Result<Self, Error> {
        let file = File::create(filename)?;
        Ok(Self {
            di_builder,
            compile_file,
            current_scope: compile_unit.as_debug_info_scope(),
            file,
            line: 0,
        })
    }

    fn write(&mut self, content: &str) -> Result<(), Error> {
        write!(self.file, "{}\n", content)?;
        self.line += 1;
        Ok(())
    }

    fn debug_location(&self, context: &'a Context) -> DILocation<'a> {
        self.di_builder
            .create_debug_location(context, self.line, 0, self.current_scope, None)
    }

    fn set_scope(&mut self, scope: DIScope<'a>) {
        self.current_scope = scope;
    }
}

pub struct EmittingFunc<'a> {
    basic_blocks: HashMap<u64, BasicBlock<'a>>,
    value: FunctionValue<'a>,
    machine: IntValue<'a>,
    allocas: RegAllocas<'a>,
    pc_alloca: PointerValue<'a>,
    memory_start: IntValue<'a>,
    indirect_dispatcher_alloca: PointerValue<'a>,
    indirect_dispatcher: Option<BasicBlock<'a>>,
    ret_block: Option<BasicBlock<'a>>,
}

impl<'a> EmittingFunc<'a> {
    pub fn emit_arbitrary_jump(
        &mut self,
        context: &'a Context,
        emit_data: &EmitData<'a>,
        next_pc: IntValue<'a>,
    ) -> Result<(), Error> {
        let block = self.fetch_indirect_dispatcher(context, emit_data)?;
        emit_data
            .builder
            .build_store(self.indirect_dispatcher_alloca, next_pc);
        emit_data.builder.build_unconditional_branch(block);
        Ok(())
    }

    // In case a function contains indirect dispatches, we will emit a special
    // basic block used to locate the correct basic block after the indirect
    // dispatch.
    pub fn fetch_indirect_dispatcher(
        &mut self,
        context: &'a Context,
        emit_data: &EmitData<'a>,
    ) -> Result<BasicBlock<'a>, Error> {
        if let Some(result) = self.indirect_dispatcher {
            return Ok(result);
        }

        let i64t = context.i64_type();

        let current_block = emit_data.builder.get_insert_block().unwrap();

        let dispatch_block = context.append_basic_block(self.value, "indirect_dispatch_block");
        emit_data.builder.position_at_end(dispatch_block);

        let test_value = emit_data.builder.build_load(
            i64t,
            self.indirect_dispatcher_alloca,
            "indirect_dispatch_test_value",
        );

        let failure_block = context.append_basic_block(self.value, "indirect_jump_failure_block");

        let mut indirect_jump_targets: Vec<(u64, BasicBlock<'a>)> =
            self.basic_blocks.iter().map(|(a, b)| (*a, *b)).collect();
        indirect_jump_targets.sort_by_key(|(a, _)| *a);
        emit_select_control(
            context,
            emit_data,
            self.value,
            dispatch_block,
            test_value.into_int_value(),
            &indirect_jump_targets,
            failure_block,
        )?;

        emit_data.builder.position_at_end(failure_block);
        // When binary search fails to find a basic block to execute, we use the
        // Rust side interpreter to execute till the next basic block end, then
        // repeat the binary search process.
        // TODO: when code is really messy, we might end up in a situation that
        // the code is jumping back-and-forth between binary search implemented
        // as LLVM generated code now, and the native Rust side interpreter. An
        // alternative solution is to move binary search to Rust side. We will need
        // more studies to know where we shall put the binary search logic. For
        // now, we leave the original binary search code unchanged here.
        let (interpret_function, data) = emit_env_ffi_function(
            context,
            emit_data,
            self.machine,
            offset_of!(LlvmAotMachineEnv, interpret),
        )?;
        let mut interpret_args = [
            interpret_function.into(),
            data.into(),
            i64t.const_int(0, false).into(),
        ];
        let interpret_result = emit_ffi_call(
            context,
            emit_data,
            self,
            &mut interpret_args,
            true,
            Some("interpret_function_result"),
        )?;

        let ret_block =
            context.append_basic_block(self.value, "indirect_dispatch_interpret_ret_block");
        let resume_block =
            context.append_basic_block(self.value, "indirect_dispatch_interpret_resume_block");
        let cmp = emit_data.builder.build_int_compare(
            IntPredicate::EQ,
            interpret_result,
            i64t.const_int(BARE_FUNC_RETURN, false).into(),
            "indirect_dispatch_interpret_result_is_return",
        );
        emit_data
            .builder
            .build_conditional_branch(cmp, ret_block, resume_block);

        emit_data.builder.position_at_end(ret_block);
        emit_data
            .builder
            .build_unconditional_branch(self.fetch_ret_block(context, emit_data)?);

        emit_data.builder.position_at_end(resume_block);
        emit_data
            .builder
            .build_store(self.indirect_dispatcher_alloca, interpret_result);
        emit_data.builder.build_unconditional_branch(dispatch_block);

        emit_data.builder.position_at_end(current_block);

        self.indirect_dispatcher = Some(dispatch_block);

        Ok(dispatch_block)
    }

    pub fn fetch_ret_block(
        &mut self,
        context: &'a Context,
        emit_data: &EmitData<'a>,
    ) -> Result<BasicBlock<'a>, Error> {
        if let Some(block) = self.ret_block {
            return Ok(block);
        }

        let current_block = emit_data.builder.get_insert_block().unwrap();

        let ret_block = context.append_basic_block(self.value, "ret_block");
        emit_data.builder.position_at_end(ret_block);

        let i64t = context.i64_type();
        let next_pc = emit_data
            .builder
            .build_load(i64t, self.pc_alloca, "target_pc")
            .into_int_value();

        let last_ra_val = emit_load_from_machine(
            context,
            emit_data,
            self.machine,
            offset_of!(LlvmAotCoreMachineData, last_ra),
            i64t,
            Some("last_ra"),
        )?;
        let cmp = emit_data.builder.build_int_compare(
            IntPredicate::EQ,
            next_pc,
            last_ra_val,
            "ra_cmp_last_ra",
        );

        let normal_ret_block = context.append_basic_block(self.value, "normal_ret_block");
        let malformed_ret_block = context.append_basic_block(self.value, "malformed_ret_block");

        emit_data
            .builder
            .build_conditional_branch(cmp, normal_ret_block, malformed_ret_block);

        emit_data.builder.position_at_end(normal_ret_block);
        emit_riscv_return(emit_data, &self.allocas)?;

        emit_data.builder.position_at_end(malformed_ret_block);
        emit_call_exit(context, emit_data, self, EXIT_REASON_REVERT_TO_INTERPRETER)?;

        emit_data.builder.position_at_end(current_block);

        self.ret_block = Some(ret_block);

        return Ok(ret_block);
    }
}

pub fn load<'a>(
    output_path: &str,
    code: &Bytes,
    symbol_prefix: &str,
    instruction_cycle_func: &InstructionCycleFunc,
    generate_debug_info: bool,
) -> Result<Code, Error> {
    let name = Path::new(output_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(output_path);
    let directory = Path::new(output_path)
        .parent()
        .and_then(|s| s.to_str())
        .unwrap_or(".");

    let funcs = preprocess(code, instruction_cycle_func)?;
    let code_hash: [u8; 32] = blake3::hash(code).into();

    Ok(Code {
        funcs,
        code_hash,
        symbol_prefix: symbol_prefix.to_string(),
        name: name.to_string(),
        directory: directory.to_string(),
        generate_debug_info,
    })
}

pub fn build_llvm_data<'a>(
    context: &'a Context,
    code: Code,
) -> Result<(EmitData<'a>, Option<DebugData<'a>>), Error> {
    let module = context.create_module(&format!("{}{}", code.symbol_prefix, code.name));
    let builder = context.create_builder();

    let debug_data = if code.generate_debug_info {
        let debug_file_name = format!("{}.debug.s", code.name);
        let (di_builder, compile_unit) = module.create_debug_info_builder(
            true,
            DWARFSourceLanguage::C,
            &debug_file_name,
            &code.directory,
            "ckb-vm-llvm-aot-engine",
            false,
            "",
            0,
            "",
            DWARFEmissionKind::Full,
            0,
            true,
            false,
            "",
            "",
        );
        let compile_file = di_builder.create_file(&debug_file_name, &code.directory);

        Some(DebugData::new(
            di_builder,
            compile_file,
            compile_unit,
            Path::new(&code.directory)
                .join(debug_file_name)
                .to_str()
                .ok_or(Error::External("invalid file name!".to_string()))?,
        )?)
    } else {
        None
    };

    let pass: PassManager<Module<'a>> = PassManager::create(());
    pass.add_promote_memory_to_register_pass();
    pass.add_instruction_combining_pass();
    pass.add_reassociate_pass();
    pass.add_gvn_pass();
    pass.add_cfg_simplification_pass();
    let emitted_funcs = HashMap::with_capacity(code.funcs.len());

    let i64t = context.i64_type();
    let i8t = context.i8_type();

    let entry_func = {
        // *mut LlvmAotCoreMachineData, pointer for the host function to execute
        let entry_function_type = i8t.fn_type(&[i64t.into(), i64t.into()], false);
        module.add_function(
            &format!("{}____entry____", code.symbol_prefix),
            entry_function_type,
            None,
        )
    };
    let exit_func = {
        // Exit function is invoked from a RISC-V function, it shares the same signature
        // as RISC-V functions, except that it has no returns, since the internal longjmp
        // will trigger unwinding.
        let t = context.void_type().fn_type(&[i64t.into(); 15], false);
        let f = module.add_function(&format!("{}____exit____", code.symbol_prefix), t, None);
        f.set_call_conventions(HHVM_CALL_CONV);
        f
    };
    let ffi_wrapper_func = {
        let t = i64t.fn_type(
            &[
                ffi_function_type(context)
                    .ptr_type(AddressSpace::default())
                    .into(),
                i64t.into(),
                i64t.into(),
            ],
            false,
        );
        module.add_function(
            &format!("{}____ffi_wrapper____", code.symbol_prefix),
            t,
            None,
        )
    };

    Ok((
        EmitData {
            code,
            emitted_funcs,
            builder,
            module,
            pass,
            entry_func,
            exit_func,
            ffi_wrapper_func,
        },
        debug_data,
    ))
}

pub fn bitcode<'a>(
    context: &'a Context,
    emit_data: &mut EmitData<'a>,
    debug_data: &mut Option<DebugData<'a>>,
    optimize: bool,
) -> Result<Bytes, Error> {
    emit(context, emit_data, debug_data, optimize)?;
    Ok(llvm_buffer_to_bytes(
        &emit_data.module.write_bitcode_to_memory(),
    ))
}

pub fn aot<'a>(
    context: &'a Context,
    emit_data: &mut EmitData<'a>,
    debug_data: &mut Option<DebugData<'a>>,
    optimize: bool,
) -> Result<Bytes, Error> {
    emit(context, emit_data, debug_data, optimize)?;

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| {
        Error::External(format!(
            "LLVM error creating target from triple: {}",
            e.to_string()
        ))
    })?;
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| Error::External("Unable to create target machine!".to_string()))?;

    let buf = tm
        .write_to_memory_buffer(&emit_data.module, FileType::Object)
        .map_err(|e| Error::External(e.to_string()))?;

    Ok(llvm_buffer_to_bytes(&buf))
}

#[derive(Clone, Debug, PartialEq)]
enum Mapping {
    Pointer,
    Pc,
    Register(usize),
}

impl fmt::Display for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mapping::Pointer => write!(f, "machine"),
            Mapping::Pc => write!(f, "pc"),
            Mapping::Register(i) => write!(f, "{}", register_names(*i)),
        }
    }
}

lazy_static! {
    static ref REGISTER_MAPPINGS: Vec<Mapping> = {
        let mut m = Vec::with_capacity(15);
        m.push(Mapping::Pc);
        m.push(Mapping::Pointer);
        m.push(Mapping::Register(registers::RA));
        m.push(Mapping::Register(registers::SP));
        m.push(Mapping::Register(registers::A0));
        m.push(Mapping::Register(registers::A1));
        m.push(Mapping::Register(registers::A2));
        m.push(Mapping::Register(registers::A3));
        m.push(Mapping::Register(registers::A4));
        m.push(Mapping::Register(registers::A5));
        m.push(Mapping::Register(registers::A7));
        m.push(Mapping::Register(registers::S0));
        m.push(Mapping::Register(registers::S1));
        m.push(Mapping::Register(registers::S2));
        m.push(Mapping::Register(registers::T0));
        m
    };
    static ref REVERSE_MAPPINGS: HashMap<usize, usize> = {
        let mut m = HashMap::default();

        for i in 0..REGISTER_MAPPINGS.len() {
            if let Mapping::Register(r) = REGISTER_MAPPINGS[i] {
                m.insert(r, i);
            }
        }

        m
    };
}

/// To minimize memory writes, we transform RISC-V functions into x64 functions
/// in HHVM calling convention: https://reviews.llvm.org/D12681
/// This way each function can take 15 arguments, and also return 14 values.
/// We will take this to the extreme side:
/// * We build the data structure TransientValues to keep 15 LLVM values, including
/// LlvmAotCoreMachineData pointer, PC register, and 13 RISC-V registers. Ideally, they will
/// all be in x86 registers
/// * HHVM use different register orders in arguments vs. return values. This data
/// structure will take care of such differences, and generate values in the order to
/// minimize x86 instructions.
/// * One register, r12 can only be used as argument, not return value, we will take
/// advantage of this and keep LlvmAotCoreMachineData pointer in r12, since no RISC-V function
/// needs to modify/return this value.
/// * One unique function, __exit__, requires an extra argument for exit_aot_reason,
/// this will also be taken care of.
/// * REGISTER_MAPPINGS data structure will contain the mapping between raw indices,
/// and the actual values such as LlvmAotCoreMachineData, pc or general RISC-V registers.
#[derive(Clone, Debug, PartialEq)]
struct TransientValues<'a, T: AnyValue<'a> + Copy> {
    data: [T; 15],
    _phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T: AnyValue<'a> + Copy> TransientValues<'a, T> {
    fn from(data: [T; 15]) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    fn new<F: FnMut(&Mapping) -> Result<T, Error>>(mut f: F) -> Result<Self, Error> {
        let mut values = Vec::with_capacity(15);
        for mapping in REGISTER_MAPPINGS.iter() {
            values.push(f(mapping)?);
        }
        let values: [T; 15] = values.try_into().expect("incorrect length!");
        Ok(Self::from(values))
    }

    fn map<U: AnyValue<'a> + Copy, F: FnMut(T, &Mapping) -> Result<U, Error>>(
        &self,
        mut f: F,
    ) -> Result<TransientValues<'a, U>, Error> {
        let mut values = Vec::with_capacity(15);
        for i in 0..15 {
            values.push(f(self.data[i], &REGISTER_MAPPINGS[i])?);
        }
        let values: [U; 15] = values.try_into().expect("incorrect length!");
        Ok(TransientValues::from(values))
    }

    // Extract pc value
    fn extract_pc(&self) -> Result<T, Error> {
        let pc_index = REGISTER_MAPPINGS
            .iter()
            .position(|m| match m {
                Mapping::Pc => true,
                _ => false,
            })
            .ok_or_else(|| Error::External("PC is missing in mapping!".to_string()))?;
        Ok(self.data[pc_index])
    }

    // Extract machine value from RISC-V function arguments
    fn extract_machine(&self) -> Result<T, Error> {
        let machine_index = REGISTER_MAPPINGS
            .iter()
            .position(|m| match m {
                Mapping::Pointer => true,
                _ => false,
            })
            .ok_or_else(|| Error::External("Machine pointer is missing in mapping!".to_string()))?;
        Ok(self.data[machine_index])
    }

    fn from_arguments(values: [T; 15]) -> Self {
        Self::from(values)
    }

    fn from_return_values(values: [T; 14], machine: T) -> Self {
        Self::from_arguments(return_values_to_arguments(values, machine))
    }

    fn to_arguments(&self) -> Result<[BasicMetadataValueEnum<'a>; 15], Error> {
        let mut values = Vec::with_capacity(15);
        for i in 0..15 {
            values.push(self.data[i].as_any_value_enum().try_into().map_err(|_| {
                Error::External(
                    "Error converting AnyValueEnum to BasicMetadataValueEnum!".to_string(),
                )
            })?);
        }
        let values: [BasicMetadataValueEnum<'a>; 15] =
            values.try_into().expect("incorrect length!");
        Ok(values)
    }

    fn to_return_values(&self) -> Result<[BasicValueEnum; 14], Error> {
        let arguments = self.to_arguments()?;
        let mut values = Vec::with_capacity(15);
        for i in 0..15 {
            values.push(arguments[i].try_into().map_err(|_| {
                Error::External(
                    "Error converting BasicMetadataValueEnum to BasicValueEnum!".to_string(),
                )
            })?);
        }
        let values: [BasicValueEnum<'a>; 15] = values.try_into().expect("incorrect length!");
        Ok(arguments_to_return_values(values))
    }

    fn iter(&self) -> TransientValuesIter<'a, T> {
        TransientValuesIter {
            values: self.clone(),
            index: 0,
        }
    }
}

struct RegAllocas<'a> {
    values: TransientValues<'a, PointerValue<'a>>,
    i64t: IntType<'a>,
}

impl<'a> RegAllocas<'a> {
    fn new(values: TransientValues<'a, PointerValue<'a>>, i64t: IntType<'a>) -> Self {
        Self { values, i64t }
    }

    fn pc_alloca(&self) -> Result<PointerValue<'a>, Error> {
        self.values.extract_pc()
    }

    fn load_value(&self, emit_data: &EmitData<'a>, idx: usize) -> Result<IntValue<'a>, Error> {
        Ok(emit_data
            .builder
            .build_load(
                self.i64t,
                self.values.data[idx],
                &format!("reg_allocas_idx_{}", idx),
            )
            .into_int_value())
    }

    fn store_value<T: BasicValue<'a>>(
        &self,
        emit_data: &EmitData<'a>,
        idx: usize,
        value: T,
    ) -> Result<(), Error> {
        emit_data.builder.build_store(self.values.data[idx], value);
        Ok(())
    }

    fn load_values(
        &self,
        emit_data: &EmitData<'a>,
    ) -> Result<TransientValues<'a, IntValue<'a>>, Error> {
        self.values.map(|value, mapping| {
            Ok(emit_data
                .builder
                .build_load(self.i64t, value, &format!("reg_allocas_tmp_{}", mapping))
                .into_int_value())
        })
    }

    fn store_values<T: BasicValue<'a> + Copy>(
        &self,
        emit_data: &EmitData<'a>,
        values: &TransientValues<'a, T>,
    ) -> Result<(), Error> {
        for (i, (value, _)) in values.iter().enumerate() {
            emit_data.builder.build_store(self.values.data[i], value);
        }
        Ok(())
    }
}

fn return_values_to_arguments<T: Copy>(values: [T; 14], machine: T) -> [T; 15] {
    [
        values[0], machine, values[1], values[13], values[2], values[3], values[4], values[5],
        values[6], values[7], values[8], values[9], values[10], values[11], values[12],
    ]
}

fn arguments_to_return_values<T: Copy>(arguments: [T; 15]) -> [T; 14] {
    [
        arguments[0],
        arguments[2],
        arguments[4],
        arguments[5],
        arguments[6],
        arguments[7],
        arguments[8],
        arguments[9],
        arguments[10],
        arguments[11],
        arguments[12],
        arguments[13],
        arguments[14],
        arguments[3],
    ]
}

struct TransientValuesIter<'a, T: AnyValue<'a> + Copy> {
    values: TransientValues<'a, T>,
    index: usize,
}

impl<'a, T: AnyValue<'a> + Copy> Iterator for TransientValuesIter<'a, T> {
    type Item = (T, Mapping);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.values.data.len() {
            let result = (
                self.values.data[self.index],
                REGISTER_MAPPINGS[self.index].clone(),
            );
            self.index += 1;
            return Some(result);
        }
        None
    }
}

fn riscv_function_type<'a>(context: &'a Context) -> FunctionType<'a> {
    let i64t = context.i64_type();
    // For each generated RISC-V function, the arguments will be 15 i64 values,
    // the return type will be a struct of also 14 i64 values. See TransientValues
    // below for more details
    let return_reg_type = [i64t.into(); 14];
    let packed_reg_type = [i64t.into(); 15];
    let return_type = context.struct_type(&return_reg_type, false);
    let function_type = return_type.fn_type(&packed_reg_type, false);
    function_type
}

fn ffi_function_type<'a>(context: &'a Context) -> FunctionType<'a> {
    let i64t = context.i64_type();
    i64t.fn_type(&[i64t.into(); 2], false)
}

fn llvm_buffer_to_bytes<'a>(buf: &MemoryBuffer) -> Bytes {
    buf.as_slice().to_vec().into()
}

fn i8pt<'a>(context: &'a Context) -> PointerType<'a> {
    context.i8_type().ptr_type(AddressSpace::default())
}

fn size_to_type<'a>(context: &'a Context, size: u8) -> Result<IntType<'a>, Error> {
    Ok(match size {
        1 => context.i8_type(),
        2 => context.i16_type(),
        4 => context.i32_type(),
        8 => context.i64_type(),
        16 => context.i128_type(),
        _ => return Err(Error::External(format!("Invalid load size: {}", size))),
    })
}

fn machine_word_type<'a>(context: &'a Context) -> Result<IntType<'a>, Error> {
    size_to_type(context, Value::BITS >> 3)
}

fn double_machine_word_type<'a>(context: &'a Context) -> Result<IntType<'a>, Error> {
    size_to_type(context, Value::BITS >> 2)
}

// Loading a register value as an LLVM value in a RISC-V function
fn emit_load_reg<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    allocas: &RegAllocas<'a>,
    reg: usize,
    name: Option<&str>,
) -> Result<IntValue<'a>, Error> {
    let i64t = context.i64_type();

    if reg == 0 {
        return Ok(i64t.const_int(0, false));
    }
    if let Some(i) = REVERSE_MAPPINGS.get(&reg) {
        allocas.load_value(emit_data, *i)
    } else {
        emit_load_from_machine(
            context,
            emit_data,
            machine,
            offset_of!(LlvmAotCoreMachineData, registers) + reg * 8,
            i64t,
            name,
        )
    }
}

// Storing an LLVM value into a register in a RISC-V function
fn emit_store_reg<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    allocas: &RegAllocas<'a>,
    reg: usize,
    value: IntValue<'a>,
) -> Result<(), Error> {
    if reg == 0 {
        return Ok(());
    }
    if let Some(i) = REVERSE_MAPPINGS.get(&reg) {
        allocas.store_value(emit_data, *i, value)
    } else {
        let i64t = context.i64_type();
        emit_store_to_machine(
            context,
            emit_data,
            machine,
            value,
            offset_of!(LlvmAotCoreMachineData, registers) + reg * 8,
            i64t,
            Some(&format!("{}", register_names(reg))),
        )
    }
}

// Emit store to LlvmAotCoreMachineData operation
fn emit_store_to_machine<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    value: IntValue<'a>,
    offset: usize,
    value_type: IntType<'a>,
    name: Option<&str>,
) -> Result<(), Error> {
    let i64t = context.i64_type();
    emit_store_with_value_offset_to_machine(
        context,
        emit_data,
        machine,
        value,
        i64t.const_int(offset as u64, false),
        value_type,
        name,
    )
}

// Similar to emit_store_to_machine, but accepts offset as an LLVM value,
// which could include runtime computable values.
fn emit_store_with_value_offset_to_machine<'a>(
    _context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    value: IntValue<'a>,
    offset: IntValue<'a>,
    value_type: IntType<'a>,
    name: Option<&str>,
) -> Result<(), Error> {
    let value_pointer_type = value_type.ptr_type(AddressSpace::default());

    let name = name.unwrap_or("uNKNOWn");
    let addr_value =
        emit_data
            .builder
            .build_int_add(machine, offset, &format!("{}_addr_val", name));
    let addr = emit_data.builder.build_int_to_ptr(
        addr_value,
        value_pointer_type,
        &format!("{}_addr", name),
    );
    emit_data.builder.build_store(addr, value);
    Ok(())
}

// Emit load from LlvmAotCoreMachineData operation
fn emit_load_from_machine<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    offset: usize,
    value_type: IntType<'a>,
    name: Option<&str>,
) -> Result<IntValue<'a>, Error> {
    let i64t = context.i64_type();
    emit_load_from_struct(
        context,
        emit_data,
        machine,
        i64t.const_int(offset as u64, false),
        value_type,
        name,
    )
}

// Load values at any offset from a struct.
fn emit_load_from_struct<'a>(
    _context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    offset: IntValue<'a>,
    value_type: IntType<'a>,
    name: Option<&str>,
) -> Result<IntValue<'a>, Error> {
    let value_pointer_type = value_type.ptr_type(AddressSpace::default());

    let name = name.unwrap_or("uNKNOWn");
    let addr_value =
        emit_data
            .builder
            .build_int_add(machine, offset, &format!("{}_addr_val", name));
    let addr = emit_data.builder.build_int_to_ptr(
        addr_value,
        value_pointer_type,
        &format!("{}_addr", name),
    );
    let value = emit_data
        .builder
        .build_load(value_type, addr, &format!("{}_loaded_value", name));
    // Note that value_type is IntType, we must be able to load an IntValue
    Ok(value.into_int_value())
}

// Emit code to restore selected register values back to LlvmAotCoreMachineData
fn emit_cleanup<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    values: &TransientValues<'a, IntValue<'a>>,
) -> Result<(), Error> {
    let machine = values.extract_machine()?;
    let i64t = context.i64_type();
    for (value, mapping) in values.iter() {
        match mapping {
            Mapping::Pointer => (),
            Mapping::Pc => {
                let offset = offset_of!(LlvmAotCoreMachineData, pc);
                emit_store_to_machine(
                    context,
                    emit_data,
                    machine,
                    value,
                    offset,
                    i64t,
                    Some("pc"),
                )?;
            }
            Mapping::Register(r) => {
                let offset = offset_of!(LlvmAotCoreMachineData, registers) + r * 8;
                emit_store_to_machine(
                    context,
                    emit_data,
                    machine,
                    value,
                    offset,
                    i64t,
                    Some(&format!("{}", mapping)),
                )?;
            }
        }
    }
    Ok(())
}

// To the exact contrary of +emit_cleanup+, This function emits code to setup
// selected register values from values in LlvmAotCoreMachineData
fn emit_setup<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
) -> Result<TransientValues<'a, IntValue<'a>>, Error> {
    let i64t = context.i64_type();
    TransientValues::new(|mapping| match mapping {
        Mapping::Pointer => Ok(machine),
        Mapping::Pc => {
            let offset = offset_of!(LlvmAotCoreMachineData, pc);
            emit_load_from_machine(context, emit_data, machine, offset, i64t, Some("pc"))
        }
        Mapping::Register(r) => {
            let offset = offset_of!(LlvmAotCoreMachineData, registers) + r * 8;
            emit_load_from_machine(
                context,
                emit_data,
                machine,
                offset,
                i64t,
                Some(&format!("{}", mapping)),
            )
        }
    })
}

// Emit return statement for a RISC-V function
fn emit_riscv_return<'a>(emit_data: &EmitData<'a>, allocas: &RegAllocas<'a>) -> Result<(), Error> {
    let ret_value = allocas.load_values(emit_data)?;

    let ret_args = ret_value.to_return_values()?;
    emit_data.builder.build_aggregate_return(&ret_args);

    Ok(())
}

// Emit the code to call another RISC-V function, and also the code to save return
// values to allocas.
fn emit_call_riscv_func<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    riscv_func_address: u64,
) -> Result<(), Error> {
    let func_llvm_value = *(emit_data
        .emitted_funcs
        .get(&riscv_func_address)
        .ok_or_else(|| {
            Error::External(format!(
                "Function at 0x{:x} does not exist!",
                riscv_func_address
            ))
        })?);

    emit_call_riscv_func_with_func_value(
        context,
        emit_data,
        emitting_func,
        Either::Left(func_llvm_value),
    )
}

fn emit_call_riscv_func_with_func_value<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    func_llvm_value: Either<FunctionValue<'a>, PointerValue<'a>>,
) -> Result<(), Error> {
    let i64t = context.i64_type();
    let machine = emitting_func.machine;
    let allocas = &emitting_func.allocas;

    // Keep track of previous last_ra value for nested calls
    let previous_last_ra = emit_load_from_machine(
        context,
        emit_data,
        machine,
        offset_of!(LlvmAotCoreMachineData, last_ra),
        i64t,
        Some("previous_last_ra"),
    )?;
    // Set current RA to last_ra to aid ret operations
    let last_ra = emit_load_reg(
        context,
        emit_data,
        machine,
        &allocas,
        registers::RA,
        Some("ra"),
    )?;
    emit_store_to_machine(
        context,
        emit_data,
        machine,
        last_ra,
        offset_of!(LlvmAotCoreMachineData, last_ra),
        i64t,
        Some("last_ra"),
    )?;

    let values = allocas.load_values(emit_data)?;
    let invoke_args = values.to_arguments()?;
    let result = match func_llvm_value {
        Either::Left(f) => emit_data
            .builder
            .build_call(f, &invoke_args, "riscv_call_result"),
        Either::Right(p) => emit_data.builder.build_indirect_call(
            riscv_function_type(context),
            p,
            &invoke_args,
            "riscv_call_result",
        ),
    };
    result.set_call_convention(HHVM_CALL_CONV);
    // When returned from the function call, restore previously saved last_ra
    emit_store_to_machine(
        context,
        emit_data,
        machine,
        previous_last_ra,
        offset_of!(LlvmAotCoreMachineData, last_ra),
        i64t,
        Some("restored_last_ra"),
    )?;
    // Convert CallsiteValue to StructValue
    let aggregate_value: StructValue<'a> = result
        .try_as_basic_value()
        .left()
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| Error::External("Call return value is not a struct value!".to_string()))?;
    // Now save return values
    let mut ret_values = [BasicValueEnum::IntValue(i64t.const_int(0, false)); 14];
    for i in 0..14 {
        ret_values[i] = emit_data
            .builder
            .build_extract_value(aggregate_value, i as u32, &format!("riscv_call_ret{}", i))
            .unwrap();
    }
    let ret_values = TransientValues::from_return_values(ret_values, machine.into());
    allocas.store_values(emit_data, &ret_values)?;
    Ok(())
}

// Emit the code to call early exit function from a RISC-V function
fn emit_call_exit<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    reason: u8,
) -> Result<(), Error> {
    let values = emitting_func.allocas.load_values(emit_data)?;

    let i8t = context.i8_type();
    emit_store_to_machine(
        context,
        emit_data,
        emitting_func.machine,
        i8t.const_int(reason as u64, true),
        offset_of!(LlvmAotCoreMachineData, exit_aot_reason),
        i8t,
        Some("exit_aot_reason"),
    )?;
    let invoke_args = values.to_arguments()?;
    let call = emit_data
        .builder
        .build_call(emit_data.exit_func, &invoke_args, "");
    call.set_call_convention(HHVM_CALL_CONV);
    emit_data.builder.build_unreachable();
    Ok(())
}

// Emit code used to locate FFI function together with the data object.
fn emit_env_ffi_function<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    machine: IntValue<'a>,
    offset: usize,
) -> Result<(PointerValue<'a>, IntValue<'a>), Error> {
    let i64t = context.i64_type();
    let env = emit_load_from_machine(
        context,
        emit_data,
        machine,
        offset_of!(LlvmAotCoreMachineData, env),
        i64t,
        Some("env"),
    )?;
    let data = emit_load_from_struct(
        context,
        emit_data,
        env,
        i64t.const_int(offset_of!(LlvmAotMachineEnv, data) as u64, false),
        i64t,
        Some("env_data"),
    )?;
    let ffi_function_value = emit_load_from_struct(
        context,
        emit_data,
        env,
        i64t.const_int(offset as u64, false),
        i64t,
        Some("env_ffi_function_value"),
    )?;
    let ffi_function = emit_data.builder.build_int_to_ptr(
        ffi_function_value,
        ffi_function_type(context).ptr_type(AddressSpace::default()),
        "env_ffi_function",
    );
    Ok((ffi_function, data))
}

// Emit a FFI call, also check the return result, when error happens,
// emit early exit function.
fn emit_ffi_call<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    args: &[BasicMetadataValueEnum<'a>; 3],
    side_effect: bool,
    name: Option<&str>,
) -> Result<IntValue<'a>, Error> {
    let name = name.unwrap_or("ffi_call_result");

    if side_effect {
        emit_cleanup(
            context,
            emit_data,
            &emitting_func.allocas.load_values(emit_data)?,
        )?;
    }

    let result = emit_data
        .builder
        .build_call(emit_data.ffi_wrapper_func, &args[..], name)
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();

    if side_effect {
        emitting_func.allocas.store_values(
            emit_data,
            &emit_setup(context, emit_data, emitting_func.machine)?,
        )?;
    }

    let success_block =
        context.append_basic_block(emitting_func.value, &format!("{}_success_block", name));
    let failure_block =
        context.append_basic_block(emitting_func.value, &format!("{}_failure_block", name));

    let cmp = emit_data.builder.build_int_compare(
        IntPredicate::NE,
        result,
        context
            .i64_type()
            .const_int(BARE_FUNC_ERROR_OR_TERMINATED, false),
        &format!("{}_ne_error_or_terminated", name),
    );
    emit_data
        .builder
        .build_conditional_branch(cmp, success_block, failure_block);

    emit_data.builder.position_at_end(failure_block);
    emit_call_exit(
        context,
        emit_data,
        emitting_func,
        EXIT_REASON_BARE_CALL_EXIT,
    )?;

    emit_data.builder.position_at_end(success_block);

    Ok(result)
}

// Given a set of icmp inputs, perform the actual cmp operation, if
// true, branch to +true_block+, otherwise create a new block to jump
// to.
fn emit_match_or_new_block<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    function: FunctionValue<'a>,
    predicate: IntPredicate,
    lhs: IntValue<'a>,
    rhs: IntValue<'a>,
    true_block: BasicBlock<'a>,
) -> BasicBlock<'a> {
    let else_block = context.append_basic_block(function, "select_control_else_block");

    let cmp = emit_data
        .builder
        .build_int_compare(predicate, lhs, rhs, "select_control_cmp");
    emit_data
        .builder
        .build_conditional_branch(cmp, true_block, else_block);

    else_block
}

// Emit a binary search pattern picking target to branch to. Jump
// to failure_block if no matching value is found.
fn emit_select_control<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    function: FunctionValue<'a>,
    current_block: BasicBlock<'a>,
    test_value: IntValue<'a>,
    targets: &[(u64, BasicBlock<'a>)],
    failure_block: BasicBlock<'a>,
) -> Result<(), Error> {
    emit_data.builder.position_at_end(current_block);
    if targets.is_empty() {
        emit_data.builder.build_unconditional_branch(failure_block);
        return Ok(());
    }

    let mid = targets.len() / 2;
    let mid_value = context.i64_type().const_int(targets[mid].0, false);
    // First test for equality
    let ne_block = emit_match_or_new_block(
        context,
        emit_data,
        function,
        IntPredicate::EQ,
        test_value,
        mid_value,
        targets[mid].1,
    );

    emit_data.builder.position_at_end(ne_block);
    let right_block = if mid > 0 {
        // Test for left branch
        let left_block = context.append_basic_block(function, "select_control_left_block");
        emit_select_control(
            context,
            emit_data,
            function,
            left_block,
            test_value,
            &targets[0..mid],
            failure_block,
        )?;

        emit_data.builder.position_at_end(ne_block);
        emit_match_or_new_block(
            context,
            emit_data,
            function,
            IntPredicate::ULT,
            test_value,
            mid_value,
            left_block,
        )
    } else {
        // Left branch is empty, switch to right branch directly
        ne_block
    };

    emit_data.builder.position_at_end(right_block);
    if mid < targets.len() - 1 {
        // Test for right branch
        emit_select_control(
            context,
            emit_data,
            function,
            right_block,
            test_value,
            &targets[(mid + 1)..targets.len()],
            failure_block,
        )?;
    } else {
        // Right branch is empty, matching results in a failure
        emit_data.builder.build_unconditional_branch(failure_block);
    }

    Ok(())
}

// Emit a CKB-VM AST value via LLVM
fn emit_value<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    value: &Value,
    name: Option<&str>,
) -> Result<IntValue<'a>, Error> {
    let machine = emitting_func.machine;
    let memory_start = emitting_func.memory_start;
    let function = emitting_func.value;
    let allocas = &emitting_func.allocas;

    let force_name = name.unwrap_or("uKNOWn_value");
    let i64t = context.i64_type();
    let i1t = context.bool_type();

    match value {
        Value::Imm(i) => Ok(i64t.const_int(*i, false)),
        Value::Register(r) => emit_load_reg(context, emit_data, machine, allocas, *r, name),
        Value::Op1(op, val) => {
            let val = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*val,
                Some(&format!("{}_val", force_name)),
            )?;
            Ok(match op {
                ActionOp1::Not => emit_data
                    .builder
                    .build_not(val, &format!("{}_nottemp", force_name)),
                ActionOp1::LogicalNot => emit_data.builder.build_and(
                    i64t.const_int(1, false),
                    emit_data
                        .builder
                        .build_not(val, &format!("{}_nottemp", force_name)),
                    &format!("{}_logicalnottemp", force_name),
                ),
                ActionOp1::Clz => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.ctlz.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for ctlz!".to_string()))
                        }
                    };
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into()])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for ctlz!".to_string())
                        })?;

                    emit_data
                        .builder
                        .build_call(
                            intrinsic_func,
                            &[val.into(), i1t.const_int(0, false).into()],
                            &format!("{}_clz", force_name),
                        )
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
                ActionOp1::Ctz => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.cttz.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for cttz!".to_string()))
                        }
                    };
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into()])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for cttz!".to_string())
                        })?;

                    emit_data
                        .builder
                        .build_call(
                            intrinsic_func,
                            &[val.into(), i1t.const_int(0, false).into()],
                            &format!("{}_ctz", force_name),
                        )
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
                ActionOp1::Cpop => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.ctpop.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for ctpop!".to_string()))
                        }
                    };
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into()])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for ctpop!".to_string())
                        })?;

                    emit_data
                        .builder
                        .build_call(
                            intrinsic_func,
                            &[val.into()],
                            &format!("{}_cpop", force_name),
                        )
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
                ActionOp1::Orcb => {
                    let mut result = i64t.const_int(0, false);
                    for i in 0..(Value::BITS / 8) {
                        let mask = 0xFFu64 << (i * 8);
                        let mask = i64t.const_int(mask, false);

                        let masked_value = emit_data.builder.build_and(
                            val,
                            mask,
                            &format!("{}_orcb_masked{}", force_name, i),
                        );
                        let cmp_value = emit_data.builder.build_int_compare(
                            IntPredicate::NE,
                            masked_value,
                            i64t.const_int(0, false),
                            &format!("{}_orcb_cmp{}", force_name, i),
                        );
                        let or_value = emit_data
                            .builder
                            .build_select(
                                cmp_value,
                                mask,
                                i64t.const_int(0, false),
                                &format!("{}_orcb_or{}", force_name, i),
                            )
                            .into_int_value();
                        result = emit_data.builder.build_or(
                            result,
                            or_value,
                            &format!("{}_orcb_round{}", force_name, i),
                        );
                    }
                    result
                }
                ActionOp1::Rev8 => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.bswap.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for bswap!".to_string()))
                        }
                    };
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into()])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for bswap!".to_string())
                        })?;

                    emit_data
                        .builder
                        .build_call(
                            intrinsic_func,
                            &[val.into()],
                            &format!("{}_rev8", force_name),
                        )
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
            })
        }
        Value::Op2(op, lhs, rhs) => {
            let lhs = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*lhs,
                Some(&format!("{}_lhs", force_name)),
            )?;
            let rhs = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*rhs,
                Some(&format!("{}_rhs", force_name)),
            )?;
            Ok(match op {
                ActionOp2::Add => {
                    emit_data
                        .builder
                        .build_int_add(lhs, rhs, &format!("{}_add", force_name))
                }
                ActionOp2::Sub => {
                    emit_data
                        .builder
                        .build_int_sub(lhs, rhs, &format!("{}_sub", force_name))
                }
                ActionOp2::Mul => {
                    emit_data
                        .builder
                        .build_int_mul(lhs, rhs, &format!("{}_mul", force_name))
                }
                ActionOp2::Bitand => {
                    emit_data
                        .builder
                        .build_and(lhs, rhs, &format!("{}_and", force_name))
                }
                ActionOp2::Bitor => {
                    emit_data
                        .builder
                        .build_or(lhs, rhs, &format!("{}_or", force_name))
                }
                ActionOp2::Bitxor => {
                    emit_data
                        .builder
                        .build_xor(lhs, rhs, &format!("{}_xor", force_name))
                }
                ActionOp2::Shl => {
                    emit_data
                        .builder
                        .build_left_shift(lhs, rhs, &format!("{}_shl", force_name))
                }
                ActionOp2::Eq => {
                    let v = emit_data.builder.build_int_compare(
                        IntPredicate::EQ,
                        lhs,
                        rhs,
                        &format!("{}_eq_i1", force_name),
                    );
                    emit_data
                        .builder
                        .build_select(
                            v,
                            i64t.const_int(1, false),
                            i64t.const_int(0, false),
                            &format!("{}_eq", force_name),
                        )
                        .into_int_value()
                }
                ActionOp2::Mulhsu => {
                    let t = machine_word_type(context)?;
                    let widening_t = double_machine_word_type(context)?;

                    let widening_lhs = emit_data.builder.build_int_cast_sign_flag(
                        lhs,
                        widening_t,
                        true,
                        &format!("{}_widening_lhs", force_name),
                    );
                    let widening_rhs = emit_data.builder.build_int_cast_sign_flag(
                        rhs,
                        widening_t,
                        false,
                        &format!("{}_widening_rhs", force_name),
                    );
                    let widening_mul = emit_data.builder.build_int_mul(
                        widening_lhs,
                        widening_rhs,
                        &format!("{}_widening_mul", force_name),
                    );
                    let shifted = emit_data.builder.build_right_shift(
                        widening_mul,
                        widening_t.const_int(Value::BITS as u64, false),
                        true,
                        &format!("{}_widening_shifted", force_name),
                    );
                    emit_data.builder.build_int_cast_sign_flag(
                        shifted,
                        t,
                        false,
                        &format!("{}_mulhsu", force_name),
                    )
                }
                ActionOp2::Clmul => {
                    let t = machine_word_type(context)?;
                    let mut result = t.const_int(0, false);
                    for i in 0..Value::BITS {
                        let rhs_shifted = emit_data.builder.build_right_shift(
                            rhs,
                            t.const_int(i as u64, false),
                            false,
                            &format!("{}_rhs_shifted{}", force_name, i),
                        );
                        let shifted_and1 = emit_data.builder.build_and(
                            rhs_shifted,
                            t.const_int(1, false),
                            &format!("{}_rhs_shifted{}_and1", force_name, i),
                        );
                        let cmp = emit_data.builder.build_int_compare(
                            IntPredicate::NE,
                            shifted_and1,
                            t.const_int(0, false),
                            &format!("{}_rhs_shifted{}_and1_cmp", force_name, i),
                        );
                        let lhs_shifted = emit_data.builder.build_left_shift(
                            lhs,
                            t.const_int(i as u64, false),
                            &format!("{}_lhs_shifted{}", force_name, i),
                        );
                        let xor_target = emit_data
                            .builder
                            .build_select(
                                cmp,
                                lhs_shifted,
                                t.const_int(0, false),
                                &format!("{}_xor_target{}", force_name, i),
                            )
                            .into_int_value();
                        result = emit_data.builder.build_xor(
                            result,
                            xor_target,
                            &format!("{}_clmul_round{}", force_name, i),
                        );
                    }
                    result
                }
                ActionOp2::Clmulh => {
                    let t = machine_word_type(context)?;
                    let mut result = t.const_int(0, false);
                    for i in 1..Value::BITS {
                        let rhs_shifted = emit_data.builder.build_right_shift(
                            rhs,
                            t.const_int(i as u64, false),
                            false,
                            &format!("{}_rhs_shifted{}", force_name, i),
                        );
                        let shifted_and1 = emit_data.builder.build_and(
                            rhs_shifted,
                            t.const_int(1, false),
                            &format!("{}_rhs_shifted{}_and1", force_name, i),
                        );
                        let cmp = emit_data.builder.build_int_compare(
                            IntPredicate::NE,
                            shifted_and1,
                            t.const_int(0, false),
                            &format!("{}_rhs_shifted{}_and1_cmp", force_name, i),
                        );
                        let lhs_shifted = emit_data.builder.build_right_shift(
                            lhs,
                            emit_data.builder.build_int_sub(
                                t.const_int(Value::BITS as u64, false),
                                t.const_int(i as u64, false),
                                &format!("{}_lhs_shifted_amount{}", force_name, i),
                            ),
                            false,
                            &format!("{}_lhs_shifted{}", force_name, i),
                        );
                        let xor_target = emit_data
                            .builder
                            .build_select(
                                cmp,
                                lhs_shifted,
                                t.const_int(0, false),
                                &format!("{}_xor_target{}", force_name, i),
                            )
                            .into_int_value();
                        result = emit_data.builder.build_xor(
                            result,
                            xor_target,
                            &format!("{}_clmul_round{}", force_name, i),
                        );
                    }
                    result
                }
                ActionOp2::Clmulr => {
                    let t = machine_word_type(context)?;
                    let mut result = t.const_int(0, false);
                    for i in 0..Value::BITS {
                        let rhs_shifted = emit_data.builder.build_right_shift(
                            rhs,
                            t.const_int(i as u64, false),
                            false,
                            &format!("{}_rhs_shifted{}", force_name, i),
                        );
                        let shifted_and1 = emit_data.builder.build_and(
                            rhs_shifted,
                            t.const_int(1, false),
                            &format!("{}_rhs_shifted{}_and1", force_name, i),
                        );
                        let cmp = emit_data.builder.build_int_compare(
                            IntPredicate::NE,
                            shifted_and1,
                            t.const_int(0, false),
                            &format!("{}_rhs_shifted{}_and1_cmp", force_name, i),
                        );
                        let lhs_shifted = emit_data.builder.build_right_shift(
                            lhs,
                            emit_data.builder.build_int_sub(
                                t.const_int(Value::BITS as u64 - 1, false),
                                t.const_int(i as u64, false),
                                &format!("{}_lhs_shifted_amount{}", force_name, i),
                            ),
                            false,
                            &format!("{}_lhs_shifted{}", force_name, i),
                        );
                        let xor_target = emit_data
                            .builder
                            .build_select(
                                cmp,
                                lhs_shifted,
                                t.const_int(0, false),
                                &format!("{}_xor_target{}", force_name, i),
                            )
                            .into_int_value();
                        result = emit_data.builder.build_xor(
                            result,
                            xor_target,
                            &format!("{}_clmul_round{}", force_name, i),
                        );
                    }
                    result
                }
                ActionOp2::Rol => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.fshl.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for fshl!".to_string()))
                        }
                    };
                    // Even though the intrinsic actually takes 3 arguments, we are only using the
                    // first one here to locate overloaded intrinsic, which has only one placeholder
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into(); 1])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for fshl!".to_string())
                        })?;

                    let args = [lhs.into(), lhs.into(), rhs.into()];
                    emit_data
                        .builder
                        .build_call(intrinsic_func, &args, &format!("{}_rol", force_name))
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
                ActionOp2::Ror => {
                    let t = machine_word_type(context)?;

                    let intrinsic = match Intrinsic::find("llvm.fshr.*") {
                        Some(i) => i,
                        None => {
                            return Err(Error::External("Missing intrinsic for fshr!".to_string()))
                        }
                    };
                    let intrinsic_func = intrinsic
                        .get_declaration(&emit_data.module, &[t.into(); 1])
                        .ok_or_else(|| {
                            Error::External("Unable to get declaration for fshr!".to_string())
                        })?;

                    let args = [lhs.into(), lhs.into(), rhs.into()];
                    emit_data
                        .builder
                        .build_call(intrinsic_func, &args, &format!("{}_ror", force_name))
                        .try_as_basic_value()
                        .unwrap_left()
                        .into_int_value()
                }
            })
        }
        Value::SignOp2(op, lhs, rhs_original, signed) => {
            let lhs = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*lhs,
                Some(&format!("{}_lhs", force_name)),
            )?;
            let rhs = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*rhs_original,
                Some(&format!("{}_rhs", force_name)),
            )?;
            Ok(match op {
                SignActionOp2::Shr => {
                    let suffix = if *signed { "ashr" } else { "lshr" };
                    emit_data.builder.build_right_shift(
                        lhs,
                        rhs,
                        *signed,
                        &format!("{}_{}", force_name, suffix),
                    )
                }
                SignActionOp2::Lt => {
                    let p = if *signed {
                        IntPredicate::SLT
                    } else {
                        IntPredicate::ULT
                    };
                    let suffix = if *signed { "alt" } else { "llt" };
                    let v = emit_data.builder.build_int_compare(
                        p,
                        lhs,
                        rhs,
                        &format!("{}_{}_i1", force_name, suffix),
                    );
                    emit_data
                        .builder
                        .build_select(
                            v,
                            i64t.const_int(1, false),
                            i64t.const_int(0, false),
                            &format!("{}_{}", force_name, suffix),
                        )
                        .into_int_value()
                }
                SignActionOp2::Extend => {
                    // For certain rhs value, we can build shortcuts.
                    let target_type = match &**rhs_original {
                        Value::Imm(i) if *i == 8 => Some(context.i8_type()),
                        Value::Imm(i) if *i == 16 => Some(context.i16_type()),
                        Value::Imm(i) if *i == 32 => Some(context.i32_type()),
                        Value::Imm(i) if *i == 64 => Some(i64t),
                        _ => None,
                    };
                    if let Some(target_type) = target_type {
                        let v = emit_data.builder.build_int_cast_sign_flag(
                            lhs,
                            target_type,
                            *signed,
                            &format!("{}_cast", force_name),
                        );
                        if *signed {
                            emit_data.builder.build_int_s_extend(
                                v,
                                i64t,
                                &format!("{}_shortcut_extend", force_name),
                            )
                        } else {
                            emit_data.builder.build_int_z_extend(
                                v,
                                i64t,
                                &format!("{}_shortcut_extend", force_name),
                            )
                        }
                    } else {
                        let shifts = emit_data.builder.build_int_sub(
                            i64t.const_int(64, false),
                            rhs,
                            &format!("{}_shifts", force_name),
                        );
                        emit_data.builder.build_right_shift(
                            emit_data.builder.build_left_shift(
                                lhs,
                                shifts,
                                &format!("{}_slowpath_extend_intermediate", force_name),
                            ),
                            shifts,
                            *signed,
                            &format!("{}_slowpath_extend", force_name),
                        )
                    }
                }
                SignActionOp2::Mulh => {
                    let i128t = context.i128_type();
                    let lhs128 = emit_data.builder.build_int_cast_sign_flag(
                        lhs,
                        i128t,
                        *signed,
                        &format!("{}_lhs_128", force_name),
                    );
                    let rhs128 = emit_data.builder.build_int_cast_sign_flag(
                        rhs,
                        i128t,
                        *signed,
                        &format!("{}_rhs_128", force_name),
                    );
                    let result_128 = emit_data.builder.build_int_mul(
                        lhs128,
                        rhs128,
                        &format!("{}_result_128", force_name),
                    );
                    // It doesn't matter what shift we use
                    let result_shifts_128 = emit_data.builder.build_right_shift(
                        result_128,
                        i128t.const_int(64, false),
                        false,
                        &format!("{}_result_shifts_128", force_name),
                    );
                    emit_data.builder.build_int_cast_sign_flag(
                        result_shifts_128,
                        i64t,
                        *signed,
                        &format!("{}_mulh", force_name),
                    )
                }
                SignActionOp2::Div | SignActionOp2::Rem => {
                    let is_div = if let SignActionOp2::Div = op {
                        true
                    } else {
                        false
                    };
                    let current_block = emit_data.builder.get_insert_block().unwrap();
                    let zero_block = context
                        .append_basic_block(function, &format!("{}_zero_rhs_block", force_name));
                    let non_zero_block = context
                        .append_basic_block(function, &format!("{}_zero_rhs_block", force_name));
                    let final_merge_block = context
                        .append_basic_block(function, &format!("{}_final_merge_block", force_name));

                    emit_data.builder.position_at_end(current_block);
                    let rhs_is_0 = emit_data.builder.build_int_compare(
                        IntPredicate::EQ,
                        rhs,
                        i64t.const_int(0, false),
                        &format!("{}_rhs_is_zero", force_name),
                    );
                    emit_data.builder.build_conditional_branch(
                        rhs_is_0,
                        zero_block,
                        non_zero_block,
                    );

                    emit_data.builder.position_at_end(zero_block);
                    let zero_result = if is_div {
                        i64t.const_int(u64::max_value(), false)
                    } else {
                        lhs
                    };
                    emit_data
                        .builder
                        .build_unconditional_branch(final_merge_block);

                    let (else_result, else_block) = if *signed {
                        let overflow_block = context.append_basic_block(
                            function,
                            &format!("{}_overflow_block", force_name),
                        );
                        let compute_block = context
                            .append_basic_block(function, &format!("{}_compute_block", force_name));
                        let non_zero_merge_block = context.append_basic_block(
                            function,
                            &format!("{}_non_zero_merge_block", force_name),
                        );

                        emit_data.builder.position_at_end(non_zero_block);
                        let overflow = emit_data.builder.build_and(
                            emit_data.builder.build_int_compare(
                                IntPredicate::EQ,
                                lhs,
                                i64t.const_int(i64::min_value() as u64, true),
                                &format!("{}_overflow_lhs", force_name),
                            ),
                            emit_data.builder.build_int_compare(
                                IntPredicate::EQ,
                                rhs,
                                i64t.const_int((-1i64) as u64, true),
                                &format!("{}_overflow_rhs", force_name),
                            ),
                            &format!("{}_overflow", force_name),
                        );
                        emit_data.builder.build_conditional_branch(
                            overflow,
                            overflow_block,
                            compute_block,
                        );
                        emit_data.builder.position_at_end(overflow_block);
                        let overflow_result = if is_div {
                            lhs
                        } else {
                            i64t.const_int(0, false)
                        };
                        emit_data
                            .builder
                            .build_unconditional_branch(non_zero_merge_block);
                        emit_data.builder.position_at_end(compute_block);
                        let compute_result = if is_div {
                            emit_data.builder.build_int_signed_div(
                                lhs,
                                rhs,
                                &format!("{}_actual_sdiv", force_name),
                            )
                        } else {
                            emit_data.builder.build_int_signed_rem(
                                lhs,
                                rhs,
                                &format!("{}_actual_srem", force_name),
                            )
                        };
                        emit_data
                            .builder
                            .build_unconditional_branch(non_zero_merge_block);
                        emit_data.builder.position_at_end(non_zero_merge_block);
                        let non_zero_merge_result = emit_data
                            .builder
                            .build_phi(i64t, &format!("{}_non_zero_merge_result", force_name));
                        non_zero_merge_result.add_incoming(&[
                            (&overflow_result, overflow_block),
                            (&compute_result, compute_block),
                        ]);
                        emit_data
                            .builder
                            .build_unconditional_branch(final_merge_block);
                        (non_zero_merge_result.as_basic_value(), non_zero_merge_block)
                    } else {
                        emit_data.builder.position_at_end(non_zero_block);
                        let compute_result = if is_div {
                            emit_data.builder.build_int_unsigned_div(
                                lhs,
                                rhs,
                                &format!("{}_actual_unsigned_v", force_name),
                            )
                        } else {
                            emit_data.builder.build_int_unsigned_rem(
                                lhs,
                                rhs,
                                &format!("{}_actual_urem", force_name),
                            )
                        };
                        emit_data
                            .builder
                            .build_unconditional_branch(final_merge_block);
                        (compute_result.into(), non_zero_block)
                    };

                    emit_data.builder.position_at_end(final_merge_block);
                    let final_result = emit_data
                        .builder
                        .build_phi(i64t, &format!("{}_final_merge_result", force_name));
                    final_result
                        .add_incoming(&[(&zero_result, zero_block), (&else_result, else_block)]);

                    final_result.as_basic_value().into_int_value()
                }
            })
        }
        Value::Cond(c, t, f) => {
            let t = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*t,
                Some(&format!("{}_t", force_name)),
            )?;
            let f = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*f,
                Some(&format!("{}_f", force_name)),
            )?;
            let c = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*c,
                Some(&format!("{}_c", force_name)),
            )?;
            let c = emit_data.builder.build_int_compare(
                IntPredicate::EQ,
                c,
                i64t.const_int(1, false),
                &format!("{}_c_i1", force_name),
            );
            Ok(emit_data
                .builder
                .build_select(c, t, f, &format!("{}_cond", force_name))
                .into_int_value())
        }
        Value::Load(addr, size) => {
            let t = size_to_type(context, *size)?;
            // TODO: maybe we should provide 2 modes:
            // * Safe mode adds memory boundary checks
            // * Fast mode relies on mmap-ed pages and OS to detect memory overflows
            let addr = emit_value(
                context,
                emit_data,
                &emitting_func,
                &*addr,
                Some(&format!("{}_addr", force_name)),
            )?;
            let real_address_value = emit_data.builder.build_int_add(
                memory_start,
                addr,
                &format!("{}_real_addr_val", force_name),
            );
            let real_address = emit_data.builder.build_int_to_ptr(
                real_address_value,
                t.ptr_type(AddressSpace::default()),
                &format!("{}_real_addr", force_name),
            );
            let value = emit_data
                .builder
                .build_load(t, real_address, &format!("{}_loaded_value", force_name))
                .into_int_value();
            Ok(emit_data.builder.build_int_cast_sign_flag(
                value,
                i64t,
                false,
                &format!("{}_load_casted", force_name),
            ))
        }
    }
}

// Emit a series of writes atomically
fn emit_writes<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    emitting_func: &EmittingFunc<'a>,
    writes: &[Write],
    prefix: &str,
) -> Result<(), Error> {
    let mut memory_ops = Vec::new();
    let mut register_ops = Vec::new();

    for (i, write) in writes.iter().enumerate() {
        let prefix = format!("{}_write{}", prefix, i);
        match write {
            Write::Memory {
                address,
                size,
                value,
            } => {
                let address = emit_value(
                    context,
                    emit_data,
                    emitting_func,
                    address,
                    Some(&format!("{}_address", prefix)),
                )?;
                let value = emit_value(
                    context,
                    emit_data,
                    emitting_func,
                    value,
                    Some(&format!("{}_value", prefix)),
                )?;
                let t = size_to_type(context, *size)?;
                let casted_value = emit_data.builder.build_int_cast_sign_flag(
                    value,
                    t,
                    false,
                    &format!("{}_casted_value", prefix),
                );
                memory_ops.push((address, casted_value, t, prefix));
            }
            Write::Register { index, value } => {
                let value = emit_value(
                    context,
                    emit_data,
                    emitting_func,
                    value,
                    Some(&format!("{}_value", prefix)),
                )?;

                register_ops.push((index, value));
            }
        }
    }

    for (address, value, t, prefix) in memory_ops {
        let real_address_value = emit_data.builder.build_int_add(
            emitting_func.memory_start,
            address,
            &format!("{}_real_addr_val", prefix),
        );
        let real_address = emit_data.builder.build_int_to_ptr(
            real_address_value,
            t.ptr_type(AddressSpace::default()),
            &format!("{}_real_addr", prefix),
        );
        emit_data.builder.build_store(real_address, value);
    }
    for (index, value) in register_ops {
        emit_store_reg(
            context,
            emit_data,
            emitting_func.machine,
            &emitting_func.allocas,
            *index,
            value,
        )?;
    }
    Ok(())
}

// Emit a RISC-V function
fn emit_riscv_func<'a>(
    context: &'a Context,
    emit_data: &EmitData<'a>,
    debug_data: &mut Option<DebugData<'a>>,
    func: &Func,
) -> Result<(), Error> {
    if func.basic_blocks.is_empty() {
        return Err(Error::External(format!(
            "Func {} at 0x{:x} does not have basic blocks!",
            func.force_name(true),
            func.range.start
        )));
    }
    let i64t = context.i64_type();

    let function = emit_data.emitted_funcs[&func.range.start];

    if let Some(debug_data) = debug_data {
        let di_function_type = {
            let di_i64t = debug_data
                .di_builder
                .create_basic_type(
                    "uint64", 64, 0x07, // DW_ATE_unsigned
                    0,    // DIFlags
                )
                .map_err(|e| Error::External(e.to_string()))?
                .as_type();

            let argts = [di_i64t; 15];
            let return_type = debug_data.di_builder.create_struct_type(
                debug_data.compile_file.as_debug_info_scope(),
                "riscv_return_struct",
                debug_data.compile_file,
                0, // LineNumber
                14 * 64,
                0,
                0, // Flags
                None,
                &argts[0..14],
                0,
                None,
                "",
            );

            debug_data.di_builder.create_subroutine_type(
                debug_data.compile_file,
                Some(return_type.as_type()),
                &argts,
                0,
            )
        };

        let di_function = debug_data.di_builder.create_function(
            debug_data.compile_file.as_debug_info_scope(),
            &format!("{}{}", emit_data.code.symbol_prefix, func.force_name(false)),
            None,
            debug_data.compile_file,
            0,
            di_function_type,
            false,
            true,
            0,
            0,
            true,
        );
        function.set_subprogram(di_function);

        debug_data.set_scope(di_function.as_debug_info_scope());
        debug_data.write(&format!(
            "<{} (0x{:x})>:",
            func.force_name(true),
            func.range.start,
        ))?;
        emit_data
            .builder
            .set_current_debug_location(debug_data.debug_location(context));
    }

    let mut emitting_func = {
        // This is a dummy entry block in case some jumps point to the start of
        // the function. LLVM does not allow predecessors for entry block.
        let entry_block = context.append_basic_block(
            function,
            &format!("basic_block_entry_{}", func.force_name(true)),
        );

        let basic_blocks: HashMap<u64, BasicBlock<'a>> = {
            let mut bbs = HashMap::default();

            for block in &func.basic_blocks {
                let b = context
                    .append_basic_block(function, &emit_data.basic_block_name(block.range.start));

                bbs.insert(block.range.start, b);
            }

            bbs
        };
        emit_data.builder.position_at_end(entry_block);

        // Fetch arguments
        let args = {
            let mut args = [i64t.const_int(0, false); 15];
            for i in 0..15 {
                args[i] = function.get_nth_param(i as u32).unwrap().into_int_value();
            }
            TransientValues::from_arguments(args)
        };
        let machine = args.extract_machine()?;

        // Build allocas for arguments
        let vars = {
            let entry_block = function.get_first_basic_block().unwrap();
            emit_data.builder.position_at_end(entry_block);

            let alloca_args = args.map(|arg, mapping| {
                let var = emit_data
                    .builder
                    .build_alloca(i64t, &format!("alloc_{}", mapping));
                emit_data.builder.build_store(var, arg);
                Ok(var)
            })?;

            RegAllocas::new(alloca_args, i64t)
        };
        let pc_alloca = vars.pc_alloca()?;

        // Build one memory_start construct per function
        let memory_start = emit_load_from_machine(
            context,
            emit_data,
            machine,
            offset_of!(LlvmAotCoreMachineData, memory),
            i64t,
            Some("memory_start"),
        )?;

        let indirect_dispatcher_alloca = emit_data
            .builder
            .build_alloca(i64t, "indirect_dispatch_test_alloca");

        EmittingFunc {
            basic_blocks,
            value: function,
            machine,
            allocas: vars,
            pc_alloca,
            memory_start,
            indirect_dispatcher_alloca,
            indirect_dispatcher: None,
            ret_block: None,
        }
    };

    // Jump to the first actual basic block
    emit_data
        .builder
        .build_unconditional_branch(emitting_func.basic_blocks[&func.range.start]);

    let mut control_blocks: HashMap<u64, BasicBlock<'a>> = HashMap::default();
    // Emit code for each basic block
    for block in &func.basic_blocks {
        emit_data
            .builder
            .position_at_end(emitting_func.basic_blocks[&block.range.start]);

        if block.cycles > 0 {
            // Emit cycle calculation logic
            let current_cycles = emit_load_from_machine(
                context,
                emit_data,
                emitting_func.machine,
                offset_of!(LlvmAotCoreMachineData, cycles),
                i64t,
                Some("current_cycles"),
            )?;
            let max_cycles = emit_load_from_machine(
                context,
                emit_data,
                emitting_func.machine,
                offset_of!(LlvmAotCoreMachineData, max_cycles),
                i64t,
                Some("max_cycles"),
            )?;
            let updated_cycles = emit_data.builder.build_int_add(
                current_cycles,
                i64t.const_int(block.cycles, false),
                "updated_cycles",
            );
            let cycle_overflow_cmp = emit_data.builder.build_int_compare(
                IntPredicate::ULT,
                updated_cycles,
                current_cycles,
                "cycle_overflow_cmp",
            );
            let cycle_exceeded_cmp = emit_data.builder.build_int_compare(
                IntPredicate::UGT,
                updated_cycles,
                max_cycles,
                "cycle_exceeded_cmp",
            );
            let cycle_overflow_block = context.append_basic_block(
                function,
                &format!("cycle_overflow_block_0x{:x}", block.range.start),
            );
            let cycle_no_overflow_block = context.append_basic_block(
                function,
                &format!("cycle_no_overflow_block_0x{:x}", block.range.start),
            );
            let cycle_exceeded_block = context.append_basic_block(
                function,
                &format!("cycle_exceeded_block_0x{:x}", block.range.start),
            );
            let cycle_ok_block = context.append_basic_block(
                function,
                &format!("cycle_ok_block_0x{:x}", block.range.start),
            );

            emit_data.builder.build_conditional_branch(
                cycle_overflow_cmp,
                cycle_overflow_block,
                cycle_no_overflow_block,
            );
            emit_data.builder.position_at_end(cycle_overflow_block);
            emit_call_exit(
                context,
                emit_data,
                &emitting_func,
                EXIT_REASON_CYCLES_OVERFLOW,
            )?;
            emit_data.builder.position_at_end(cycle_no_overflow_block);
            emit_data.builder.build_conditional_branch(
                cycle_exceeded_cmp,
                cycle_exceeded_block,
                cycle_ok_block,
            );
            emit_data.builder.position_at_end(cycle_exceeded_block);
            emit_call_exit(
                context,
                emit_data,
                &emitting_func,
                EXIT_REASON_MAX_CYCLES_EXCEEDED,
            )?;
            emit_data.builder.position_at_end(cycle_ok_block);
            emit_store_to_machine(
                context,
                emit_data,
                emitting_func.machine,
                updated_cycles,
                offset_of!(LlvmAotCoreMachineData, cycles),
                i64t,
                Some("updated_cycles"),
            )?;
        }

        // Emit normal register writes, memory writes
        for (i, write_batch) in block.write_batches.iter().enumerate() {
            if let Some(debug_data) = debug_data {
                if let Some(debug_line) = block.debug_lines.get(i) {
                    debug_data.write(&format!("  {}", debug_line))?;
                    emit_data
                        .builder
                        .set_current_debug_location(debug_data.debug_location(context));
                }
            }
            let prefix = format!("batch{}", i);
            emit_writes(context, emit_data, &emitting_func, &write_batch, &prefix)?;
        }

        // Update PC & writes together with PC
        if let Some(debug_data) = debug_data {
            if let Some(debug_line) = block.debug_lines.get(block.write_batches.len()) {
                debug_data.write(&format!("  {}", debug_line))?;
                emit_data
                    .builder
                    .set_current_debug_location(debug_data.debug_location(context));
            }
        }
        let next_pc = emit_value(
            context,
            emit_data,
            &emitting_func,
            &block.control.pc(),
            Some("pc"),
        )?;
        if let Some(last_writes) = block.control.writes() {
            emit_writes(
                context,
                emit_data,
                &emitting_func,
                last_writes,
                "last_write",
            )?;
        }
        emit_data
            .builder
            .build_store(emitting_func.pc_alloca, next_pc);

        let control_block = context.append_basic_block(
            function,
            &format!("control_block_0x{:x}", block.range.start),
        );

        emit_data.builder.build_unconditional_branch(control_block);

        control_blocks.insert(block.range.start, control_block);
    }

    // Emit code for each control block
    for block in &func.basic_blocks {
        emit_data
            .builder
            .position_at_end(control_blocks[&block.range.start]);

        let next_pc = emit_data
            .builder
            .build_load(i64t, emitting_func.pc_alloca, "target_pc")
            .into_int_value();

        // Emit control flow changes, there might be several cases:
        // 1(a). Simple jump to another basic block, note this includes fallthroughs;
        // 1(b). A simple cond with 2 immediate values for 2 other known basic blocks,
        // this translates to LLVM CondBr
        // 1(c). Register based jumps, call by function pointers typically use this
        // 2. Function call
        // 3. Return
        // 4. Ecall
        // 5. Ebreak
        // Any other values will result in an early exit to the interpreter side.
        // Those are not invalid instructions per RISC-V specification, it's just that
        // AOT mode cannot infer the correct result, and has to leverage help on the
        // intepreter.
        let mut terminated = false;
        match &block.control {
            Control::Jump { pc, .. } => match pc {
                Value::Imm(i) => {
                    if let Some(target_block) = emitting_func.basic_blocks.get(i) {
                        // 1(a)
                        emit_data.builder.build_unconditional_branch(*target_block);
                        terminated = true;
                    }
                }
                Value::Cond(c, t, f) => {
                    if let (Value::Imm(t), Value::Imm(f)) = (&**t, &**f) {
                        if let (Some(true_block), Some(false_block)) = (
                            emitting_func.basic_blocks.get(t),
                            emitting_func.basic_blocks.get(f),
                        ) {
                            // 1(b)
                            let c =
                                emit_value(context, emit_data, &emitting_func, c, Some("pc_cond"))?;
                            let c = emit_data.builder.build_int_compare(
                                IntPredicate::EQ,
                                c,
                                i64t.const_int(1, false),
                                "pc_cond_i1",
                            );
                            emit_data.builder.build_conditional_branch(
                                c,
                                *true_block,
                                *false_block,
                            );
                            terminated = true;
                        }
                    }
                }
                _ => {}
            },
            Control::Call { address, .. } => {
                // 2
                emit_call_riscv_func(context, emit_data, &emitting_func, *address)?;
                if let Some(resume_address) = block.control.call_resume_address() {
                    // When returned from the call, update PC using resume address
                    emit_data.builder.build_store(
                        emitting_func.pc_alloca,
                        i64t.const_int(resume_address, false),
                    );
                    if let Some(resume_block) = emitting_func.basic_blocks.get(&resume_address) {
                        emit_data.builder.build_unconditional_branch(*resume_block);
                        terminated = true;
                    }
                }
            }
            Control::IndirectCall { .. } => {
                // 2
                // First, query the function to call via LlvmAotMachineEnv
                let (query_function, data) = emit_env_ffi_function(
                    context,
                    emit_data,
                    emitting_func.machine,
                    offset_of!(LlvmAotMachineEnv, query_function),
                )?;
                let mut query_args = [query_function.into(), data.into(), next_pc.into()];
                let query_result = emit_ffi_call(
                    context,
                    emit_data,
                    &emitting_func,
                    &mut query_args,
                    false,
                    Some("query_function_result"),
                )?;
                // There might be 3 cases here:
                // * query_result is +BARE_FUNC_ERROR_OR_TERMINATED+: errors happen at
                // Rust side, this case is handled within +emit_ffi_call+
                // * query_result is +BARE_FUNC_MISSING+, meaning Rust side failed
                // to find a native function. There might still be a case we want
                // to handle: loop unrolling generates from a function within current
                // function. See +memset+ from +newlib+ for an example here. In
                // this case, we first test if +next_pc+ lies within current function,
                // if so, we will handle it like an indirect jump.
                // * Any other value will be treated like a proper x64 function to
                // call.
                let normal_call_block = context.append_basic_block(function, "normal_call_block");
                let interpret_block = context.append_basic_block(function, "interpret_block");
                let resume_block = context.append_basic_block(function, "resume_block");

                let cmp = emit_data.builder.build_int_compare(
                    IntPredicate::NE,
                    query_result,
                    i64t.const_int(BARE_FUNC_MISSING, false),
                    "cmp_query_result_to_invalid_address",
                );
                emit_data
                    .builder
                    .build_conditional_branch(cmp, normal_call_block, interpret_block);

                emit_data.builder.position_at_end(normal_call_block);
                // When a proper function is returned(non-zero), use the function
                // to build the actual RISC-V call.
                let query_result_function = emit_data.builder.build_int_to_ptr(
                    query_result,
                    riscv_function_type(context).ptr_type(AddressSpace::default()),
                    "query_function_result_function",
                );
                emit_call_riscv_func_with_func_value(
                    context,
                    emit_data,
                    &emitting_func,
                    Either::Right(query_result_function),
                )?;
                emit_data.builder.build_unconditional_branch(resume_block);

                emit_data.builder.position_at_end(interpret_block);
                // Call the function via Rust side interpreter
                let (interpret_function, data) = emit_env_ffi_function(
                    context,
                    emit_data,
                    emitting_func.machine,
                    offset_of!(LlvmAotMachineEnv, interpret),
                )?;
                let mut interpret_args = [
                    interpret_function.into(),
                    data.into(),
                    i64t.const_int(1, false).into(),
                ];
                emit_ffi_call(
                    context,
                    emit_data,
                    &emitting_func,
                    &mut interpret_args,
                    true,
                    Some("interpret_function_result"),
                )?;
                emit_data.builder.build_unconditional_branch(resume_block);

                emit_data.builder.position_at_end(resume_block);
                if let Some(resume_address) = block.control.call_resume_address() {
                    // When returned from the call, update PC using resume address
                    emit_data.builder.build_store(
                        emitting_func.pc_alloca,
                        i64t.const_int(resume_address, false),
                    );
                    if let Some(resume_block) = emitting_func.basic_blocks.get(&resume_address) {
                        emit_data.builder.build_unconditional_branch(*resume_block);
                        terminated = true;
                    }
                }
            }
            Control::Tailcall { address, .. } => {
                // 2
                let func_llvm_value =
                    *(emit_data.emitted_funcs.get(&address).ok_or_else(|| {
                        Error::External(format!("Function at 0x{:x} does not exist!", address))
                    })?);

                let values = emitting_func.allocas.load_values(emit_data)?;
                let invoke_args = values.to_arguments()?;
                let result = emit_data.builder.build_call(
                    func_llvm_value,
                    &invoke_args,
                    "riscv_tailcall_result",
                );
                result.set_call_convention(HHVM_CALL_CONV);
                result.set_tail_call(true);
                emit_data
                    .builder
                    .build_return(Some(&result.try_as_basic_value().unwrap_left()));
                terminated = true;
            }
            Control::Return { .. } => {
                // 3
                let ret_block = emitting_func.fetch_ret_block(context, emit_data)?;
                emit_data.builder.build_unconditional_branch(ret_block);

                terminated = true;
            }
            Control::Ecall { .. } => {
                // 4
                let (ecall_function, data) = emit_env_ffi_function(
                    context,
                    emit_data,
                    emitting_func.machine,
                    offset_of!(LlvmAotMachineEnv, ecall),
                )?;
                let mut ecall_args = [
                    ecall_function.into(),
                    data.into(),
                    i64t.const_int(0, false).into(),
                ];
                emit_ffi_call(
                    context,
                    emit_data,
                    &emitting_func,
                    &mut ecall_args,
                    true,
                    None,
                )?;
                if let Some(target_block) = emitting_func.basic_blocks.get(&block.range.end) {
                    emit_data.builder.build_unconditional_branch(*target_block);
                } else {
                    // There might be cases that an ecall is put at the very end
                    // of code section to exit the program. For well-organized
                    // program, this might be the only place that is triggering
                    // an unnecessary arbitrary jump. Hence we will choose to revert
                    // to interpreter code here for a better tradeoff.
                    emit_call_exit(
                        context,
                        emit_data,
                        &emitting_func,
                        EXIT_REASON_REVERT_TO_INTERPRETER,
                    )?;
                }
                terminated = true;
            }
            Control::Ebreak { .. } => {
                // 5
                let (ebreak_function, data) = emit_env_ffi_function(
                    context,
                    emit_data,
                    emitting_func.machine,
                    offset_of!(LlvmAotMachineEnv, ebreak),
                )?;
                let mut ebreak_args = [
                    ebreak_function.into(),
                    data.into(),
                    i64t.const_int(0, false).into(),
                ];
                emit_ffi_call(
                    context,
                    emit_data,
                    &emitting_func,
                    &mut ebreak_args,
                    true,
                    None,
                )?;
                if let Some(target_block) = emitting_func.basic_blocks.get(&block.range.end) {
                    emit_data.builder.build_unconditional_branch(*target_block);
                    terminated = true;
                }
            }
        }

        if !terminated {
            debug!(
                "Control that triggers arbitrary jump: {:?}, block: {:x}",
                block.control, block.range.start
            );
            // Arbitrary jump here. Use interpreter to execute till next basic
            // block end.
            emitting_func.emit_arbitrary_jump(context, emit_data, next_pc)?;
        }
    }

    Ok(())
}

// Emit a wrapper function handling environment requirements for calling Rust
// funcs via FFI. At the moment, the only such requirement is to align the stack
// at 16 bytes, which has already been taken care of by the function attributes.
// One possible alternative, is to require all AOTed functions to align on 16 byte
// boundary. We actually tried this path, but the problem is: LLVM's alignment code
// is quite straightforward in that it always pollute rbp, while HHVM calling
// convention uses rbp as an input argument as well as returned result, causing
// incorrect code to be generated. That's why we settled on this FFI wrapper function.
fn emit_ffi_wrapper<'a>(context: &'a Context, emit_data: &EmitData<'a>) -> Result<(), Error> {
    let entry_block = context.append_basic_block(emit_data.ffi_wrapper_func, "__entry_block__");
    emit_data.builder.position_at_end(entry_block);

    let func = emit_data
        .ffi_wrapper_func
        .get_nth_param(0)
        .unwrap()
        .into_pointer_value();
    let env = emit_data.ffi_wrapper_func.get_nth_param(1).unwrap();
    let arg = emit_data.ffi_wrapper_func.get_nth_param(2).unwrap();

    let args = [env.into(), arg.into()];
    let result = emit_data.builder.build_indirect_call(
        ffi_function_type(context),
        func,
        &args,
        "wrapper_call_result",
    );
    emit_data
        .builder
        .build_return(Some(&result.try_as_basic_value().unwrap_left()));

    Ok(())
}

// Emit an early exit function from RISC-V, this does stack unwinding via longjmp
fn emit_exit<'a>(context: &'a Context, emit_data: &EmitData<'a>) -> Result<(), Error> {
    let longjmp_intrinsic = match Intrinsic::find("llvm.eh.sjlj.longjmp") {
        Some(i) => i,
        None => {
            return Err(Error::External(
                "Missing intrinsics for longjmp!".to_string(),
            ));
        }
    };
    let longjmp_type = match longjmp_intrinsic.get_declaration(&emit_data.module, &[]) {
        Some(declaration) => declaration.get_type(),
        None => {
            return Err(Error::External(
                "Error locating declaration type for longjmp!".to_string(),
            ))
        }
    };
    // Refer to the comments on +module_inline_asm+ method for more details
    let longjmp_func =
        emit_data
            .module
            .add_function(&emit_data.hosted_longjmp_name(), longjmp_type, None);

    let noreturn_attr_id = Attribute::get_named_enum_kind_id("noreturn");
    if noreturn_attr_id == 0 {
        return Err(Error::External(
            "LLVM is missing noreturn attr!".to_string(),
        ));
    }
    let noreturn_attr = context.create_enum_attribute(noreturn_attr_id, 0);
    longjmp_func.add_attribute(AttributeLoc::Function, noreturn_attr);

    let exit_function = emit_data.exit_func;

    let exit_block = context.append_basic_block(exit_function, "__exit_start_block__");
    emit_data.builder.position_at_end(exit_block);

    let i64t = context.i64_type();
    let vars = {
        let mut vars = [i64t.const_int(0, false); 15];
        for i in 0..15 {
            vars[i] = exit_function
                .get_nth_param(i as u32)
                .unwrap()
                .into_int_value();
        }
        TransientValues::from_arguments(vars)
    };
    let machine = vars.extract_machine()?;

    emit_cleanup(context, emit_data, &vars)?;

    // exit reason is passed within LlvmAotCoreMachineData

    let longjmp_args = [emit_data
        .builder
        .build_int_to_ptr(
            emit_data.builder.build_int_add(
                machine,
                i64t.const_int(offset_of!(LlvmAotCoreMachineData, jmpbuf) as u64, false),
                "jmpbuf_addr_val",
            ),
            i8pt(context),
            "jmpbuf_addr",
        )
        .into()];
    emit_data
        .builder
        .build_call(longjmp_func, &longjmp_args, "");
    emit_data.builder.build_unreachable();

    Ok(())
}

// Emit the entrypoint function to AOT code
fn emit_entry<'a>(context: &'a Context, emit_data: &EmitData<'a>) -> Result<(), Error> {
    let i64t = context.i64_type();
    let i32t = context.i32_type();
    let i8t = context.i8_type();

    let setjmp_intrinsic = match Intrinsic::find("llvm.eh.sjlj.setjmp") {
        Some(i) => i,
        None => {
            return Err(Error::External(
                "Missing intrinsics for setjmp!".to_string(),
            ))
        }
    };
    let setjmp_type = match setjmp_intrinsic.get_declaration(&emit_data.module, &[]) {
        Some(declaration) => declaration.get_type(),
        None => {
            return Err(Error::External(
                "Error locating declaration type for setjmp!".to_string(),
            ))
        }
    };
    // Refer to the comments on +module_inline_asm+ method for more details
    let setjmp_func =
        emit_data
            .module
            .add_function(&emit_data.hosted_setjmp_name(), setjmp_type, None);

    let entry_function = emit_data.entry_func;
    let entry_block = context.append_basic_block(entry_function, "__entry_start_block__");
    emit_data.builder.position_at_end(entry_block);
    let machine = entry_function.get_nth_param(0).unwrap().into_int_value();
    let target = entry_function.get_nth_param(1).unwrap().into_int_value();

    let call_block = context.append_basic_block(entry_function, "__entry_call_block__");
    let ret_block = context.append_basic_block(entry_function, "__entry_ret_block__");

    let setjmp_args = [emit_data
        .builder
        .build_int_to_ptr(
            emit_data.builder.build_int_add(
                machine,
                i64t.const_int(offset_of!(LlvmAotCoreMachineData, jmpbuf) as u64, false),
                "jmpbuf_addr_val",
            ),
            i8pt(context),
            "jmpbuf_addr",
        )
        .into()];
    let setjmp_result = emit_data
        .builder
        .build_call(setjmp_func, &setjmp_args, "setjmp_result")
        .try_as_basic_value()
        .unwrap_left()
        .into_int_value();
    let setjmp_cond = emit_data.builder.build_int_compare(
        IntPredicate::EQ,
        setjmp_result,
        i32t.const_int(0, false),
        "setjmp_result_cmp_zero",
    );
    emit_data
        .builder
        .build_conditional_branch(setjmp_cond, call_block, ret_block);

    emit_data.builder.position_at_end(call_block);
    let vars = emit_setup(context, emit_data, machine)?;

    let target_type = riscv_function_type(context);
    let target_pt = target_type.ptr_type(AddressSpace::default());
    let target_func =
        emit_data
            .builder
            .build_int_to_ptr(target, target_pt, "target_function_pointer");
    let invoke_args = vars.to_arguments()?;
    let target_result = emit_data.builder.build_indirect_call(
        target_type,
        target_func,
        &invoke_args,
        "target_function_result",
    );
    target_result.set_call_convention(HHVM_CALL_CONV);
    let aggregate_value: StructValue<'a> = target_result
        .try_as_basic_value()
        .left()
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| Error::External("Call return value is not a struct value!".to_string()))?;
    // In most cases, the code will result in the other branch of setjmp above.
    // A direct return here will be very rare, but we still need to perform cleanups.
    let mut target_return_values = [i64t.const_int(0, false); 14];
    for i in 0..14 {
        target_return_values[i] = emit_data
            .builder
            .build_extract_value(aggregate_value, i as u32, &format!("target_ret{}", i))
            .unwrap()
            .into_int_value();
    }
    let target_return_values = TransientValues::from_return_values(target_return_values, machine);
    emit_cleanup(context, emit_data, &target_return_values)?;
    emit_data.builder.build_unconditional_branch(ret_block);

    emit_data.builder.position_at_end(ret_block);
    // At this point, cleanups have been taken care of either by call_block or exit function
    let reason = emit_load_from_machine(
        context,
        emit_data,
        machine,
        offset_of!(LlvmAotCoreMachineData, exit_aot_reason),
        i8t,
        Some("exit_reason"),
    )?;

    emit_data.builder.build_return(Some(&reason));

    Ok(())
}

fn emit<'a>(
    context: &'a Context,
    emit_data: &mut EmitData<'a>,
    debug_data: &mut Option<DebugData<'a>>,
    optimize: bool,
) -> Result<(), Error> {
    let inline_asm = emit_data.module_inline_asm();
    emit_data.module.set_inline_assembly(&inline_asm);

    let table_len: u32 = (emit_data.code.funcs.len() * 2 + 1)
        .try_into()
        .map_err(|_e| {
            Error::External(format!(
                "There are too many functions provided: {}",
                emit_data.code.funcs.len()
            ))
        })?;

    let i64t = context.i64_type();
    let i8t = context.i8_type();

    let mut address_table_values = Vec::with_capacity(table_len as usize);
    address_table_values.push(i64t.const_int(emit_data.code.funcs.len() as u64, false));
    for func in &emit_data.code.funcs {
        // Build signature for each function here, so we can piece together calls later
        let function_name = format!("{}{}", emit_data.code.symbol_prefix, func.force_name(false));
        let function =
            emit_data
                .module
                .add_function(&function_name, riscv_function_type(context), None);
        function.set_call_conventions(HHVM_CALL_CONV);

        emit_data.emitted_funcs.insert(func.range.start, function);
        address_table_values.push(i64t.const_int(func.range.start, false));
        address_table_values.push(
            function
                .as_global_value()
                .as_pointer_value()
                .const_to_int(i64t),
        );
    }

    let address_table = i64t.const_array(&address_table_values);

    let address_table_type = i64t.array_type(table_len);
    let address_table_var = emit_data.module.add_global(
        address_table_type,
        None,
        &format!("{}____address_table____", emit_data.code.symbol_prefix),
    );
    address_table_var.set_alignment(8);
    address_table_var.set_initializer(&address_table);

    let code_hash_array: Vec<IntValue<'a>> = emit_data
        .code
        .code_hash
        .iter()
        .map(|b| i8t.const_int(*b as u64, false))
        .collect();
    let code_hash = i8t.const_array(&code_hash_array);
    let code_hash_type = i8t.array_type(emit_data.code.code_hash.len() as u32);
    let code_hash_var = emit_data.module.add_global(
        code_hash_type,
        None,
        &format!("{}____code_hash____", emit_data.code.symbol_prefix),
    );
    code_hash_var.set_initializer(&code_hash);

    let align_stack_attr_id = Attribute::get_named_enum_kind_id("alignstack");
    if align_stack_attr_id == 0 {
        return Err(Error::External(
            "LLVM is missing alignstack attr!".to_string(),
        ));
    }
    let align_stack_attr = context.create_enum_attribute(align_stack_attr_id, 16);
    emit_data
        .ffi_wrapper_func
        .add_attribute(AttributeLoc::Function, align_stack_attr);

    // Build the actual function bodies
    for func in &emit_data.code.funcs {
        emit_riscv_func(context, emit_data, debug_data, func)?;
    }

    emit_entry(context, emit_data)?;
    emit_exit(context, emit_data)?;
    emit_ffi_wrapper(context, emit_data)?;

    // Finalize all dependencies
    if let Some(debug_data) = debug_data {
        debug_data.di_builder.finalize();
    }

    if optimize {
        emit_data.pass.run_on(&emit_data.module);
    }

    // Verify all functions
    // TODO: Ideally we should use LLVMPrintMessageAction and return an error at
    // Rust side, but in reality LLVM and Rust will be competing for STDOUT when
    // printing errors. Using LLVMAbortProcessAction here at least enables us to
    // see the full generated error message at LLVM side. Later we shall revisit
    // this to see how we can tackle the problem.
    if !emit_data.entry_func.verify(false) {
        return Err(Error::External(
            "Entry function fails verification".to_string(),
        ));
    }
    if !emit_data.exit_func.verify(false) {
        return Err(Error::External(
            "Exit function fails verification".to_string(),
        ));
    }
    if !emit_data.ffi_wrapper_func.verify(false) {
        return Err(Error::External(
            "FFI wrapper function fails verification".to_string(),
        ));
    }
    for func in &emit_data.code.funcs {
        let function = emit_data.emitted_funcs[&func.range.start];
        if !function.verify(false) {
            return Err(Error::External(format!(
                "verifying function {} at 0x{:x}",
                func.force_name(true),
                func.range.start
            )));
        }
    }

    Ok(())
}
