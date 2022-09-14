use super::{
    ast::{register_names, Control, Write},
    macros::*,
    preprocessor::{preprocess, Func},
    runner::{
        LlvmAotCoreMachineData, LlvmAotMachineEnv, EXIT_REASON_BARE_CALL_EXIT,
        EXIT_REASON_CYCLES_OVERFLOW, EXIT_REASON_EBREAK_UNREACHABLE, EXIT_REASON_ECALL_UNREACHABLE,
        EXIT_REASON_MALFORMED_INDIRECT_CALL, EXIT_REASON_MALFORMED_RETURN,
        EXIT_REASON_MAX_CYCLES_EXCEEDED, EXIT_REASON_UNKNOWN_BLOCK, EXIT_REASON_UNKNOWN_PC_VALUE,
        EXIT_REASON_UNKNOWN_RESUME_ADDRESS,
    },
    utils::cs,
};
use ckb_vm::{
    instructions::ast::{ActionOp1, ActionOp2, SignActionOp2, Value},
    machine::InstructionCycleFunc,
    Bytes, Error, Register,
};
use ckb_vm_definitions::registers;
use lazy_static::lazy_static;
use llvm_sys::{
    analysis::*,
    bit_writer::*,
    core::*,
    debuginfo::*,
    prelude::*,
    target::*,
    target_machine::*,
    transforms::{scalar::*, util::*},
    LLVMCallConv, LLVMIntPredicate,
};
use memoffset::offset_of;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::io::Write as StdWrite;
use std::path::Path;
use std::ptr;

pub struct LlvmCompilingMachine {
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    pass: LLVMPassManagerRef,

    di_builder: LLVMDIBuilderRef,
    compile_file: LLVMMetadataRef,
    compile_unit: LLVMMetadataRef,
    debug_file_writer: Option<DebugFileWriter>,

    funcs: Vec<Func>,
    code_hash: [u8; 32],
    symbol_prefix: String,

    // RISC-V function address -> generated LLVM function reference
    emitted_funcs: HashMap<u64, LLVMValueRef>,
    // Entry function to AOT code
    entry_func: LLVMValueRef,
    // RISC-V early exit function
    exit_func: LLVMValueRef,
    // Wrapper to call into Rust FFI functions
    ffi_wrapper_func: LLVMValueRef,
}

extern "C" {
    // LLVMIntrinsicGetType in llvm-sys is missing an argument:
    // https://gitlab.com/taricorp/llvm-sys.rs/-/blob/ffd4a8028b02331a53ef4fdf194d411d5e95819d/src/core.rs#L812
    #[link_name = "LLVMIntrinsicGetType"]
    fn PatchedLLVMIntrinsicGetType(
        Ctx: LLVMContextRef,
        ID: u32,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: usize,
    ) -> LLVMTypeRef;
}

impl LlvmCompilingMachine {
    pub fn initialize() -> Result<(), Error> {
        assert_llvm_call!(LLVM_InitializeNativeTarget(), "init native target");
        u!(LLVM_InitializeNativeAsmPrinter());
        u!(LLVM_InitializeNativeAsmParser());
        assert_eq!(REGISTER_MAPPINGS.len(), 15);
        assert_eq!(REGISTER_MAPPINGS[1], Mapping::Pointer);
        assert!(REGISTER_MAPPINGS.iter().any(|m| match m {
            Mapping::Pc => true,
            _ => false,
        }));
        Ok(())
    }

    // TODO: memory leaks might occur in case of errors. We will need to
    // build wrappers on pointer types or switch to a higher level crate
    // than llvm-sys.
    pub fn load(
        name: &str,
        code: &Bytes,
        symbol_prefix: &str,
        instruction_cycle_func: &InstructionCycleFunc,
        generate_debug_info: bool,
    ) -> Result<Self, Error> {
        let name = Path::new(name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(name);
        let directory = Path::new(name)
            .parent()
            .and_then(|s| s.to_str())
            .unwrap_or(".");

        let funcs = preprocess(code, instruction_cycle_func)?;
        let context = assert_llvm_create!(LLVMContextCreate(), "context");
        let module = assert_llvm_create!(
            LLVMModuleCreateWithNameInContext(
                cs(&format!("{}{}", symbol_prefix, name))?.as_ptr(),
                context
            ),
            "module"
        );
        let builder = assert_llvm_create!(LLVMCreateBuilderInContext(context), "builder");

        let (di_builder, compile_file, compile_unit, debug_file_writer) = if generate_debug_info {
            let debug_file_name = format!("{}.debug.s", name);
            let di_builder = assert_llvm_create!(LLVMCreateDIBuilder(module), "di builder");
            let compile_file = u!(LLVMDIBuilderCreateFile(
                di_builder,
                debug_file_name.as_ptr() as *const _,
                debug_file_name.len(),
                directory.as_ptr() as *const _,
                directory.len(),
            ));
            let producer = "ckb-vm-llvm-aot-engine";
            let compile_unit = assert_llvm_create!(
                LLVMDIBuilderCreateCompileUnit(
                    di_builder,
                    LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC,
                    compile_file,
                    producer.as_ptr() as *const _,
                    producer.len(),
                    0,                         // isOptimized
                    "\0".as_ptr() as *const _, // Flags
                    0,
                    0,                         // RuntimeVer
                    "\0".as_ptr() as *const _, // SplitName
                    0,
                    LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
                    0,                         // DWOId
                    1,                         // SplitDebugInlining
                    0,                         // DebugInfoForProfiling
                    "\0".as_ptr() as *const _, // SysRoot
                    0,
                    "\0".as_ptr() as *const _, // SDK
                    0
                ),
                "compile unit"
            );
            let debug_file_writer = DebugFileWriter::new(
                context,
                compile_unit,
                Path::new(directory)
                    .join(debug_file_name)
                    .to_str()
                    .ok_or(Error::External("invalid file name!".to_string()))?,
            )?;
            (
                di_builder,
                compile_file,
                compile_unit,
                Some(debug_file_writer),
            )
        } else {
            (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), None)
        };

        let pass = assert_llvm_create!(LLVMCreatePassManager(), "create pass manager");
        unsafe {
            LLVMAddPromoteMemoryToRegisterPass(pass);
            LLVMAddInstructionCombiningPass(pass);
            LLVMAddReassociatePass(pass);
            LLVMAddGVNPass(pass);
            LLVMAddCFGSimplificationPass(pass);
        }
        assert_llvm_call!(
            LLVMInitializeFunctionPassManager(pass),
            "initialize pass manager"
        );
        let emitted_funcs = HashMap::with_capacity(funcs.len());
        let code_hash: [u8; 32] = blake3::hash(code).into();

        Ok(Self {
            context,
            builder,
            module,
            pass,
            di_builder,
            compile_file,
            compile_unit,
            debug_file_writer,
            funcs,
            emitted_funcs,
            code_hash,
            symbol_prefix: symbol_prefix.to_string(),
            // Those will be updated in emit function
            entry_func: ptr::null_mut(),
            exit_func: ptr::null_mut(),
            ffi_wrapper_func: ptr::null_mut(),
        })
    }

    pub fn bitcode(mut self, optimize: bool) -> Result<Bytes, Error> {
        self.emit(optimize)?;

        let llvm_buf = assert_llvm_create!(
            LLVMWriteBitcodeToMemoryBuffer(self.module),
            "write to memory buffer"
        );

        llvm_buffer_to_bytes(llvm_buf)
    }

    pub fn aot(mut self, optimize: bool) -> Result<Bytes, Error> {
        self.emit(optimize)?;

        let triple = assert_llvm_create!(LLVMGetDefaultTargetTriple(), "get default triple");
        let mut target = ptr::null_mut();
        assert_llvm_err!(LLVMGetTargetFromTriple, triple, &mut target);
        let tm = assert_llvm_create!(
            LLVMCreateTargetMachine(
                target,
                triple,
                b"generic\0".as_ptr() as *const _,
                b"\0".as_ptr() as *const _,
                LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
                LLVMRelocMode::LLVMRelocPIC,
                LLVMCodeModel::LLVMCodeModelDefault,
            ),
            "create target machine"
        );

        let mut err = ptr::null_mut();
        let mut buf = ptr::null_mut();

        assert_llvm_named_err!(
            LLVMTargetMachineEmitToMemoryBuffer(
                tm,
                self.module,
                LLVMCodeGenFileType::LLVMObjectFile,
                &mut err,
                &mut buf,
            ),
            err
        );

        u!(LLVMDisposeTargetMachine(tm));
        llvm_buffer_to_bytes(buf)
    }

    fn emit(&mut self, optimize: bool) -> Result<(), Error> {
        let inline_asm = self.module_inline_asm();
        u!(LLVMSetModuleInlineAsm2(
            self.module,
            inline_asm.as_ptr() as *const _,
            inline_asm.len()
        ));

        let table_len: u32 = (self.funcs.len() * 2 + 1).try_into().map_err(|_e| {
            Error::External(format!(
                "There are too many functions provided: {}",
                self.funcs.len()
            ))
        })?;

        let i64t = self.i64t()?;
        let i8t = self.i8t()?;

        let address_table_type = u!(LLVMArrayType(i64t, table_len));
        let mut address_table_values = Vec::with_capacity(table_len as usize);

        address_table_values.push(u!(LLVMConstInt(i64t, self.funcs.len() as u64, 0)));
        for func in self.funcs.clone() {
            // Build signature for each function here, so we can piece together calls later
            let function_name = format!("{}{}", self.symbol_prefix, func.force_name(false));
            let function_name_cstr = cs(&function_name)?;
            let function = assert_llvm_create!(
                LLVMAddFunction(
                    self.module,
                    function_name_cstr.as_ptr(),
                    self.riscv_function_type()?
                ),
                "create function"
            );
            u!(LLVMSetFunctionCallConv(
                function,
                LLVMCallConv::LLVMHHVMCallConv as u32
            ));

            self.emitted_funcs.insert(func.range.start, function);
            address_table_values.push(u!(LLVMConstInt(i64t, func.range.start, 0)));
            address_table_values.push(u!(LLVMConstPtrToInt(function, i64t)))
        }

        let address_table = assert_llvm_create!(
            LLVMConstArray(i64t, address_table_values.as_mut_ptr(), table_len),
            "create address table array"
        );

        let address_table_var = assert_llvm_create!(
            LLVMAddGlobal(
                self.module,
                address_table_type,
                cs(&format!("{}____address_table____", self.symbol_prefix))?.as_ptr(),
            ),
            "create address table global var"
        );
        u!(LLVMSetAlignment(address_table_var, 8));
        u!(LLVMSetInitializer(address_table_var, address_table));

        let mut code_hash_array: Vec<LLVMValueRef> = self
            .code_hash
            .iter()
            .map(|b| u!(LLVMConstInt(i8t, *b as u64, 0)))
            .collect();
        let code_hash = assert_llvm_create!(
            LLVMConstArray(
                i8t,
                code_hash_array.as_mut_ptr(),
                code_hash_array.len() as u32
            ),
            "create code hash array"
        );
        let code_hash_type = u!(LLVMArrayType(i8t, self.code_hash.len() as u32));
        let code_hash_var = assert_llvm_create!(
            LLVMAddGlobal(
                self.module,
                code_hash_type,
                cs(&format!("{}____code_hash____", self.symbol_prefix))?.as_ptr(),
            ),
            "create code hash var"
        );
        u!(LLVMSetInitializer(code_hash_var, code_hash));

        self.entry_func = {
            let entry_function_type = {
                // *mut LlvmAotCoreMachineData, pointer for the host function to execute
                let mut args_type = [i64t, i64t];
                assert_llvm_create!(
                    LLVMFunctionType(i8t, args_type.as_mut_ptr(), args_type.len() as u32, 0),
                    "entry function type"
                )
            };
            assert_llvm_create!(
                LLVMAddFunction(
                    self.module,
                    cs(&format!("{}____entry____", self.symbol_prefix))?.as_ptr(),
                    entry_function_type
                ),
                "create entry function"
            )
        };
        self.exit_func = assert_llvm_create!(
            LLVMAddFunction(
                self.module,
                cs(&format!("{}____exit____", self.symbol_prefix))?.as_ptr(),
                self.exit_function_type()?,
            ),
            "create exit function"
        );
        u!(LLVMSetFunctionCallConv(
            self.exit_func,
            LLVMCallConv::LLVMHHVMCallConv as u32
        ));

        self.ffi_wrapper_func = assert_llvm_create!(
            LLVMAddFunction(
                self.module,
                cs(&format!("{}____ffi_wrapper____", self.symbol_prefix))?.as_ptr(),
                self.ffi_wrapper_function_type()?,
            ),
            "create ffi function"
        );
        let align_stack_attr_id = u!(LLVMGetEnumAttributeKindForName(
            b"alignstack\0".as_ptr() as *const _,
            10
        ));
        if align_stack_attr_id == 0 {
            return Err(Error::External(
                "LLVM is missing alignstack attr!".to_string(),
            ));
        }
        let align_stack_attr = assert_llvm_create!(
            LLVMCreateEnumAttribute(self.context, align_stack_attr_id, 16),
            "create alignstack attr"
        );
        u!(LLVMAddAttributeAtIndex(
            self.ffi_wrapper_func,
            u32::max_value(),
            align_stack_attr,
        ));

        // Build the actual function bodies
        for func in self.funcs.clone() {
            self.emit_riscv_func(&func)?;
        }

        self.emit_entry()?;
        self.emit_exit()?;
        self.emit_ffi_wrapper()?;

        // Finalize all dependencies
        if !self.di_builder.is_null() {
            u!(LLVMDIBuilderFinalize(self.di_builder));
        }

        if optimize {
            u!(LLVMRunPassManager(self.pass, self.module));
        }

        // Verify all functions
        // TODO: Ideally we should use LLVMPrintMessageAction and return an error at
        // Rust side, but in reality LLVM and Rust will be competing for STDOUT when
        // printing errors. Using LLVMAbortProcessAction here at least enables us to
        // see the full generated error message at LLVM side. Later we shall revisit
        // this to see how we can tackle the problem.
        assert_llvm_call!(
            LLVMVerifyFunction(
                self.entry_func,
                LLVMVerifierFailureAction::LLVMAbortProcessAction
            ),
            "verifying entry function".to_string()
        );
        assert_llvm_call!(
            LLVMVerifyFunction(
                self.exit_func,
                LLVMVerifierFailureAction::LLVMAbortProcessAction
            ),
            "verifying exit function".to_string()
        );
        assert_llvm_call!(
            LLVMVerifyFunction(
                self.ffi_wrapper_func,
                LLVMVerifierFailureAction::LLVMAbortProcessAction
            ),
            "verifying ffi wrapper function".to_string()
        );
        for func in &self.funcs {
            let function = self.emitted_funcs[&func.range.start];
            assert_llvm_call!(
                LLVMVerifyFunction(function, LLVMVerifierFailureAction::LLVMAbortProcessAction),
                &format!(
                    "verifying function {} at 0x{:x}",
                    func.force_name(true),
                    func.range.start
                )
            );
        }

        Ok(())
    }

    // Emit the entrypoint function to AOT code
    fn emit_entry(&mut self) -> Result<(), Error> {
        let i64t = self.i64t()?;
        let i8t = self.i8t()?;

        let setjmp_id = u!(LLVMLookupIntrinsicID(
            b"llvm.eh.sjlj.setjmp\0".as_ptr() as *const _,
            19
        ));
        if setjmp_id == 0 {
            return Err(Error::External(
                "Missing intrinsics for setjmp!".to_string(),
            ));
        }
        let setjmp_type = assert_llvm_create!(
            PatchedLLVMIntrinsicGetType(self.context, setjmp_id, std::ptr::null_mut(), 0),
            "setjmp type"
        );
        // Refer to the comments on +module_inline_asm+ method for more details
        let setjmp_func = assert_llvm_create!(
            LLVMAddFunction(
                self.module,
                cs(&self.hosted_setjmp_name())?.as_ptr(),
                setjmp_type
            ),
            "create setjmp function"
        );

        let entry_function = self.entry_func;
        let entry_block = assert_llvm_create!(
            LLVMAppendBasicBlockInContext(
                self.context,
                entry_function,
                b"__entry_start_block__\0".as_ptr() as *const _,
            ),
            "create entry basic block"
        );
        u!(LLVMPositionBuilderAtEnd(self.builder, entry_block));
        let machine = u!(LLVMGetParam(entry_function, 0));
        let target = u!(LLVMGetParam(entry_function, 1));

        let call_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                b"__entry_call_block__\0".as_ptr() as *const _,
            ),
            "create call block"
        );
        u!(LLVMAppendExistingBasicBlock(entry_function, call_block));
        let ret_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                b"__entry_ret_block__\0".as_ptr() as *const _,
            ),
            "create ret block"
        );
        u!(LLVMAppendExistingBasicBlock(entry_function, ret_block));

        let mut setjmp_args = [u!(LLVMBuildIntToPtr(
            self.builder,
            LLVMBuildAdd(
                self.builder,
                machine,
                LLVMConstInt(i64t, offset_of!(LlvmAotCoreMachineData, jmpbuf) as u64, 0),
                b"jmpbuf_addr_val\0".as_ptr() as *const _,
            ),
            self.i8pt()?,
            b"jmpbuf_addr\0".as_ptr() as *const _,
        ))];
        let setjmp_result = u!(LLVMBuildCall2(
            self.builder,
            setjmp_type,
            setjmp_func,
            setjmp_args.as_mut_ptr() as *mut _,
            setjmp_args.len() as u32,
            b"setjmp_result\0".as_ptr() as *const _
        ));
        let setjmp_cond = u!(LLVMBuildICmp(
            self.builder,
            LLVMIntPredicate::LLVMIntEQ,
            setjmp_result,
            LLVMConstInt(self.i32t()?, 0, 0),
            b"setjmp_result_cmp_zero\0".as_ptr() as *const _,
        ));
        u!(LLVMBuildCondBr(
            self.builder,
            setjmp_cond,
            call_block,
            ret_block,
        ));

        u!(LLVMPositionBuilderAtEnd(self.builder, call_block));
        let vars = self.emit_setup(machine)?;

        let target_type = self.riscv_function_type()?;
        let target_pt = assert_llvm_create!(LLVMPointerType(target_type, 0), "target type");
        let target_func = u!(LLVMBuildIntToPtr(
            self.builder,
            target,
            target_pt,
            b"target_function_pointer\0".as_ptr() as *const _,
        ));
        let mut invoke_args = vars.to_arguments();
        let target_result = u!(LLVMBuildCall2(
            self.builder,
            target_type,
            target_func,
            invoke_args.as_mut_ptr(),
            invoke_args.len() as u32,
            b"target_function_result\0".as_ptr() as *const _,
        ));
        u!(LLVMSetInstructionCallConv(
            target_result,
            LLVMCallConv::LLVMHHVMCallConv as u32
        ));
        // In most cases, the code will result in the other branch of setjmp above.
        // A direct return here will be very rare, but we still need to perform cleanups.
        let mut target_return_values = [ptr::null_mut(); 14];
        for i in 0..14 {
            target_return_values[i] = u!(LLVMBuildExtractValue(
                self.builder,
                target_result,
                i as u32,
                cs(&format!("target_ret{}", i))?.as_ptr(),
            ));
        }
        let target_return_values =
            TransientValues::from_return_values(target_return_values, machine);
        self.emit_cleanup(&target_return_values)?;
        u!(LLVMBuildBr(self.builder, ret_block));

        u!(LLVMPositionBuilderAtEnd(self.builder, ret_block));
        // At this point, cleanups have been taken care of either by call_block or exit function
        let reason = self.emit_load_from_machine(
            machine,
            offset_of!(LlvmAotCoreMachineData, exit_aot_reason),
            i8t,
            Some("exit_reason"),
        )?;

        u!(LLVMBuildRet(self.builder, reason));

        Ok(())
    }

    // Emit an early exit function from RISC-V, this does stack unwinding via longjmp
    fn emit_exit(&mut self) -> Result<(), Error> {
        let longjmp_id = u!(LLVMLookupIntrinsicID(
            b"llvm.eh.sjlj.longjmp\0".as_ptr() as *const _,
            20
        ));
        if longjmp_id == 0 {
            return Err(Error::External(
                "Missing intrinsics for longjmp!".to_string(),
            ));
        }
        let longjmp_type = assert_llvm_create!(
            PatchedLLVMIntrinsicGetType(self.context, longjmp_id, std::ptr::null_mut(), 0),
            "longjmp type"
        );
        // Refer to the comments on +module_inline_asm+ method for more details
        let longjmp_func = assert_llvm_create!(
            LLVMAddFunction(
                self.module,
                cs(&self.hosted_longjmp_name())?.as_ptr(),
                longjmp_type
            ),
            "create longjmp function"
        );
        let noreturn_attr_id = u!(LLVMGetEnumAttributeKindForName(
            b"noreturn\0".as_ptr() as *const _,
            8
        ));
        if noreturn_attr_id == 0 {
            return Err(Error::External(
                "LLVM is missing noreturn attr!".to_string(),
            ));
        }
        let noreturn_attr = assert_llvm_create!(
            LLVMCreateEnumAttribute(self.context, noreturn_attr_id, 0),
            "create alignstack attr"
        );
        u!(LLVMAddAttributeAtIndex(
            longjmp_func,
            u32::max_value(),
            noreturn_attr,
        ));

        let exit_function = self.exit_func;

        let exit_block = assert_llvm_create!(
            LLVMAppendBasicBlockInContext(
                self.context,
                exit_function,
                b"__exit_start_block__\0".as_ptr() as *const _,
            ),
            "create exit basic block"
        );
        u!(LLVMPositionBuilderAtEnd(self.builder, exit_block));

        let vars = {
            let mut vars = [ptr::null_mut(); 15];
            for i in 0..15 {
                vars[i] = u!(LLVMGetParam(exit_function, i as u32));
            }
            TransientValues::from_arguments(vars)
        };
        let machine = vars.extract_machine()?;

        self.emit_cleanup(&vars)?;

        // exit reason is passed within LlvmAotCoreMachineData

        let mut longjmp_args = [u!(LLVMBuildIntToPtr(
            self.builder,
            LLVMBuildAdd(
                self.builder,
                machine,
                LLVMConstInt(
                    self.i64t()?,
                    offset_of!(LlvmAotCoreMachineData, jmpbuf) as u64,
                    0
                ),
                b"jmpbuf_addr_val\0".as_ptr() as *const _,
            ),
            self.i8pt()?,
            b"jmpbuf_addr\0".as_ptr() as *const _,
        ))];
        u!(LLVMBuildCall2(
            self.builder,
            longjmp_type,
            longjmp_func,
            longjmp_args.as_mut_ptr() as *mut _,
            longjmp_args.len() as u32,
            b"\0".as_ptr() as *const _
        ));
        u!(LLVMBuildUnreachable(self.builder));

        self.exit_func = exit_function;
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
    fn emit_ffi_wrapper(&mut self) -> Result<(), Error> {
        let entry_block = assert_llvm_create!(
            LLVMAppendBasicBlockInContext(
                self.context,
                self.ffi_wrapper_func,
                b"__entry_block__\0".as_ptr() as *const _,
            ),
            "create ffi wrapper entry block"
        );
        u!(LLVMPositionBuilderAtEnd(self.builder, entry_block));

        let func = u!(LLVMGetParam(self.ffi_wrapper_func, 0));
        let env = u!(LLVMGetParam(self.ffi_wrapper_func, 1));
        let arg = u!(LLVMGetParam(self.ffi_wrapper_func, 2));

        let mut args = [env, arg];
        let result = u!(LLVMBuildCall2(
            self.builder,
            self.ffi_function_type()?,
            func,
            args.as_mut_ptr(),
            args.len() as u32,
            b"wrapper_call_result\0".as_ptr() as *const _,
        ));
        u!(LLVMBuildRet(self.builder, result));

        Ok(())
    }

    // Emit a FFI call, also check the return result, when error happens,
    // emit early exit function.
    fn emit_ffi_call(
        &mut self,
        machine: LLVMValueRef,
        function: LLVMValueRef,
        allocas: &RegAllocas,
        args: &mut [LLVMValueRef; 3],
        side_effect: bool,
        name: Option<&str>,
    ) -> Result<LLVMValueRef, Error> {
        let name = name.unwrap_or("ffi_call_result");

        if side_effect {
            self.emit_cleanup(&allocas.load_values()?)?;
        }

        let result = u!(LLVMBuildCall2(
            self.builder,
            self.ffi_wrapper_function_type()?,
            self.ffi_wrapper_func,
            args.as_mut_ptr(),
            args.len() as u32,
            cs(name)?.as_ptr(),
        ));

        if side_effect {
            allocas.store_values(&self.emit_setup(machine)?)?;
        }

        let success_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                cs(&format!("{}_success_block", name))?.as_ptr(),
            ),
            "create success block"
        );
        u!(LLVMAppendExistingBasicBlock(function, success_block));
        let failure_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                cs(&format!("{}_failure_block", name))?.as_ptr(),
            ),
            "create failure block"
        );
        u!(LLVMAppendExistingBasicBlock(function, failure_block));

        let cmp = u!(LLVMBuildICmp(
            self.builder,
            LLVMIntPredicate::LLVMIntNE,
            result,
            LLVMConstInt(self.i64t()?, 0, 0),
            cs(&format!("{}_ne_zero", name))?.as_ptr(),
        ));
        u!(LLVMBuildCondBr(
            self.builder,
            cmp,
            success_block,
            failure_block,
        ));
        u!(LLVMPositionBuilderAtEnd(self.builder, failure_block));
        self.emit_call_exit(machine, EXIT_REASON_BARE_CALL_EXIT, &allocas)?;

        u!(LLVMPositionBuilderAtEnd(self.builder, success_block));

        Ok(result)
    }

    // Emit a RISC-V function
    fn emit_riscv_func(&mut self, func: &Func) -> Result<LLVMValueRef, Error> {
        if func.basic_blocks.is_empty() {
            return Err(Error::External(format!(
                "Func {} at 0x{:x} does not have basic blocks!",
                func.force_name(true),
                func.range.start
            )));
        }
        let i64t = self.i64t()?;

        let function = self.emitted_funcs[&func.range.start];

        if let Some(debug_file_writer) = &mut self.debug_file_writer {
            let di_function_type = {
                let di_i64t = assert_llvm_create!(
                    LLVMDIBuilderCreateBasicType(
                        self.di_builder,
                        b"uint64\0".as_ptr() as *const _,
                        5,
                        64,
                        0x07, // DW_ATE_unsigned
                        0
                    ),
                    "create basic type"
                );

                let mut argts = [di_i64t; 16];
                let return_type = assert_llvm_create!(
                    LLVMDIBuilderCreateStructType(
                        self.di_builder,
                        self.compile_unit,
                        b"riscv_return_struct\0".as_ptr() as *const _,
                        19,
                        self.compile_file,
                        0, // LineNumber
                        14 * 64,
                        0,
                        0, // Flags
                        ptr::null_mut(),
                        argts.as_mut_ptr(),
                        14,
                        0,
                        ptr::null_mut(),
                        "\0".as_ptr() as *const _,
                        0
                    ),
                    "create di return struct type"
                );
                argts[0] = return_type;

                assert_llvm_create!(
                    LLVMDIBuilderCreateSubroutineType(
                        self.di_builder,
                        self.compile_file,
                        argts.as_mut_ptr(),
                        16,
                        0
                    ),
                    "create di subroutine type"
                )
            };

            let function_name = format!("{}{}", self.symbol_prefix, func.force_name(false));
            let di_function = assert_llvm_create!(
                LLVMDIBuilderCreateFunction(
                    self.di_builder,
                    self.compile_file,
                    function_name.as_ptr() as *const _,
                    function_name.len(),
                    b"\0".as_ptr() as *const _, // LinkageName,
                    0,
                    self.compile_file,
                    0,
                    di_function_type, // Ty
                    0,                // IsLocalToUnit
                    1,                // IsDefinition
                    0,
                    0,
                    1,
                ),
                "create di function"
            );
            u!(LLVMSetSubprogram(function, di_function));
            debug_file_writer.set_scope(di_function);
            debug_file_writer.write(&format!(
                "<{} (0x{:x})>:",
                func.force_name(true),
                func.range.start,
            ))?;
            u!(LLVMSetCurrentDebugLocation2(
                self.builder,
                debug_file_writer.debug_location()?,
            ));
        }

        // This is a dummy entry block in case some jumps point to the start of
        // the function. LLVM does not allow predecessors for entry block.
        let entry_block = assert_llvm_create!(
            LLVMAppendBasicBlockInContext(
                self.context,
                function,
                cs(&format!("basic_block_entry_{}", func.force_name(true)))?.as_ptr(),
            ),
            "create basic block"
        );

        let basic_blocks: HashMap<u64, LLVMBasicBlockRef> = {
            let mut bbs = HashMap::default();

            for block in &func.basic_blocks {
                let b = assert_llvm_create!(
                    LLVMCreateBasicBlockInContext(
                        self.context,
                        self.basic_block_name(block.range.start)?.as_ptr() as *const _
                    ),
                    "create basic block"
                );
                u!(LLVMAppendExistingBasicBlock(function, b));

                bbs.insert(block.range.start, b);
            }

            bbs
        };
        u!(LLVMPositionBuilderAtEnd(self.builder, entry_block));

        // Fetch arguments
        let args = {
            let mut args = [ptr::null_mut(); 15];
            for i in 0..15 {
                args[i] = u!(LLVMGetParam(function, i as u32));
            }
            TransientValues::from_arguments(args)
        };
        let machine = args.extract_machine()?;

        // Build allocas for arguments
        let vars = {
            let entry_block =
                assert_llvm_create!(LLVMGetEntryBasicBlock(function), "get entry block");
            u!(LLVMPositionBuilderAtEnd(self.builder, entry_block));

            let alloca_args = args.map(|arg, mapping| {
                let var = u!(LLVMBuildAlloca(
                    self.builder,
                    i64t,
                    cs(&format!("alloc_{}", mapping))?.as_ptr()
                ));
                u!(LLVMBuildStore(self.builder, arg, var));
                Ok(var)
            })?;

            RegAllocas::new(alloca_args, self.builder, self.i64t()?)
        };
        let pc_alloca = vars.pc_alloca()?;

        // Build one memory_start construct per function
        let memory_start = self.emit_load_from_machine(
            machine,
            offset_of!(LlvmAotCoreMachineData, memory),
            i64t,
            Some("memory_start"),
        )?;

        let indirect_dispatch_test_alloca = u!(LLVMBuildAlloca(
            self.builder,
            i64t,
            b"indirect_dispatch_test_alloca\0".as_ptr() as *const _,
        ));

        // Jump to the first actual basic block
        u!(LLVMBuildBr(self.builder, basic_blocks[&func.range.start]));

        let mut control_blocks: HashMap<u64, LLVMBasicBlockRef> = HashMap::default();
        // Emit code for each basic block
        for block in &func.basic_blocks {
            u!(LLVMPositionBuilderAtEnd(
                self.builder,
                basic_blocks[&block.range.start]
            ));

            // Emit cycle calculation logic
            let current_cycles = self.emit_load_from_machine(
                machine,
                offset_of!(LlvmAotCoreMachineData, cycles),
                i64t,
                Some("current_cycles"),
            )?;
            let max_cycles = self.emit_load_from_machine(
                machine,
                offset_of!(LlvmAotCoreMachineData, max_cycles),
                i64t,
                Some("max_cycles"),
            )?;
            let updated_cycles = u!(LLVMBuildAdd(
                self.builder,
                current_cycles,
                LLVMConstInt(i64t, block.cycles, 0),
                b"updated_cycles\0".as_ptr() as *const _,
            ));
            let cycle_overflow_cmp = u!(LLVMBuildICmp(
                self.builder,
                LLVMIntPredicate::LLVMIntULT,
                updated_cycles,
                current_cycles,
                b"cycle_overflow_cmp\0".as_ptr() as *const _,
            ));
            let cycle_exceeded_cmp = u!(LLVMBuildICmp(
                self.builder,
                LLVMIntPredicate::LLVMIntUGT,
                updated_cycles,
                max_cycles,
                b"cycle_exceeded_cmp\0".as_ptr() as *const _,
            ));
            let cycle_overflow_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    cs(&format!("cycle_overflow_block_0x{:x}", block.range.start))?.as_ptr(),
                ),
                "create cycle overflow block"
            );
            u!(LLVMAppendExistingBasicBlock(function, cycle_overflow_block));
            let cycle_no_overflow_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    cs(&format!(
                        "cycle_no_overflow_block_0x{:x}",
                        block.range.start
                    ))?
                    .as_ptr(),
                ),
                "create cycle no overflow block"
            );
            u!(LLVMAppendExistingBasicBlock(
                function,
                cycle_no_overflow_block
            ));
            let cycle_exceeded_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    cs(&format!("cycle_exceeded_block_0x{:x}", block.range.start))?.as_ptr(),
                ),
                "create cycle exceeded block"
            );
            u!(LLVMAppendExistingBasicBlock(function, cycle_exceeded_block));
            let cycle_ok_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    cs(&format!("cycle_ok_block_0x{:x}", block.range.start))?.as_ptr(),
                ),
                "create cycle ok block"
            );
            u!(LLVMAppendExistingBasicBlock(function, cycle_ok_block));

            u!(LLVMBuildCondBr(
                self.builder,
                cycle_overflow_cmp,
                cycle_overflow_block,
                cycle_no_overflow_block,
            ));
            u!(LLVMPositionBuilderAtEnd(self.builder, cycle_overflow_block));
            self.emit_call_exit(machine, EXIT_REASON_CYCLES_OVERFLOW, &vars)?;
            u!(LLVMPositionBuilderAtEnd(
                self.builder,
                cycle_no_overflow_block
            ));
            u!(LLVMBuildCondBr(
                self.builder,
                cycle_exceeded_cmp,
                cycle_exceeded_block,
                cycle_ok_block,
            ));
            u!(LLVMPositionBuilderAtEnd(self.builder, cycle_exceeded_block));
            self.emit_call_exit(machine, EXIT_REASON_MAX_CYCLES_EXCEEDED, &vars)?;
            u!(LLVMPositionBuilderAtEnd(self.builder, cycle_ok_block));
            self.emit_store_to_machine(
                machine,
                updated_cycles,
                offset_of!(LlvmAotCoreMachineData, cycles),
                i64t,
                Some("updated_cycles"),
            )?;

            // Emit normal register writes, memory writes
            for (i, write_batch) in block.write_batches.iter().enumerate() {
                if let Some(debug_file_writer) = &mut self.debug_file_writer {
                    if let Some(debug_line) = block.debug_lines.get(i) {
                        debug_file_writer.write(&format!("  {}", debug_line))?;
                        u!(LLVMSetCurrentDebugLocation2(
                            self.builder,
                            debug_file_writer.debug_location()?,
                        ));
                    }
                }
                let prefix = format!("batch{}", i);
                self.emit_writes(
                    machine,
                    memory_start,
                    function,
                    &vars,
                    &write_batch,
                    &prefix,
                )?;
            }

            // Update PC & writes together with PC
            if let Some(debug_file_writer) = &mut self.debug_file_writer {
                if let Some(debug_line) = block.debug_lines.get(block.write_batches.len()) {
                    debug_file_writer.write(&format!("  {}", debug_line))?;
                    u!(LLVMSetCurrentDebugLocation2(
                        self.builder,
                        debug_file_writer.debug_location()?,
                    ));
                }
            }
            let next_pc = self.emit_value(
                machine,
                memory_start,
                function,
                &vars,
                &block.control.pc(),
                Some("pc"),
            )?;
            if let Some(last_writes) = block.control.writes() {
                self.emit_writes(
                    machine,
                    memory_start,
                    function,
                    &vars,
                    last_writes,
                    "last_write",
                )?;
            }
            u!(LLVMBuildStore(self.builder, next_pc, pc_alloca));

            let control_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    cs(&format!("control_block_0x{:x}", block.range.start))?.as_ptr() as *const _
                ),
                "create control basic block"
            );
            u!(LLVMAppendExistingBasicBlock(function, control_block));

            u!(LLVMBuildBr(self.builder, control_block));

            control_blocks.insert(block.range.start, control_block);
        }

        let mut indirect_dispatch_block = None;
        // Emit code for each control block
        for block in &func.basic_blocks {
            u!(LLVMPositionBuilderAtEnd(
                self.builder,
                control_blocks[&block.range.start]
            ));

            let next_pc = u!(LLVMBuildLoad2(
                self.builder,
                i64t,
                pc_alloca,
                b"target_pc\0".as_ptr() as *const _,
            ));

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
                        if let Some(target_block) = basic_blocks.get(i) {
                            // 1(a)
                            u!(LLVMBuildBr(self.builder, *target_block));
                        } else {
                            self.emit_call_exit(machine, EXIT_REASON_UNKNOWN_BLOCK, &vars)?;
                        }
                        terminated = true;
                    }
                    Value::Cond(c, t, f) => {
                        if let (Value::Imm(t), Value::Imm(f)) = (&**t, &**f) {
                            if let (Some(true_block), Some(false_block)) =
                                (basic_blocks.get(t), basic_blocks.get(f))
                            {
                                // 1(b)
                                let c = self.emit_value(
                                    machine,
                                    memory_start,
                                    function,
                                    &vars,
                                    c,
                                    Some("pc_cond"),
                                )?;
                                let c = u!(LLVMBuildICmp(
                                    self.builder,
                                    LLVMIntPredicate::LLVMIntEQ,
                                    c,
                                    LLVMConstInt(i64t, 1, 0),
                                    b"pc_cond_i1\0".as_ptr() as *const _,
                                ));
                                u!(LLVMBuildCondBr(self.builder, c, *true_block, *false_block));
                                terminated = true;
                            }
                        }
                    }
                    _ => {
                        // Indirect jump here. Use interpreter to execute till next basic
                        // block end.
                        if indirect_dispatch_block.is_none() {
                            indirect_dispatch_block = Some(self.emit_indirect_dispatch_block(
                                machine,
                                function,
                                &vars,
                                indirect_dispatch_test_alloca,
                                &basic_blocks,
                            )?);
                        }
                        let (interpret_function, data) = self.emit_env_ffi_function(
                            machine,
                            offset_of!(LlvmAotMachineEnv, interpret),
                        )?;
                        let mut interpret_args =
                            [interpret_function, data, u!(LLVMConstInt(i64t, 0, 0))];
                        let interpret_result = self.emit_ffi_call(
                            machine,
                            function,
                            &vars,
                            &mut interpret_args,
                            true,
                            Some("interpret_function_result"),
                        )?;

                        let ret_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                b"indirect_ret_block\0".as_ptr() as *const _,
                            ),
                            "create ret block"
                        );
                        u!(LLVMAppendExistingBasicBlock(function, ret_block));
                        let jump_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                b"indirect_jump_block\0".as_ptr() as *const _,
                            ),
                            "create jump block"
                        );
                        u!(LLVMAppendExistingBasicBlock(function, jump_block));

                        let last_ra_val = self.emit_load_from_machine(
                            machine,
                            offset_of!(LlvmAotCoreMachineData, last_ra),
                            i64t,
                            Some("last_ra"),
                        )?;
                        let cmp = u!(LLVMBuildICmp(
                            self.builder,
                            LLVMIntPredicate::LLVMIntEQ,
                            interpret_result,
                            last_ra_val,
                            b"pc_cmp_last_ra\0".as_ptr() as *const _,
                        ));
                        u!(LLVMBuildCondBr(self.builder, cmp, ret_block, jump_block));

                        u!(LLVMPositionBuilderAtEnd(self.builder, ret_block));
                        self.emit_riscv_return(&vars)?;

                        u!(LLVMPositionBuilderAtEnd(self.builder, jump_block));
                        u!(LLVMBuildStore(
                            self.builder,
                            interpret_result,
                            indirect_dispatch_test_alloca,
                        ));
                        u!(LLVMBuildBr(self.builder, indirect_dispatch_block.unwrap()));
                        terminated = true;
                    }
                },
                Control::Call { address, .. } => {
                    // 2
                    self.emit_call_riscv_func(machine, &vars, *address)?;
                    if let Some(resume_address) = block.control.call_resume_address() {
                        // When returned from the call, update PC using resume address
                        u!(LLVMBuildStore(
                            self.builder,
                            LLVMConstInt(i64t, resume_address, 0),
                            pc_alloca
                        ));
                        if let Some(resume_block) = basic_blocks.get(&resume_address) {
                            u!(LLVMBuildBr(self.builder, *resume_block));
                        } else {
                            self.emit_call_exit(
                                machine,
                                EXIT_REASON_UNKNOWN_RESUME_ADDRESS,
                                &vars,
                            )?;
                        }
                        terminated = true;
                    }
                }
                Control::IndirectCall { .. } => {
                    // 2
                    // First, query the function to call via LlvmAotMachineEnv
                    let (query_function, data) = self.emit_env_ffi_function(
                        machine,
                        offset_of!(LlvmAotMachineEnv, query_function),
                    )?;
                    let mut query_args = [query_function, data, next_pc];
                    let query_result = self.emit_ffi_call(
                        machine,
                        function,
                        &vars,
                        &mut query_args,
                        false,
                        Some("query_function_result"),
                    )?;
                    // There might be 3 cases here:
                    // * query_result is 0: errors happen at Rust side, this case
                    // is handled within +emit_ffi_call+
                    // * query_result is +u64::max_value()+, meaning Rust side failed
                    // to find a native function. There might still be a case we want
                    // to handle: loop unrolling generates from a function within current
                    // function. See +memset+ from +newlib+ for an example here. In
                    // this case, we first test if +next_pc+ lies within current function,
                    // if so, we will handle it like an indirect jump.
                    // * Any other value will be treated like a proper x64 function to
                    // call.
                    let normal_call_block = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"normal_call_block\0".as_ptr() as *const _,
                        ),
                        "create normal call block"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, normal_call_block));
                    let interpret_block = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"interpret_block\0".as_ptr() as *const _,
                        ),
                        "create interpret block"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, interpret_block));
                    let failure_block = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"failure_block\0".as_ptr() as *const _,
                        ),
                        "create failure block"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, failure_block));

                    let cmp = u!(LLVMBuildICmp(
                        self.builder,
                        LLVMIntPredicate::LLVMIntNE,
                        query_result,
                        LLVMConstInt(i64t, u64::max_value(), 0),
                        b"cmp_query_result_to_zero\0".as_ptr() as *const _,
                    ));
                    u!(LLVMBuildCondBr(
                        self.builder,
                        cmp,
                        normal_call_block,
                        interpret_block,
                    ));

                    u!(LLVMPositionBuilderAtEnd(self.builder, failure_block));
                    self.emit_call_exit(machine, EXIT_REASON_MALFORMED_INDIRECT_CALL, &vars)?;

                    u!(LLVMPositionBuilderAtEnd(self.builder, normal_call_block));
                    let query_result_function = u!(LLVMBuildIntToPtr(
                        self.builder,
                        query_result,
                        LLVMPointerType(self.riscv_function_type()?, 0),
                        b"query_function_result_function\0".as_ptr() as *const _,
                    ));
                    // When a proper function is returned(non-zero), use the function
                    // to build the actual RISC-V call.
                    self.emit_call_riscv_func_with_func_value(
                        machine,
                        &vars,
                        query_result_function,
                    )?;
                    if let Some(resume_address) = block.control.call_resume_address() {
                        // When returned from the call, update PC using resume address
                        u!(LLVMBuildStore(
                            self.builder,
                            LLVMConstInt(i64t, resume_address, 0),
                            pc_alloca
                        ));
                        if let Some(resume_block) = basic_blocks.get(&resume_address) {
                            u!(LLVMBuildBr(self.builder, *resume_block));
                        } else {
                            self.emit_call_exit(
                                machine,
                                EXIT_REASON_UNKNOWN_RESUME_ADDRESS,
                                &vars,
                            )?;
                        }
                    } else {
                        return Err(Error::External(format!(
                            "Invalid resume address: {}",
                            block.control
                        )));
                    }

                    u!(LLVMPositionBuilderAtEnd(self.builder, interpret_block));
                    // Ensure next_pc is within current function
                    // TODO: we could also expand this to interpret a whole function
                    // at a different place.
                    let cmp_left = u!(LLVMBuildICmp(
                        self.builder,
                        LLVMIntPredicate::LLVMIntUGE,
                        next_pc,
                        LLVMConstInt(i64t, func.range.start, 0),
                        b"cmp_next_pc_to_func_start\0".as_ptr() as *const _,
                    ));
                    let cmp_right = u!(LLVMBuildICmp(
                        self.builder,
                        LLVMIntPredicate::LLVMIntULT,
                        next_pc,
                        LLVMConstInt(i64t, func.range.end, 0),
                        b"cmp_next_pc_to_func_end\0".as_ptr() as *const _,
                    ));

                    let interpret_block2 = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"interpret_block2\0".as_ptr() as *const _,
                        ),
                        "create interpret block2"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, interpret_block2));
                    let interpret_block3 = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"interpret_block3\0".as_ptr() as *const _,
                        ),
                        "create interpret block3"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, interpret_block3));

                    u!(LLVMBuildCondBr(
                        self.builder,
                        cmp_left,
                        interpret_block2,
                        failure_block
                    ));
                    u!(LLVMPositionBuilderAtEnd(self.builder, interpret_block2));
                    u!(LLVMBuildCondBr(
                        self.builder,
                        cmp_right,
                        interpret_block3,
                        failure_block
                    ));
                    u!(LLVMPositionBuilderAtEnd(self.builder, interpret_block3));

                    // Interpret to the next basic block end
                    if indirect_dispatch_block.is_none() {
                        indirect_dispatch_block = Some(self.emit_indirect_dispatch_block(
                            machine,
                            function,
                            &vars,
                            indirect_dispatch_test_alloca,
                            &basic_blocks,
                        )?);
                    }
                    let (interpret_function, data) = self
                        .emit_env_ffi_function(machine, offset_of!(LlvmAotMachineEnv, interpret))?;
                    let mut interpret_args =
                        [interpret_function, data, u!(LLVMConstInt(i64t, 0, 0))];
                    let interpret_result = self.emit_ffi_call(
                        machine,
                        function,
                        &vars,
                        &mut interpret_args,
                        true,
                        Some("interpret_function_result"),
                    )?;
                    // Dispatch to the correct basic block
                    u!(LLVMBuildStore(
                        self.builder,
                        interpret_result,
                        indirect_dispatch_test_alloca,
                    ));
                    u!(LLVMBuildBr(self.builder, indirect_dispatch_block.unwrap()));

                    terminated = true;
                }
                Control::Tailcall { address, .. } => {
                    // 2
                    let func_llvm_value = *(self.emitted_funcs.get(&address).ok_or_else(|| {
                        Error::External(format!("Function at 0x{:x} does not exist!", address))
                    })?);

                    let values = vars.load_values()?;
                    let mut invoke_args = values.to_arguments();
                    let result = u!(LLVMBuildCall2(
                        self.builder,
                        self.riscv_function_type()?,
                        func_llvm_value,
                        invoke_args.as_mut_ptr(),
                        invoke_args.len() as u32,
                        b"riscv_tailcall_result\0".as_ptr() as *const _
                    ));
                    u!(LLVMSetInstructionCallConv(
                        result,
                        LLVMCallConv::LLVMHHVMCallConv as u32
                    ));
                    u!(LLVMSetTailCall(result, 1));
                    u!(LLVMBuildRet(self.builder, result));
                    terminated = true;
                }
                Control::Return { .. } => {
                    // 3
                    let last_ra_val = self.emit_load_from_machine(
                        machine,
                        offset_of!(LlvmAotCoreMachineData, last_ra),
                        i64t,
                        Some("last_ra"),
                    )?;
                    let cmp = u!(LLVMBuildICmp(
                        self.builder,
                        LLVMIntPredicate::LLVMIntEQ,
                        next_pc,
                        last_ra_val,
                        b"ra_cmp_last_ra\0".as_ptr() as *const _,
                    ));

                    let ret_block = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"ret_block\0".as_ptr() as *const _,
                        ),
                        "create ret block"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, ret_block));
                    let exit_block = assert_llvm_create!(
                        LLVMCreateBasicBlockInContext(
                            self.context,
                            b"exit_block\0".as_ptr() as *const _,
                        ),
                        "create exit block"
                    );
                    u!(LLVMAppendExistingBasicBlock(function, exit_block));

                    u!(LLVMBuildCondBr(self.builder, cmp, ret_block, exit_block));

                    u!(LLVMPositionBuilderAtEnd(self.builder, ret_block));
                    self.emit_riscv_return(&vars)?;

                    u!(LLVMPositionBuilderAtEnd(self.builder, exit_block));
                    self.emit_call_exit(machine, EXIT_REASON_MALFORMED_RETURN, &vars)?;

                    terminated = true;
                }
                Control::Ecall { .. } => {
                    // 4
                    let (ecall_function, data) =
                        self.emit_env_ffi_function(machine, offset_of!(LlvmAotMachineEnv, ecall))?;
                    let mut ecall_args = [ecall_function, data, u!(LLVMConstInt(i64t, 0, 0))];
                    self.emit_ffi_call(machine, function, &vars, &mut ecall_args, true, None)?;
                    if let Some(target_block) = basic_blocks.get(&block.range.end) {
                        u!(LLVMBuildBr(self.builder, *target_block));
                    } else {
                        self.emit_call_exit(machine, EXIT_REASON_ECALL_UNREACHABLE, &vars)?;
                    }
                    terminated = true;
                }
                Control::Ebreak { .. } => {
                    // 5
                    let (ebreak_function, data) =
                        self.emit_env_ffi_function(machine, offset_of!(LlvmAotMachineEnv, ebreak))?;
                    let mut ebreak_args = [ebreak_function, data, u!(LLVMConstInt(i64t, 0, 0))];
                    self.emit_ffi_call(machine, function, &vars, &mut ebreak_args, true, None)?;
                    if let Some(target_block) = basic_blocks.get(&block.range.end) {
                        u!(LLVMBuildBr(self.builder, *target_block));
                    } else {
                        self.emit_call_exit(machine, EXIT_REASON_EBREAK_UNREACHABLE, &vars)?;
                    }
                    terminated = true;
                }
            }

            if !terminated {
                return Err(Error::External(format!(
                    "Unexpected control structure: {:?}",
                    block.control
                )));
            }
        }

        Ok(function)
    }

    // Emit code to restore selected register values back to LlvmAotCoreMachineData
    fn emit_cleanup(&mut self, values: &TransientValues) -> Result<(), Error> {
        let machine = values.extract_machine()?;
        let i64t = self.i64t()?;
        for (value, mapping) in values.iter() {
            match mapping {
                Mapping::Pointer => (),
                Mapping::Pc => {
                    let offset = offset_of!(LlvmAotCoreMachineData, pc);
                    self.emit_store_to_machine(machine, value, offset, i64t, Some("pc"))?;
                }
                Mapping::Register(r) => {
                    let offset = offset_of!(LlvmAotCoreMachineData, registers) + r * 8;
                    self.emit_store_to_machine(
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
    fn emit_setup(&mut self, machine: LLVMValueRef) -> Result<TransientValues, Error> {
        let i64t = self.i64t()?;
        TransientValues::new(|mapping| match mapping {
            Mapping::Pointer => Ok(machine),
            Mapping::Pc => {
                let offset = offset_of!(LlvmAotCoreMachineData, pc);
                self.emit_load_from_machine(machine, offset, i64t, Some("pc"))
            }
            Mapping::Register(r) => {
                let offset = offset_of!(LlvmAotCoreMachineData, registers) + r * 8;
                self.emit_load_from_machine(machine, offset, i64t, Some(&format!("{}", mapping)))
            }
        })
    }

    // Emit return statement for a RISC-V function
    fn emit_riscv_return(&mut self, allocas: &RegAllocas) -> Result<(), Error> {
        let ret_value = allocas.load_values()?;

        let mut ret_args = ret_value.to_return_values();
        u!(LLVMBuildAggregateRet(
            self.builder,
            ret_args.as_mut_ptr(),
            ret_args.len() as u32,
        ));

        Ok(())
    }

    // Emit a series of writes atomically
    fn emit_writes(
        &mut self,
        machine: LLVMValueRef,
        memory_start: LLVMValueRef,
        function: LLVMValueRef,
        allocas: &RegAllocas,
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
                    let address = self.emit_value(
                        machine,
                        memory_start,
                        function,
                        allocas,
                        address,
                        Some(&format!("{}_address", prefix)),
                    )?;
                    let value = self.emit_value(
                        machine,
                        memory_start,
                        function,
                        allocas,
                        value,
                        Some(&format!("{}_value", prefix)),
                    )?;
                    let t = self.size_to_type(*size)?;
                    let casted_value = u!(LLVMBuildIntCast2(
                        self.builder,
                        value,
                        t,
                        0,
                        cs(&format!("{}_casted_value", prefix))?.as_ptr()
                    ));

                    memory_ops.push((address, casted_value, t, prefix));
                }
                Write::Register { index, value } => {
                    let value = self.emit_value(
                        machine,
                        memory_start,
                        function,
                        allocas,
                        value,
                        Some(&format!("{}_value", prefix)),
                    )?;

                    register_ops.push((index, value));
                }
            }
        }

        for (address, value, t, prefix) in memory_ops {
            let real_address_value = u!(LLVMBuildAdd(
                self.builder,
                memory_start,
                address,
                cs(&format!("{}_real_addr_val", prefix))?.as_ptr(),
            ));
            let real_address = u!(LLVMBuildIntToPtr(
                self.builder,
                real_address_value,
                LLVMPointerType(t, 0),
                cs(&format!("{}_real_addr", prefix))?.as_ptr(),
            ));
            u!(LLVMBuildStore(self.builder, value, real_address));
        }
        for (index, value) in register_ops {
            self.emit_store_reg(machine, allocas, *index, value)?;
        }
        Ok(())
    }

    // Emit a CKB-VM AST value via LLVM
    fn emit_value(
        &mut self,
        machine: LLVMValueRef,
        memory_start: LLVMValueRef,
        function: LLVMValueRef,
        allocas: &RegAllocas,
        value: &Value,
        name: Option<&str>,
    ) -> Result<LLVMValueRef, Error> {
        let force_name = name.unwrap_or("uKNOWn_value");
        let i64t = self.i64t()?;

        match value {
            Value::Imm(i) => Ok(u!(LLVMConstInt(i64t, *i, 0))),
            Value::Register(r) => self.emit_load_reg(machine, allocas, *r, name),
            Value::Op1(op, val) => {
                let val = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*val,
                    Some(&format!("{}_val", force_name)),
                )?;
                Ok(match op {
                    ActionOp1::Not => u!(LLVMBuildNot(
                        self.builder,
                        val,
                        cs(&format!("{}_nottemp", force_name))?.as_ptr(),
                    )),
                    ActionOp1::LogicalNot => u!(LLVMBuildAnd(
                        self.builder,
                        LLVMConstInt(i64t, 1, 0),
                        LLVMBuildNot(
                            self.builder,
                            val,
                            cs(&format!("{}_nottemp", force_name))?.as_ptr(),
                        ),
                        cs(&format!("{}_logicalnottemp", force_name))?.as_ptr(),
                    )),
                    ActionOp1::Clz => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.ctlz.*\0".as_ptr() as *const _,
                            11
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External("Missing intrinsic for ctlz!".to_string()));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len(),
                            ),
                            "get ctlz intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get ctlz intrinsic type"
                        );

                        let mut args = [val, u!(LLVMConstInt(self.i1t()?, 0, 0))];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_clz", force_name))?.as_ptr(),
                        ))
                    }
                    ActionOp1::Ctz => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.cttz.*\0".as_ptr() as *const _,
                            11
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External("Missing intrinsic for cttz!".to_string()));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len(),
                            ),
                            "get cttz intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get cttz intrinsic type"
                        );

                        let mut args = [val, u!(LLVMConstInt(self.i1t()?, 0, 0))];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_ctz", force_name))?.as_ptr(),
                        ))
                    }
                    ActionOp1::Cpop => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.ctpop.*\0".as_ptr() as *const _,
                            12
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External(
                                "Missing intrinsic for ctpop!".to_string(),
                            ));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len(),
                            ),
                            "get ctpop intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get ctpop intrinsic type"
                        );

                        let mut args = [val];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_cpop", force_name))?.as_ptr(),
                        ))
                    }
                    ActionOp1::Orcb => {
                        let mut result = u!(LLVMConstInt(i64t, 0, 0));
                        for i in 0..(Value::BITS / 8) {
                            let mask = 0xFFu64 << (i * 8);
                            let mask = u!(LLVMConstInt(i64t, mask, 0));

                            let masked_value = u!(LLVMBuildAnd(
                                self.builder,
                                val,
                                mask,
                                cs(&format!("{}_orcb_masked{}", force_name, i))?.as_ptr(),
                            ));
                            let cmp_value = u!(LLVMBuildICmp(
                                self.builder,
                                LLVMIntPredicate::LLVMIntNE,
                                masked_value,
                                LLVMConstInt(i64t, 0, 0),
                                cs(&format!("{}_orcb_cmp{}", force_name, i))?.as_ptr(),
                            ));
                            let or_value = u!(LLVMBuildSelect(
                                self.builder,
                                cmp_value,
                                mask,
                                LLVMConstInt(i64t, 0, 0),
                                cs(&format!("{}_orcb_or{}", force_name, i))?.as_ptr(),
                            ));
                            result = u!(LLVMBuildOr(
                                self.builder,
                                result,
                                or_value,
                                cs(&format!("{}_orcb_round{}", force_name, i))?.as_ptr(),
                            ));
                        }
                        result
                    }
                    ActionOp1::Rev8 => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.bswap.*\0".as_ptr() as *const _,
                            12
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External(
                                "Missing intrinsic for bswap!".to_string(),
                            ));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len(),
                            ),
                            "get bswap intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get bswap intrinsic type"
                        );

                        let mut args = [val];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_rev8", force_name))?.as_ptr(),
                        ))
                    }
                })
            }
            Value::Op2(op, lhs, rhs) => {
                let lhs = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*lhs,
                    Some(&format!("{}_lhs", force_name)),
                )?;
                let rhs = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*rhs,
                    Some(&format!("{}_rhs", force_name)),
                )?;
                Ok(match op {
                    ActionOp2::Add => u!(LLVMBuildAdd(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_add", force_name))?.as_ptr()
                    )),
                    ActionOp2::Sub => u!(LLVMBuildSub(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_sub", force_name))?.as_ptr()
                    )),
                    ActionOp2::Mul => u!(LLVMBuildMul(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_mul", force_name))?.as_ptr()
                    )),
                    ActionOp2::Bitand => u!(LLVMBuildAnd(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_and", force_name))?.as_ptr()
                    )),
                    ActionOp2::Bitor => u!(LLVMBuildOr(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_or", force_name))?.as_ptr()
                    )),
                    ActionOp2::Bitxor => u!(LLVMBuildXor(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_xor", force_name))?.as_ptr()
                    )),
                    ActionOp2::Shl => u!(LLVMBuildShl(
                        self.builder,
                        lhs,
                        rhs,
                        cs(&format!("{}_shl", force_name))?.as_ptr()
                    )),
                    ActionOp2::Eq => {
                        let v = u!(LLVMBuildICmp(
                            self.builder,
                            LLVMIntPredicate::LLVMIntEQ,
                            lhs,
                            rhs,
                            cs(&format!("{}_eq_i1", force_name))?.as_ptr()
                        ));
                        u!(LLVMBuildSelect(
                            self.builder,
                            v,
                            LLVMConstInt(i64t, 1, 0),
                            LLVMConstInt(i64t, 0, 0),
                            cs(&format!("{}_eq", force_name))?.as_ptr()
                        ))
                    }
                    ActionOp2::Mulhsu => {
                        let t = self.machine_word_type()?;
                        let widening_t = self.double_machine_word_type()?;

                        let widening_lhs = u!(LLVMBuildIntCast2(
                            self.builder,
                            lhs,
                            widening_t,
                            1,
                            cs(&format!("{}_widening_lhs", force_name))?.as_ptr(),
                        ));
                        let widening_rhs = u!(LLVMBuildIntCast2(
                            self.builder,
                            rhs,
                            widening_t,
                            0,
                            cs(&format!("{}_widening_rhs", force_name))?.as_ptr(),
                        ));
                        let widening_mul = u!(LLVMBuildMul(
                            self.builder,
                            widening_lhs,
                            widening_rhs,
                            cs(&format!("{}_widening_mul", force_name))?.as_ptr(),
                        ));
                        let shifted = u!(LLVMBuildAShr(
                            self.builder,
                            widening_mul,
                            LLVMConstInt(widening_t, Value::BITS as u64, 0),
                            cs(&format!("{}_widening_shifted", force_name))?.as_ptr(),
                        ));
                        u!(LLVMBuildIntCast2(
                            self.builder,
                            shifted,
                            t,
                            0,
                            cs(&format!("{}_mulhsu", force_name))?.as_ptr(),
                        ))
                    }
                    ActionOp2::Clmul => {
                        let t = self.machine_word_type()?;
                        let mut result = u!(LLVMConstInt(t, 0, 0));
                        for i in 0..Value::BITS {
                            let rhs_shifted = u!(LLVMBuildLShr(
                                self.builder,
                                rhs,
                                LLVMConstInt(t, i as u64, 0),
                                cs(&format!("{}_rhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let shifted_and1 = u!(LLVMBuildAnd(
                                self.builder,
                                rhs_shifted,
                                LLVMConstInt(t, 1, 0),
                                cs(&format!("{}_rhs_shifted{}_and1", force_name, i))?.as_ptr(),
                            ));
                            let cmp = u!(LLVMBuildICmp(
                                self.builder,
                                LLVMIntPredicate::LLVMIntNE,
                                shifted_and1,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_rhs_shifted{}_and1_cmp", force_name, i))?.as_ptr(),
                            ));
                            let lhs_shifted = u!(LLVMBuildShl(
                                self.builder,
                                lhs,
                                LLVMConstInt(t, i as u64, 0),
                                cs(&format!("{}_lhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let xor_target = u!(LLVMBuildSelect(
                                self.builder,
                                cmp,
                                lhs_shifted,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_xor_target{}", force_name, i))?.as_ptr(),
                            ));
                            result = u!(LLVMBuildXor(
                                self.builder,
                                result,
                                xor_target,
                                cs(&format!("{}_clmul_round{}", force_name, i))?.as_ptr(),
                            ));
                        }
                        result
                    }
                    ActionOp2::Clmulh => {
                        let t = self.machine_word_type()?;
                        let mut result = u!(LLVMConstInt(t, 0, 0));
                        for i in 1..Value::BITS {
                            let rhs_shifted = u!(LLVMBuildLShr(
                                self.builder,
                                rhs,
                                LLVMConstInt(t, i as u64, 0),
                                cs(&format!("{}_rhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let shifted_and1 = u!(LLVMBuildAnd(
                                self.builder,
                                rhs_shifted,
                                LLVMConstInt(t, 1, 0),
                                cs(&format!("{}_rhs_shifted{}_and1", force_name, i))?.as_ptr(),
                            ));
                            let cmp = u!(LLVMBuildICmp(
                                self.builder,
                                LLVMIntPredicate::LLVMIntNE,
                                shifted_and1,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_rhs_shifted{}_and1_cmp", force_name, i))?.as_ptr(),
                            ));
                            let lhs_shifted = u!(LLVMBuildLShr(
                                self.builder,
                                lhs,
                                LLVMBuildSub(
                                    self.builder,
                                    LLVMConstInt(t, Value::BITS as u64, 0),
                                    LLVMConstInt(t, i as u64, 0),
                                    cs(&format!("{}_lhs_shifted_amount{}", force_name, i))?
                                        .as_ptr(),
                                ),
                                cs(&format!("{}_lhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let xor_target = u!(LLVMBuildSelect(
                                self.builder,
                                cmp,
                                lhs_shifted,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_xor_target{}", force_name, i))?.as_ptr(),
                            ));
                            result = u!(LLVMBuildXor(
                                self.builder,
                                result,
                                xor_target,
                                cs(&format!("{}_clmul_round{}", force_name, i))?.as_ptr(),
                            ));
                        }
                        result
                    }
                    ActionOp2::Clmulr => {
                        let t = self.machine_word_type()?;
                        let mut result = u!(LLVMConstInt(t, 0, 0));
                        for i in 0..Value::BITS {
                            let rhs_shifted = u!(LLVMBuildLShr(
                                self.builder,
                                rhs,
                                LLVMConstInt(t, i as u64, 0),
                                cs(&format!("{}_rhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let shifted_and1 = u!(LLVMBuildAnd(
                                self.builder,
                                rhs_shifted,
                                LLVMConstInt(t, 1, 0),
                                cs(&format!("{}_rhs_shifted{}_and1", force_name, i))?.as_ptr(),
                            ));
                            let cmp = u!(LLVMBuildICmp(
                                self.builder,
                                LLVMIntPredicate::LLVMIntNE,
                                shifted_and1,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_rhs_shifted{}_and1_cmp", force_name, i))?.as_ptr(),
                            ));
                            let lhs_shifted = u!(LLVMBuildLShr(
                                self.builder,
                                lhs,
                                LLVMBuildSub(
                                    self.builder,
                                    LLVMConstInt(t, Value::BITS as u64 - 1, 0),
                                    LLVMConstInt(t, i as u64, 0),
                                    cs(&format!("{}_lhs_shifted_amount{}", force_name, i))?
                                        .as_ptr(),
                                ),
                                cs(&format!("{}_lhs_shifted{}", force_name, i))?.as_ptr(),
                            ));
                            let xor_target = u!(LLVMBuildSelect(
                                self.builder,
                                cmp,
                                lhs_shifted,
                                LLVMConstInt(t, 0, 0),
                                cs(&format!("{}_xor_target{}", force_name, i))?.as_ptr(),
                            ));
                            result = u!(LLVMBuildXor(
                                self.builder,
                                result,
                                xor_target,
                                cs(&format!("{}_clmul_round{}", force_name, i))?.as_ptr(),
                            ));
                        }
                        result
                    }
                    ActionOp2::Rol => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t, t, t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.fshl.*\0".as_ptr() as *const _,
                            11
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External("Missing intrinsic for fshl!".to_string()));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                1,
                            ),
                            "get fshl intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get fshl intrinsic type"
                        );

                        let mut args = [lhs, lhs, rhs];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_rol", force_name))?.as_ptr(),
                        ))
                    }
                    ActionOp2::Ror => {
                        let t = self.machine_word_type()?;
                        let mut args_type = [t, t, t];

                        let intrinsic_id = u!(LLVMLookupIntrinsicID(
                            b"llvm.fshr.*\0".as_ptr() as *const _,
                            11
                        ));
                        if intrinsic_id == 0 {
                            return Err(Error::External("Missing intrinsic for fshr!".to_string()));
                        }
                        let intrinsic_func = assert_llvm_create!(
                            LLVMGetIntrinsicDeclaration(
                                self.module,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                1,
                            ),
                            "get fshr intrinsic func"
                        );
                        let intrinsic_type = assert_llvm_create!(
                            PatchedLLVMIntrinsicGetType(
                                self.context,
                                intrinsic_id,
                                args_type.as_mut_ptr(),
                                args_type.len() as usize,
                            ),
                            "get fshr intrinsic type"
                        );

                        let mut args = [lhs, lhs, rhs];
                        u!(LLVMBuildCall2(
                            self.builder,
                            intrinsic_type,
                            intrinsic_func,
                            args.as_mut_ptr(),
                            args.len() as u32,
                            cs(&format!("{}_ror", force_name))?.as_ptr(),
                        ))
                    }
                })
            }
            Value::SignOp2(op, lhs, rhs_original, signed) => {
                let lhs = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*lhs,
                    Some(&format!("{}_lhs", force_name)),
                )?;
                let rhs = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*rhs_original,
                    Some(&format!("{}_rhs", force_name)),
                )?;
                Ok(match op {
                    SignActionOp2::Shr => {
                        let f = if *signed {
                            LLVMBuildAShr
                        } else {
                            LLVMBuildLShr
                        };
                        let suffix = if *signed { "ashr" } else { "lshr" };
                        u!(f(
                            self.builder,
                            lhs,
                            rhs,
                            cs(&format!("{}_{}", force_name, suffix))?.as_ptr()
                        ))
                    }
                    SignActionOp2::Lt => {
                        let p = if *signed {
                            LLVMIntPredicate::LLVMIntSLT
                        } else {
                            LLVMIntPredicate::LLVMIntULT
                        };
                        let suffix = if *signed { "alt" } else { "llt" };
                        let v = u!(LLVMBuildICmp(
                            self.builder,
                            p,
                            lhs,
                            rhs,
                            cs(&format!("{}_{}_i1", force_name, suffix))?.as_ptr()
                        ));
                        u!(LLVMBuildSelect(
                            self.builder,
                            v,
                            LLVMConstInt(i64t, 1, 0),
                            LLVMConstInt(i64t, 0, 0),
                            cs(&format!("{}_{}", force_name, suffix))?.as_ptr()
                        ))
                    }
                    SignActionOp2::Extend => {
                        // For certain rhs value, we can build shortcuts.
                        let target_type = match &**rhs_original {
                            Value::Imm(i) if *i == 8 => Some(self.i8t()?),
                            Value::Imm(i) if *i == 16 => Some(self.i16t()?),
                            Value::Imm(i) if *i == 32 => Some(self.i32t()?),
                            Value::Imm(i) if *i == 64 => Some(i64t),
                            _ => None,
                        };
                        if let Some(target_type) = target_type {
                            let f = if *signed {
                                LLVMBuildSExt
                            } else {
                                LLVMBuildZExt
                            };
                            u!(f(
                                self.builder,
                                LLVMBuildIntCast2(
                                    self.builder,
                                    lhs,
                                    target_type,
                                    if *signed { 1 } else { 0 },
                                    cs(&format!("{}_cast", force_name))?.as_ptr(),
                                ),
                                i64t,
                                cs(&format!("{}_shortcut_extend", force_name))?.as_ptr(),
                            ))
                        } else {
                            let shifts = u!(LLVMBuildSub(
                                self.builder,
                                LLVMConstInt(i64t, 64, 0),
                                rhs,
                                cs(&format!("{}_shifts", force_name))?.as_ptr(),
                            ));
                            let shr_f = if *signed {
                                LLVMBuildAShr
                            } else {
                                LLVMBuildLShr
                            };
                            u!(shr_f(
                                self.builder,
                                LLVMBuildShl(
                                    self.builder,
                                    lhs,
                                    shifts,
                                    cs(&format!("{}_slowpath_extend_intermediate", force_name))?
                                        .as_ptr(),
                                ),
                                shifts,
                                cs(&format!("{}_slowpath_extend", force_name))?.as_ptr(),
                            ))
                        }
                    }
                    SignActionOp2::Mulh => {
                        let i128t = self.i128t()?;
                        let llvm_signed = if *signed { 1 } else { 0 };
                        let lhs128 = u!(LLVMBuildIntCast2(
                            self.builder,
                            lhs,
                            i128t,
                            llvm_signed,
                            cs(&format!("{}_lhs_128", force_name))?.as_ptr()
                        ));
                        let rhs128 = u!(LLVMBuildIntCast2(
                            self.builder,
                            rhs,
                            i128t,
                            llvm_signed,
                            cs(&format!("{}_rhs_128", force_name))?.as_ptr()
                        ));
                        let result_128 = u!(LLVMBuildMul(
                            self.builder,
                            lhs128,
                            rhs128,
                            cs(&format!("{}_result_128", force_name))?.as_ptr()
                        ));
                        // It doesn't matter what shift we use
                        let result_shifts_128 = u!(LLVMBuildLShr(
                            self.builder,
                            result_128,
                            LLVMConstInt(i128t, 64, 0),
                            cs(&format!("{}_result_shifts_128", force_name))?.as_ptr()
                        ));
                        u!(LLVMBuildIntCast2(
                            self.builder,
                            result_shifts_128,
                            i64t,
                            llvm_signed,
                            cs(&format!("{}_mulh", force_name))?.as_ptr()
                        ))
                    }
                    SignActionOp2::Div | SignActionOp2::Rem => {
                        let is_div = if let SignActionOp2::Div = op {
                            true
                        } else {
                            false
                        };
                        let current_block = u!(LLVMGetInsertBlock(self.builder));
                        let zero_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                cs(&format!("{}_zero_rhs_block", force_name))?.as_ptr()
                            ),
                            "create zero block"
                        );
                        u!(LLVMAppendExistingBasicBlock(function, zero_block));
                        let non_zero_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                cs(&format!("{}_zero_rhs_block", force_name))?.as_ptr()
                            ),
                            "create zero block"
                        );
                        u!(LLVMAppendExistingBasicBlock(function, non_zero_block));
                        let overflow_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                cs(&format!("{}_overflow_block", force_name))?.as_ptr()
                            ),
                            "create overflow block"
                        );
                        let final_merge_block = assert_llvm_create!(
                            LLVMCreateBasicBlockInContext(
                                self.context,
                                cs(&format!("{}_final_merge_block", force_name))?.as_ptr()
                            ),
                            "create final merge block"
                        );
                        u!(LLVMAppendExistingBasicBlock(function, final_merge_block));

                        u!(LLVMPositionBuilderAtEnd(self.builder, current_block));
                        let rhs_is_0 = u!(LLVMBuildICmp(
                            self.builder,
                            LLVMIntPredicate::LLVMIntEQ,
                            rhs,
                            LLVMConstInt(i64t, 0, 0),
                            cs(&format!("{}_rhs_is_zero", force_name))?.as_ptr()
                        ));
                        u!(LLVMBuildCondBr(
                            self.builder,
                            rhs_is_0,
                            zero_block,
                            non_zero_block,
                        ));

                        u!(LLVMPositionBuilderAtEnd(self.builder, zero_block));
                        let zero_result = if is_div {
                            u!(LLVMConstInt(i64t, u64::max_value(), 0))
                        } else {
                            lhs
                        };
                        u!(LLVMBuildBr(self.builder, final_merge_block));

                        let (else_result, else_block) = if *signed {
                            u!(LLVMAppendExistingBasicBlock(function, overflow_block));
                            let compute_block = assert_llvm_create!(
                                LLVMCreateBasicBlockInContext(
                                    self.context,
                                    cs(&format!("{}_compute_block", force_name))?.as_ptr()
                                ),
                                "create compute block"
                            );
                            u!(LLVMAppendExistingBasicBlock(function, compute_block));
                            let non_zero_merge_block = assert_llvm_create!(
                                LLVMCreateBasicBlockInContext(
                                    self.context,
                                    cs(&format!("{}_non_zero_merge_block", force_name))?.as_ptr()
                                ),
                                "create non zero merge block"
                            );
                            u!(LLVMAppendExistingBasicBlock(function, non_zero_merge_block));

                            u!(LLVMPositionBuilderAtEnd(self.builder, non_zero_block));
                            let overflow = u!(LLVMBuildAnd(
                                self.builder,
                                LLVMBuildICmp(
                                    self.builder,
                                    LLVMIntPredicate::LLVMIntEQ,
                                    lhs,
                                    LLVMConstInt(i64t, i64::min_value() as u64, 1),
                                    cs(&format!("{}_overflow_lhs", force_name))?.as_ptr()
                                ),
                                LLVMBuildICmp(
                                    self.builder,
                                    LLVMIntPredicate::LLVMIntEQ,
                                    rhs,
                                    LLVMConstInt(i64t, (-1i64) as u64, 1),
                                    cs(&format!("{}_overflow_rhs", force_name))?.as_ptr()
                                ),
                                cs(&format!("{}_overflow", force_name))?.as_ptr()
                            ));
                            u!(LLVMBuildCondBr(
                                self.builder,
                                overflow,
                                overflow_block,
                                compute_block,
                            ));
                            u!(LLVMPositionBuilderAtEnd(self.builder, overflow_block));
                            let overflow_result = if is_div {
                                lhs
                            } else {
                                u!(LLVMConstInt(i64t, 0, 0))
                            };
                            u!(LLVMBuildBr(self.builder, non_zero_merge_block));
                            u!(LLVMPositionBuilderAtEnd(self.builder, compute_block));
                            let compute_result = if is_div {
                                u!(LLVMBuildSDiv(
                                    self.builder,
                                    lhs,
                                    rhs,
                                    cs(&format!("{}_actual_sdiv", force_name))?.as_ptr()
                                ))
                            } else {
                                u!(LLVMBuildSRem(
                                    self.builder,
                                    lhs,
                                    rhs,
                                    cs(&format!("{}_actual_srem", force_name))?.as_ptr()
                                ))
                            };
                            u!(LLVMBuildBr(self.builder, non_zero_merge_block));
                            u!(LLVMPositionBuilderAtEnd(self.builder, non_zero_merge_block));
                            let non_zero_merge_result = u!(LLVMBuildPhi(
                                self.builder,
                                i64t,
                                cs(&format!("{}_non_zero_merge_result", force_name))?.as_ptr()
                            ));
                            let mut incoming_values = [overflow_result, compute_result];
                            let mut incoming_blocks = [overflow_block, compute_block];
                            u!(LLVMAddIncoming(
                                non_zero_merge_result,
                                incoming_values.as_mut_ptr(),
                                incoming_blocks.as_mut_ptr(),
                                2
                            ));
                            u!(LLVMBuildBr(self.builder, final_merge_block));
                            (non_zero_merge_result, non_zero_merge_block)
                        } else {
                            u!(LLVMPositionBuilderAtEnd(self.builder, non_zero_block));
                            let compute_result = if is_div {
                                u!(LLVMBuildUDiv(
                                    self.builder,
                                    lhs,
                                    rhs,
                                    cs(&format!("{}_actual_unsigned_v", force_name))?.as_ptr()
                                ))
                            } else {
                                u!(LLVMBuildURem(
                                    self.builder,
                                    lhs,
                                    rhs,
                                    cs(&format!("{}_actual_urem", force_name))?.as_ptr()
                                ))
                            };
                            u!(LLVMBuildBr(self.builder, final_merge_block));
                            (compute_result, non_zero_block)
                        };

                        u!(LLVMPositionBuilderAtEnd(self.builder, final_merge_block));
                        let final_result = u!(LLVMBuildPhi(
                            self.builder,
                            i64t,
                            cs(&format!("{}_final_merge_result", force_name))?.as_ptr()
                        ));
                        let mut incoming_values = [zero_result, else_result];
                        let mut incoming_blocks = [zero_block, else_block];
                        u!(LLVMAddIncoming(
                            final_result,
                            incoming_values.as_mut_ptr(),
                            incoming_blocks.as_mut_ptr(),
                            2
                        ));

                        final_result
                    }
                })
            }
            Value::Cond(c, t, f) => {
                let t = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*t,
                    Some(&format!("{}_t", force_name)),
                )?;
                let f = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*f,
                    Some(&format!("{}_f", force_name)),
                )?;
                let c = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*c,
                    Some(&format!("{}_c", force_name)),
                )?;
                let c = u!(LLVMBuildICmp(
                    self.builder,
                    LLVMIntPredicate::LLVMIntEQ,
                    c,
                    LLVMConstInt(i64t, 1, 0),
                    cs(&format!("{}_c_i1", force_name))?.as_ptr(),
                ));
                Ok(u!(LLVMBuildSelect(
                    self.builder,
                    c,
                    t,
                    f,
                    cs(&format!("{}_cond", force_name))?.as_ptr()
                )))
            }
            Value::Load(addr, size) => {
                let t = self.size_to_type(*size)?;
                // TODO: should we cap addr somehow?
                let addr = self.emit_value(
                    machine,
                    memory_start,
                    function,
                    allocas,
                    &*addr,
                    Some(&format!("{}_addr", force_name)),
                )?;
                let real_address_value = u!(LLVMBuildAdd(
                    self.builder,
                    memory_start,
                    addr,
                    cs(&format!("{}_real_addr_val", force_name))?.as_ptr(),
                ));
                let real_address = u!(LLVMBuildIntToPtr(
                    self.builder,
                    real_address_value,
                    LLVMPointerType(t, 0),
                    cs(&format!("{}_real_addr", force_name))?.as_ptr(),
                ));
                let value = u!(LLVMBuildLoad2(
                    self.builder,
                    t,
                    real_address,
                    cs(&format!("{}_loaded_value", force_name))?.as_ptr()
                ));
                Ok(u!(LLVMBuildIntCast2(
                    self.builder,
                    value,
                    i64t,
                    0,
                    cs(&format!("{}_load_casted", force_name))?.as_ptr()
                )))
            }
        }
    }

    // Emit the code to call another RISC-V function, and also the code to save return
    // values to allocas.
    fn emit_call_riscv_func(
        &mut self,
        machine: LLVMValueRef,
        allocas: &RegAllocas,
        riscv_func_address: u64,
    ) -> Result<(), Error> {
        let func_llvm_value = *(self.emitted_funcs.get(&riscv_func_address).ok_or_else(|| {
            Error::External(format!(
                "Function at 0x{:x} does not exist!",
                riscv_func_address
            ))
        })?);

        self.emit_call_riscv_func_with_func_value(machine, allocas, func_llvm_value)
    }

    fn emit_call_riscv_func_with_func_value(
        &mut self,
        machine: LLVMValueRef,
        allocas: &RegAllocas,
        func_llvm_value: LLVMValueRef,
    ) -> Result<(), Error> {
        let i64t = self.i64t()?;
        let riscv_function_type = self.riscv_function_type()?;

        // Keep track of previous last_ra value for nested calls
        let previous_last_ra = self.emit_load_from_machine(
            machine,
            offset_of!(LlvmAotCoreMachineData, last_ra),
            i64t,
            Some("previous_last_ra"),
        )?;
        // Set current RA to last_ra to aid ret operations
        let last_ra = self.emit_load_reg(machine, &allocas, registers::RA, Some("ra"))?;
        self.emit_store_to_machine(
            machine,
            last_ra,
            offset_of!(LlvmAotCoreMachineData, last_ra),
            i64t,
            Some("last_ra"),
        )?;

        let values = allocas.load_values()?;
        let mut invoke_args = values.to_arguments();
        let result = u!(LLVMBuildCall2(
            self.builder,
            riscv_function_type,
            func_llvm_value,
            invoke_args.as_mut_ptr(),
            invoke_args.len() as u32,
            b"riscv_call_result\0".as_ptr() as *const _
        ));
        u!(LLVMSetInstructionCallConv(
            result,
            LLVMCallConv::LLVMHHVMCallConv as u32
        ));
        // When returned from the function call, restore previously saved last_ra
        self.emit_store_to_machine(
            machine,
            previous_last_ra,
            offset_of!(LlvmAotCoreMachineData, last_ra),
            i64t,
            Some("restored_last_ra"),
        )?;
        // Now save return values
        let mut ret_values = [ptr::null_mut(); 14];
        for i in 0..14 {
            ret_values[i] = u!(LLVMBuildExtractValue(
                self.builder,
                result,
                i as u32,
                cs(&format!("riscv_call_ret{}", i))?.as_ptr(),
            ));
        }
        let ret_values = TransientValues::from_return_values(ret_values, machine);
        allocas.store_values(&ret_values)?;
        Ok(())
    }

    // Emit the code to call early exit function from a RISC-V function
    fn emit_call_exit(
        &mut self,
        machine: LLVMValueRef,
        reason: u8,
        allocas: &RegAllocas,
    ) -> Result<(), Error> {
        let values = allocas.load_values()?;

        let i8t = self.i8t()?;
        self.emit_store_to_machine(
            machine,
            u!(LLVMConstInt(i8t, reason as u64, 1)),
            offset_of!(LlvmAotCoreMachineData, exit_aot_reason),
            i8t,
            Some("exit_aot_reason"),
        )?;
        let mut invoke_args = values.to_arguments();
        let call = u!(LLVMBuildCall2(
            self.builder,
            self.exit_function_type()?,
            self.exit_func,
            invoke_args.as_mut_ptr(),
            invoke_args.len() as u32,
            b"\0".as_ptr() as *const _
        ));
        u!(LLVMSetInstructionCallConv(
            call,
            LLVMCallConv::LLVMHHVMCallConv as u32
        ));
        u!(LLVMBuildUnreachable(self.builder));
        Ok(())
    }

    // Loading a register value as an LLVM value in a RISC-V function
    fn emit_load_reg(
        &mut self,
        machine: LLVMValueRef,
        allocas: &RegAllocas,
        reg: usize,
        name: Option<&str>,
    ) -> Result<LLVMValueRef, Error> {
        if reg == 0 {
            return Ok(u!(LLVMConstInt(self.i64t()?, 0, 0)));
        }
        let i64t = self.i64t()?;
        if let Some(i) = REVERSE_MAPPINGS.get(&reg) {
            allocas.load_value(*i)
        } else {
            self.emit_load_from_machine(
                machine,
                offset_of!(LlvmAotCoreMachineData, registers) + reg * 8,
                i64t,
                name,
            )
        }
    }

    // Storing an LLVM value into a register in a RISC-V function
    fn emit_store_reg(
        &mut self,
        machine: LLVMValueRef,
        allocas: &RegAllocas,
        reg: usize,
        value: LLVMValueRef,
    ) -> Result<(), Error> {
        if reg == 0 {
            return Ok(());
        }
        if let Some(i) = REVERSE_MAPPINGS.get(&reg) {
            allocas.store_value(*i, value)
        } else {
            let i64t = self.i64t()?;
            self.emit_store_to_machine(
                machine,
                value,
                offset_of!(LlvmAotCoreMachineData, registers) + reg * 8,
                i64t,
                Some(&format!("{}", register_names(reg))),
            )
        }
    }

    // Emit store to LlvmAotCoreMachineData operation
    fn emit_store_to_machine(
        &mut self,
        machine: LLVMValueRef,
        value: LLVMValueRef,
        offset: usize,
        value_type: LLVMTypeRef,
        name: Option<&str>,
    ) -> Result<(), Error> {
        let i64t = self.i64t()?;
        self.emit_store_with_value_offset_to_machine(
            machine,
            value,
            u!(LLVMConstInt(i64t, offset as u64, 0)),
            value_type,
            name,
        )
    }

    // Similar to emit_store_to_machine, but accepts offset as an LLVM value,
    // which could include runtime computable values.
    fn emit_store_with_value_offset_to_machine(
        &mut self,
        machine: LLVMValueRef,
        value: LLVMValueRef,
        offset: LLVMValueRef,
        value_type: LLVMTypeRef,
        name: Option<&str>,
    ) -> Result<(), Error> {
        let value_pointer_type =
            assert_llvm_create!(LLVMPointerType(value_type, 0), "pointer type");

        let name = name.unwrap_or("uNKNOWn");
        let addr_value = u!(LLVMBuildAdd(
            self.builder,
            machine,
            offset,
            cs(&format!("{}_addr_val", name))?.as_ptr(),
        ));
        let addr = u!(LLVMBuildIntToPtr(
            self.builder,
            addr_value,
            value_pointer_type,
            cs(&format!("{}_addr", name))?.as_ptr(),
        ));
        u!(LLVMBuildStore(self.builder, value, addr));
        Ok(())
    }

    // Emit load from LlvmAotCoreMachineData operation
    fn emit_load_from_machine(
        &mut self,
        machine: LLVMValueRef,
        offset: usize,
        value_type: LLVMTypeRef,
        name: Option<&str>,
    ) -> Result<LLVMValueRef, Error> {
        let i64t = self.i64t()?;
        self.emit_load_from_struct(
            machine,
            u!(LLVMConstInt(i64t, offset as u64, 0)),
            value_type,
            name,
        )
    }

    // Load values at any offset from a struct.
    fn emit_load_from_struct(
        &mut self,
        machine: LLVMValueRef,
        offset: LLVMValueRef,
        value_type: LLVMTypeRef,
        name: Option<&str>,
    ) -> Result<LLVMValueRef, Error> {
        let value_pointer_type =
            assert_llvm_create!(LLVMPointerType(value_type, 0), "pointer type");

        let name = name.unwrap_or("uNKNOWn");
        let addr_value = u!(LLVMBuildAdd(
            self.builder,
            machine,
            offset,
            cs(&format!("{}_addr_val", name))?.as_ptr(),
        ));
        let addr = u!(LLVMBuildIntToPtr(
            self.builder,
            addr_value,
            value_pointer_type,
            cs(&format!("{}_addr", name))?.as_ptr(),
        ));
        let value = u!(LLVMBuildLoad2(
            self.builder,
            value_type,
            addr,
            cs(&format!("{}_loaded_value", name))?.as_ptr(),
        ));
        Ok(value)
    }

    // Emit code used to locate FFI function together with the data object.
    fn emit_env_ffi_function(
        &mut self,
        machine: LLVMValueRef,
        offset: usize,
    ) -> Result<(LLVMValueRef, LLVMValueRef), Error> {
        let i64t = self.i64t()?;
        let env = self.emit_load_from_machine(
            machine,
            offset_of!(LlvmAotCoreMachineData, env),
            i64t,
            Some("env"),
        )?;
        let data = self.emit_load_from_struct(
            env,
            u!(LLVMConstInt(
                i64t,
                offset_of!(LlvmAotMachineEnv, data) as u64,
                0
            )),
            i64t,
            Some("env_data"),
        )?;
        let ffi_function_value = self.emit_load_from_struct(
            env,
            u!(LLVMConstInt(i64t, offset as u64, 0)),
            i64t,
            Some("env_ffi_function_value"),
        )?;
        let ffi_function = u!(LLVMBuildIntToPtr(
            self.builder,
            ffi_function_value,
            LLVMPointerType(self.ffi_function_type()?, 0),
            b"env_ffi_function\0".as_ptr() as *const _,
        ));
        Ok((ffi_function, data))
    }

    // In case a function contains indirect dispatches, we will emit a special
    // basic block used to locate the correct basic block after the indirect
    // dispatch.
    fn emit_indirect_dispatch_block(
        &mut self,
        machine: LLVMValueRef,
        function: LLVMValueRef,
        allocas: &RegAllocas,
        test_value_alloca: LLVMValueRef,
        basic_blocks: &HashMap<u64, LLVMBasicBlockRef>,
    ) -> Result<LLVMBasicBlockRef, Error> {
        let current_block = u!(LLVMGetInsertBlock(self.builder));

        let dispatch_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                b"indirect_dispatch_block\0".as_ptr() as *const _,
            ),
            "create dispatch block"
        );
        u!(LLVMAppendExistingBasicBlock(function, dispatch_block));
        u!(LLVMPositionBuilderAtEnd(self.builder, dispatch_block));

        let test_value = u!(LLVMBuildLoad2(
            self.builder,
            self.i64t()?,
            test_value_alloca,
            b"indirect_dispatch_test_value\0".as_ptr() as *const _,
        ));

        let failure_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                b"indirect_jump_failure_block\0".as_ptr() as *const _,
            ),
            "create failure block"
        );
        u!(LLVMAppendExistingBasicBlock(function, failure_block));

        let mut indirect_jump_targets: Vec<(u64, LLVMBasicBlockRef)> =
            basic_blocks.iter().map(|(a, b)| (*a, *b)).collect();
        indirect_jump_targets.sort_by_key(|(a, _)| *a);
        self.emit_select_control(
            function,
            dispatch_block,
            test_value,
            &indirect_jump_targets,
            failure_block,
        )?;

        u!(LLVMPositionBuilderAtEnd(self.builder, failure_block));
        self.emit_call_exit(machine, EXIT_REASON_UNKNOWN_PC_VALUE, &allocas)?;

        u!(LLVMPositionBuilderAtEnd(self.builder, current_block));

        Ok(dispatch_block)
    }

    // Emit a binary search pattern picking target to branch to. Jump
    // to failure_block if no matching value is found.
    fn emit_select_control(
        &mut self,
        function: LLVMValueRef,
        current_block: LLVMBasicBlockRef,
        test_value: LLVMValueRef,
        targets: &[(u64, LLVMBasicBlockRef)],
        failure_block: LLVMBasicBlockRef,
    ) -> Result<(), Error> {
        u!(LLVMPositionBuilderAtEnd(self.builder, current_block));
        if targets.is_empty() {
            u!(LLVMBuildBr(self.builder, failure_block));
            return Ok(());
        }

        let mid = targets.len() / 2;
        let mid_value = u!(LLVMConstInt(self.i64t()?, targets[mid].0, 0));
        // First test for equality
        let ne_block = self.emit_match_or_new_block(
            function,
            LLVMIntPredicate::LLVMIntEQ,
            test_value,
            mid_value,
            targets[mid].1,
        )?;

        u!(LLVMPositionBuilderAtEnd(self.builder, ne_block));
        let right_block = if mid > 0 {
            // Test for left branch
            let left_block = assert_llvm_create!(
                LLVMCreateBasicBlockInContext(
                    self.context,
                    b"select_control_left_block\0".as_ptr() as *const _,
                ),
                "create left block"
            );
            u!(LLVMAppendExistingBasicBlock(function, left_block));
            self.emit_select_control(
                function,
                left_block,
                test_value,
                &targets[0..mid],
                failure_block,
            )?;

            u!(LLVMPositionBuilderAtEnd(self.builder, ne_block));
            self.emit_match_or_new_block(
                function,
                LLVMIntPredicate::LLVMIntULT,
                test_value,
                mid_value,
                left_block,
            )?
        } else {
            // Left branch is empty, switch to right branch directly
            ne_block
        };

        u!(LLVMPositionBuilderAtEnd(self.builder, right_block));
        if mid < targets.len() - 1 {
            // Test for right branch
            self.emit_select_control(
                function,
                right_block,
                test_value,
                &targets[(mid + 1)..targets.len()],
                failure_block,
            )?;
        } else {
            // Right branch is empty, matching results in a failure
            u!(LLVMBuildBr(self.builder, failure_block));
        }

        Ok(())
    }

    // Given a set of icmp inputs, perform the actual cmp operation, if
    // true, branch to +true_block+, otherwise create a new block to jump
    // to.
    fn emit_match_or_new_block(
        &mut self,
        function: LLVMValueRef,
        predicate: LLVMIntPredicate,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef,
        true_block: LLVMBasicBlockRef,
    ) -> Result<LLVMBasicBlockRef, Error> {
        let else_block = assert_llvm_create!(
            LLVMCreateBasicBlockInContext(
                self.context,
                b"select_control_else_block\0".as_ptr() as *const _,
            ),
            "create else block"
        );
        u!(LLVMAppendExistingBasicBlock(function, else_block));

        let cmp = u!(LLVMBuildICmp(
            self.builder,
            predicate,
            lhs,
            rhs,
            b"select_control_cmp\0".as_ptr() as *const _,
        ));
        u!(LLVMBuildCondBr(self.builder, cmp, true_block, else_block));

        Ok(else_block)
    }

    fn ffi_wrapper_function_type(&mut self) -> Result<LLVMTypeRef, Error> {
        let i64t = self.i64t()?;
        let mut argts = [
            u!(LLVMPointerType(self.ffi_function_type()?, 0)),
            i64t,
            i64t,
        ];
        Ok(assert_llvm_create!(
            LLVMFunctionType(i64t, argts.as_mut_ptr(), argts.len() as u32, 0),
            "ffi function type"
        ))
    }

    fn ffi_function_type(&mut self) -> Result<LLVMTypeRef, Error> {
        let i64t = self.i64t()?;
        let mut argts = [i64t, i64t];
        Ok(assert_llvm_create!(
            LLVMFunctionType(i64t, argts.as_mut_ptr(), argts.len() as u32, 0),
            "ffi function type"
        ))
    }

    fn riscv_function_type(&mut self) -> Result<LLVMTypeRef, Error> {
        // For each generated RISC-V function, the arguments will be 15 i64 values,
        // the return type will be a struct of also 14 i64 values. See TransientValues
        // below for more details
        let mut packed_reg_type = [self.i64t()?; 15];
        let return_type = assert_llvm_create!(
            LLVMStructTypeInContext(self.context, packed_reg_type.as_mut_ptr(), 14, 0),
            "struct type"
        );
        let function_type = assert_llvm_create!(
            LLVMFunctionType(return_type, packed_reg_type.as_mut_ptr(), 15, 0),
            "function type"
        );
        Ok(function_type)
    }

    fn exit_function_type(&mut self) -> Result<LLVMTypeRef, Error> {
        // Exit function is invoked from a RISC-V function, it shares the same signature
        // as RISC-V functions, except that it has no returns, since the internal longjmp
        // will trigger unwinding.
        let mut args_type = [self.i64t()?; 15];
        let t = assert_llvm_create!(
            LLVMFunctionType(
                LLVMVoidTypeInContext(self.context),
                args_type.as_mut_ptr(),
                args_type.len() as u32,
                0
            ),
            "exit function type"
        );
        Ok(t)
    }

    fn machine_word_type(&mut self) -> Result<LLVMTypeRef, Error> {
        self.size_to_type(Value::BITS >> 3)
    }

    fn double_machine_word_type(&mut self) -> Result<LLVMTypeRef, Error> {
        self.size_to_type(Value::BITS >> 2)
    }

    fn size_to_type(&mut self, size: u8) -> Result<LLVMTypeRef, Error> {
        match size {
            1 => self.i8t(),
            2 => self.i16t(),
            4 => self.i32t(),
            8 => self.i64t(),
            16 => self.i128t(),
            _ => Err(Error::External(format!("Invalid load size: {}", size))),
        }
    }

    fn i128t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i128t = assert_llvm_create!(LLVMInt128TypeInContext(self.context), "int128 type");
        Ok(i128t)
    }

    fn i64t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i64t = assert_llvm_create!(LLVMInt64TypeInContext(self.context), "int64 type");
        Ok(i64t)
    }

    fn i32t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i32t = assert_llvm_create!(LLVMInt32TypeInContext(self.context), "int32 type");
        Ok(i32t)
    }

    fn i16t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i16t = assert_llvm_create!(LLVMInt16TypeInContext(self.context), "int16 type");
        Ok(i16t)
    }

    fn i8t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i8t = assert_llvm_create!(LLVMInt8TypeInContext(self.context), "int8 type");
        Ok(i8t)
    }

    fn i1t(&mut self) -> Result<LLVMTypeRef, Error> {
        let i1t = assert_llvm_create!(LLVMInt1TypeInContext(self.context), "int1 type");
        Ok(i1t)
    }

    fn i8pt(&mut self) -> Result<LLVMTypeRef, Error> {
        let i8pt = assert_llvm_create!(LLVMPointerType(self.i8t()?, 0), "i8 pointer type");
        Ok(i8pt)
    }

    fn basic_block_name(&self, addr: u64) -> Result<CString, Error> {
        cs(&format!("basic_block_0x{:x}", addr))
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

    fn hosted_setjmp_name(&self) -> String {
        format!("{}____hosted_setjmp____", self.symbol_prefix)
    }

    fn hosted_longjmp_name(&self) -> String {
        format!("{}____hosted_longjmp____", self.symbol_prefix)
    }
}

fn llvm_buffer_to_bytes(buf: LLVMMemoryBufferRef) -> Result<Bytes, Error> {
    let size = u!(LLVMGetBufferSize(buf));
    let start = assert_llvm_create!(LLVMGetBufferStart(buf), "get buffer start");
    let src_buf = unsafe { std::slice::from_raw_parts(start as *const u8, size) };

    let mut dst_buf = vec![0u8; size];
    dst_buf.copy_from_slice(&src_buf);

    u!(LLVMDisposeMemoryBuffer(buf));

    Ok(Bytes::from(dst_buf))
}

impl Drop for LlvmCompilingMachine {
    fn drop(&mut self) {
        if !self.di_builder.is_null() {
            u!(LLVMDisposeDIBuilder(self.di_builder));
        }
        u!(LLVMDisposeModule(self.module));
        u!(LLVMDisposeBuilder(self.builder));
        u!(LLVMContextDispose(self.context));
        u!(LLVMFinalizeFunctionPassManager(self.pass));
        u!(LLVMDisposePassManager(self.pass));
    }
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
pub struct TransientValues([LLVMValueRef; 15]);

impl TransientValues {
    fn new<F: FnMut(&Mapping) -> Result<LLVMValueRef, Error>>(mut f: F) -> Result<Self, Error> {
        let mut values = [ptr::null_mut(); 15];
        for (i, mapping) in REGISTER_MAPPINGS.iter().enumerate() {
            values[i] = f(mapping)?;
        }
        Ok(Self(values))
    }

    fn map<F: FnMut(LLVMValueRef, &Mapping) -> Result<LLVMValueRef, Error>>(
        &self,
        mut f: F,
    ) -> Result<Self, Error> {
        let mut values = [ptr::null_mut(); 15];
        for i in 0..15 {
            values[i] = f(self.0[i], &REGISTER_MAPPINGS[i])?;
        }
        Ok(Self(values))
    }

    // Extract pc value
    fn extract_pc(&self) -> Result<LLVMValueRef, Error> {
        let pc_index = REGISTER_MAPPINGS
            .iter()
            .position(|m| match m {
                Mapping::Pc => true,
                _ => false,
            })
            .ok_or_else(|| Error::External("PC is missing in mapping!".to_string()))?;
        Ok(self.0[pc_index])
    }

    // Extract machine value from RISC-V function arguments
    fn extract_machine(&self) -> Result<LLVMValueRef, Error> {
        let machine_index = REGISTER_MAPPINGS
            .iter()
            .position(|m| match m {
                Mapping::Pointer => true,
                _ => false,
            })
            .ok_or_else(|| Error::External("Machine pointer is missing in mapping!".to_string()))?;
        Ok(self.0[machine_index])
    }

    fn from_arguments(values: [LLVMValueRef; 15]) -> Self {
        Self(values)
    }

    fn from_return_values(values: [LLVMValueRef; 14], machine: LLVMValueRef) -> Self {
        Self::from_arguments(return_values_to_arguments(values, machine))
    }

    fn to_arguments(&self) -> [LLVMValueRef; 15] {
        self.0.clone()
    }

    fn to_return_values(&self) -> [LLVMValueRef; 14] {
        arguments_to_return_values(self.to_arguments())
    }

    fn iter(&self) -> TransientValuesIter {
        TransientValuesIter {
            values: self,
            index: 0,
        }
    }
}

pub struct RegAllocas {
    values: TransientValues,
    builder: LLVMBuilderRef,
    i64t: LLVMTypeRef,
}

impl RegAllocas {
    pub fn new(values: TransientValues, builder: LLVMBuilderRef, i64t: LLVMTypeRef) -> Self {
        Self {
            values,
            builder,
            i64t,
        }
    }

    pub fn pc_alloca(&self) -> Result<LLVMValueRef, Error> {
        self.values.extract_pc()
    }

    pub fn load_value(&self, idx: usize) -> Result<LLVMValueRef, Error> {
        Ok(u!(LLVMBuildLoad2(
            self.builder,
            self.i64t,
            self.values.0[idx],
            cs(&format!("reg_allocas_idx_{}", idx))?.as_ptr()
        )))
    }

    pub fn store_value(&self, idx: usize, value: LLVMValueRef) -> Result<(), Error> {
        u!(LLVMBuildStore(self.builder, value, self.values.0[idx],));
        Ok(())
    }

    pub fn load_values(&self) -> Result<TransientValues, Error> {
        self.values.map(|value, mapping| {
            Ok(u!(LLVMBuildLoad2(
                self.builder,
                self.i64t,
                value,
                cs(&format!("reg_allocas_tmp_{}", mapping))?.as_ptr()
            )))
        })
    }

    pub fn store_values(&self, values: &TransientValues) -> Result<(), Error> {
        for (i, (value, _)) in values.iter().enumerate() {
            u!(LLVMBuildStore(self.builder, value, self.values.0[i]));
        }
        Ok(())
    }
}

fn return_values_to_arguments(
    values: [LLVMValueRef; 14],
    machine: LLVMValueRef,
) -> [LLVMValueRef; 15] {
    [
        values[0], machine, values[1], values[13], values[2], values[3], values[4], values[5],
        values[6], values[7], values[8], values[9], values[10], values[11], values[12],
    ]
}

fn arguments_to_return_values(arguments: [LLVMValueRef; 15]) -> [LLVMValueRef; 14] {
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

pub struct TransientValuesIter<'a> {
    values: &'a TransientValues,
    index: usize,
}

impl<'a> Iterator for TransientValuesIter<'a> {
    type Item = (LLVMValueRef, Mapping);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.values.0.len() {
            let result = (
                self.values.0[self.index],
                REGISTER_MAPPINGS[self.index].clone(),
            );
            self.index += 1;
            return Some(result);
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mapping {
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

struct DebugFileWriter {
    context: LLVMContextRef,
    scope: LLVMMetadataRef,
    file: File,
    line: u32,
}

impl DebugFileWriter {
    pub fn new(
        context: LLVMContextRef,
        scope: LLVMMetadataRef,
        filename: &str,
    ) -> Result<Self, Error> {
        let file = File::create(filename)?;
        Ok(Self {
            context,
            scope,
            file,
            line: 0,
        })
    }

    pub fn write(&mut self, content: &str) -> Result<(), Error> {
        write!(self.file, "{}\n", content)?;
        self.line += 1;
        Ok(())
    }

    pub fn debug_location(&mut self) -> Result<LLVMMetadataRef, Error> {
        Ok(assert_llvm_create!(
            LLVMDIBuilderCreateDebugLocation(
                self.context,
                self.line,
                0,
                self.scope,
                ptr::null_mut(),
            ),
            "create debug location"
        ))
    }

    pub fn set_scope(&mut self, scope: LLVMMetadataRef) {
        self.scope = scope;
    }
}
