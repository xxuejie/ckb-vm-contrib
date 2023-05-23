use ckb_vm::{
    cost_model::estimate_cycles, machine::DefaultMachineBuilder, Bytes, Error, SupportMachine,
};
use ckb_vm_contrib::{
    llvm_aot::{
        ast::Control, preprocess, DlSymbols, Func, LlvmAotCoreMachine, LlvmAotMachine,
        LlvmCompilingMachine,
    },
    syscalls::{DebugSyscall, TimeSyscall},
};
use clap::{Parser, ValueEnum};
use log::info;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;
use tempfile::Builder;
use which::which;

#[derive(ValueEnum, Clone, Debug)]
enum Generate {
    Hash,
    Writes,
    Bitcode,
    Object,
    SharedLibrary,
    RunResult,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum OptimizeLevel {
    Default,
    None,
}

#[derive(Parser, Debug)]
#[clap(author, version, about = "A simple runner for ckb-vm's LLVM AOT engine", long_about = None)]
struct Args {
    /// Time critical operations when turned on
    #[clap(short, long)]
    time: bool,

    /// Optimize the generated code
    #[clap(long, default_value = "default")]
    optimize_level: OptimizeLevel,

    /// Generate debug information
    #[clap(short, long)]
    debug_info: bool,

    /// Max cycles when running the program
    #[clap(short, long, default_value = "18446744073709551615")]
    max_cycles: u64,

    /// CKB-VM memory size to use when running the program
    #[clap(short, long, default_value = "4194304")]
    memory_size: usize,

    /// Action to perform
    #[clap(value_enum, short, long, default_value = "object")]
    generate: Generate,

    /// Symbolx prefix to use when generating AOT module
    #[clap(short, long, default_value = "aot")]
    symbol_prefix: String,

    /// Output file path
    #[clap(short, long)]
    output: Option<String>,

    /// Input file path
    #[clap(short, long, required = true)]
    input: String,

    /// Arguments used to run the program
    #[clap()]
    run_args: Vec<String>,
}

impl Args {
    fn optimized(&self) -> bool {
        self.optimize_level == OptimizeLevel::Default
    }
}

fn main() -> Result<(), Error> {
    flexi_logger::Logger::try_with_env()
        .unwrap()
        .start()
        .unwrap();
    let args = Args::parse();

    let code: Bytes = std::fs::read(&args.input)?.into();
    LlvmCompilingMachine::initialize()?;

    match args.generate {
        Generate::Hash => {
            let code_hash: [u8; 32] = blake3::hash(&code).into();
            println!("{:x}", Bytes::from(code_hash.to_vec()));
        }
        Generate::Writes => {
            let output: Box<dyn Write> = match args.output {
                None => Box::new(io::stdout()),
                Some(s) if s == "-" => Box::new(io::stdout()),
                Some(o) => Box::new(File::create(&o)?),
            };
            let funcs = preprocess(&code, &estimate_cycles)?;
            dump_funcs(output, &funcs)?;
        }
        Generate::Bitcode => {
            let t0 = SystemTime::now();
            let output = args.output.clone().unwrap_or("a.bc".to_string());
            let machine = LlvmCompilingMachine::load(
                &output,
                &code,
                &args.symbol_prefix,
                &estimate_cycles,
                args.debug_info,
            )?;
            let bitcode = machine.bitcode(args.optimized())?;
            if args.time {
                let t1 = SystemTime::now();
                let duration = t1.duration_since(t0).expect("time went backwards");
                println!("Time to emit LLVM bitcode: {:?}", duration);
            }
            std::fs::write(&output, &bitcode)?;
        }
        Generate::Object => {
            let output = args.output.clone().unwrap_or("a.o".to_string());
            build_object(&code, &args, &output)?;
        }
        Generate::SharedLibrary => {
            let object_file = Builder::new().suffix(".o").tempfile()?;
            let object_path = object_file.path().to_str().expect("tempfile");
            build_object(&code, &args, object_path)?;
            let output = args.output.unwrap_or("a.so".to_string());
            build_shared_library(object_path, &output)?;
        }
        Generate::RunResult => {
            let object_file = Builder::new().suffix(".o").tempfile()?;
            let object_path = object_file.path().to_str().expect("tempfile");
            build_object(&code, &args, object_path)?;
            let library_file = Builder::new().suffix(".so").tempfile()?;
            let library_path = library_file.path().to_str().expect("tempfile");
            build_shared_library(object_path, library_path)?;

            let dl_symbols = DlSymbols::new(library_path, &args.symbol_prefix)?;
            let aot_symbols = &dl_symbols.aot_symbols;
            let core_machine =
                DefaultMachineBuilder::new(LlvmAotCoreMachine::new(args.memory_size)?)
                    .instruction_cycle_func(Box::new(estimate_cycles))
                    .syscall(Box::new(DebugSyscall {}))
                    .syscall(Box::new(TimeSyscall::new()))
                    .build();
            let mut machine = LlvmAotMachine::new_with_machine(core_machine, &aot_symbols)?;
            machine.set_max_cycles(args.max_cycles);
            let run_args: Vec<Bytes> = args
                .run_args
                .iter()
                .map(|a| Bytes::from(a.as_bytes().to_vec()))
                .collect();
            let t0 = SystemTime::now();
            machine.load_program(&code, &run_args)?;
            let exit = machine.run();
            if args.time {
                let t1 = SystemTime::now();
                let duration = t1.duration_since(t0).expect("time went backwards");
                println!("Time to run program: {:?}", duration);
            }
            match &exit {
                Err(e) => {
                    println!("Run error encountered: {:?}", e);
                    println!("Machine: {}", machine.machine);
                }
                Ok(0) => println!("Cycles: {}", machine.machine.cycles()),
                Ok(i) => {
                    println!("Non-zero exit code: {}", i);
                    println!("Machine: {}", machine.machine);
                }
            };
            std::process::exit(exit.unwrap_or(-1) as i32);
        }
    }

    Ok(())
}

fn build_object(code: &Bytes, args: &Args, output: &str) -> Result<(), Error> {
    let t0 = SystemTime::now();
    let machine = LlvmCompilingMachine::load(
        output,
        &code,
        &args.symbol_prefix,
        &estimate_cycles,
        args.debug_info,
    )?;
    let object = machine.aot(args.optimized())?;
    if args.time {
        let t1 = SystemTime::now();
        let duration = t1.duration_since(t0).expect("time went backwards");
        println!("Time to generate object: {:?}", duration);
    }
    std::fs::write(output, &object)?;
    Ok(())
}

fn build_shared_library(input: &str, output: &str) -> Result<(), Error> {
    let mut cmd = Command::new(find_linker());
    cmd.arg("-shared").arg("-o").arg(output).arg(input);
    let output = cmd
        .output()
        .map_err(|e| Error::External(format!("Executing error: {:?}", e)))?;
    if !output.status.success() {
        return Err(Error::External(format!(
            "Error executing gcc: {:?}, stdout: {}, stderr: {}",
            output.status.code(),
            std::str::from_utf8(&output.stdout).unwrap_or("non UTF-8 data"),
            std::str::from_utf8(&output.stderr).unwrap_or("non UTF-8 data"),
        )));
    }

    Ok(())
}

fn dump_funcs(mut o: Box<dyn Write>, funcs: &[Func]) -> Result<(), Error> {
    for func in funcs {
        write!(
            o,
            "Func: {} at 0x{:x}-0x{:x}\n",
            func.force_name(true),
            func.range.start,
            func.range.end
        )?;

        for block in &func.basic_blocks {
            write!(
                o,
                "  Basic block (insts: {}) 0x{:x}-0x{:x}:\n",
                block.insts, block.range.start, block.range.end,
            )?;
            write!(
                o,
                "  Possible targets after block: {}\n",
                block
                    .possible_targets()
                    .iter()
                    .map(|a| format!("0x{:x}", a))
                    .collect::<Vec<String>>()
                    .join(", ")
            )?;
            for (i, batch) in block.write_batches.iter().enumerate() {
                write!(o, "    Write batch {}\n", i)?;
                for write in batch {
                    write!(o, "      {}\n", write)?;
                }
            }
            dump_control(&mut o, &block.control, &funcs)?;
            if let Some(last_writes) = block.control.writes() {
                for write in last_writes {
                    write!(o, "      {}\n", write)?;
                }
            }
            write!(o, "\n")?;
        }
    }
    Ok(())
}

fn dump_control(o: &mut Box<dyn Write>, control: &Control, funcs: &[Func]) -> Result<(), Error> {
    match control {
        Control::Call { address, .. } => {
            if let Some(func) = funcs.iter().find(|f| f.range.start == *address) {
                write!(o, "    {} ({})", control, func.force_name(true))?;
                return Ok(());
            }
        }
        _ => (),
    };
    write!(o, "    {}\n", control)?;
    Ok(())
}

fn find_linker() -> PathBuf {
    for candidate in ["ld.lld-15", "ld", "ld.lld"] {
        if let Ok(path) = which(candidate) {
            info!("Use linker from {:?}", path);
            return path;
        }
    }
    panic!("Cannot find a linker to use! Please install either lld from LLVM or ld from GNU toolchain!");
}
