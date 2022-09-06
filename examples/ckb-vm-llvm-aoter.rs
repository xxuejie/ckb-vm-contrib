use ckb_vm::{machine::DefaultMachineBuilder, Bytes, Error};
use ckb_vm_contrib::{
    llvm_aot::{DlSymbols, LlvmAotCoreMachine, LlvmAotMachine, LlvmCompilingMachine},
    syscalls::{DebugSyscall, TimeSyscall},
};
use clap::{Parser, ValueEnum};
use std::process::Command;
use std::time::SystemTime;
use tempfile::Builder;

#[derive(ValueEnum, Clone, Debug)]
enum Generate {
    Hash,
    Bitcode,
    Object,
    SharedLibrary,
    RunResult,
}

#[derive(Parser, Debug)]
#[clap(author, version, about = "A simple runner for ckb-vm's LLVM AOT engine", long_about = None)]
struct Args {
    /// Time critical operations when turned on
    #[clap(short, long)]
    time: bool,

    /// Optimize the generated code
    #[clap(long, default_value = "true")]
    optimize: bool,

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
    #[clap(short, long, default_value = "compiled.o")]
    output: String,

    /// Input file path
    #[clap(short, long, required = true)]
    input: String,

    /// Arguments used to run the program
    #[clap()]
    run_args: Vec<String>,
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
        Generate::Bitcode => {
            let t0 = SystemTime::now();
            let machine = LlvmCompilingMachine::load(&args.input, &code, &args.symbol_prefix)?;
            let bitcode = machine.bitcode(args.optimize)?;
            if args.time {
                let t1 = SystemTime::now();
                let duration = t1.duration_since(t0).expect("time went backwards");
                println!("Time to emit LLVM bitcode: {:?}", duration);
            }
            std::fs::write(&args.output, &bitcode)?;
        }
        Generate::Object => {
            build_object(&code, &args, &args.output)?;
        }
        Generate::SharedLibrary => {
            let object_file = Builder::new().suffix(".o").tempfile()?;
            let object_path = object_file.path().to_str().expect("tempfile");
            build_object(&code, &args, object_path)?;
            build_shared_library(object_path, &args.output)?;
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
                    .syscall(Box::new(DebugSyscall {}))
                    .syscall(Box::new(TimeSyscall::new()))
                    .build();
            let mut machine = LlvmAotMachine::new_with_machine(core_machine, &aot_symbols)?;
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
            let exit_code = exit?;
            if exit_code != 0 {
                println!("Non-zero exit code: {}\n{}", exit_code, machine.machine);
            }
            std::process::exit(exit_code as i32);
        }
    }

    Ok(())
}

fn build_object(code: &Bytes, args: &Args, output: &str) -> Result<(), Error> {
    let t0 = SystemTime::now();
    let machine = LlvmCompilingMachine::load(&args.input, &code, &args.symbol_prefix)?;
    let object = machine.aot(args.optimize)?;
    if args.time {
        let t1 = SystemTime::now();
        let duration = t1.duration_since(t0).expect("time went backwards");
        println!("Time to generate object: {:?}", duration);
    }
    std::fs::write(output, &object)?;
    Ok(())
}

fn build_shared_library(input: &str, output: &str) -> Result<(), Error> {
    let mut cmd = Command::new("gcc");
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
