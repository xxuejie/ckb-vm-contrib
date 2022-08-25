/// Common syscalls that can be reused in multiple places
use ckb_vm::{
    registers::{A0, A7},
    Memory, Register, SupportMachine, Syscalls,
};
use std::time::SystemTime;

/// Syscall handler for printing debugging information to host-side STDOUT.
/// It uses a simpler design than using STDOUT in CKB-VM itself.
pub struct DebugSyscall {}

impl<Mac: SupportMachine> Syscalls<Mac> for DebugSyscall {
    fn initialize(&mut self, _machine: &mut Mac) -> Result<(), ckb_vm::error::Error> {
        Ok(())
    }

    fn ecall(&mut self, machine: &mut Mac) -> Result<bool, ckb_vm::error::Error> {
        let code = &machine.registers()[A7];
        if code.to_i32() != 2177 {
            return Ok(false);
        }

        let mut addr = machine.registers()[A0].to_u64();
        let mut buffer = Vec::new();

        loop {
            let byte = machine
                .memory_mut()
                .load8(&Mac::REG::from_u64(addr))?
                .to_u8();
            if byte == 0 {
                break;
            }
            buffer.push(byte);
            addr += 1;
        }

        let s = String::from_utf8(buffer).unwrap();
        println!("{:?}", s);

        Ok(true)
    }
}

/// Syscall handler for providing time information since boot to CKB-VM.
/// While this is not exactly the time for the outside world, it can be used
/// to build benchmarks within CKB-VM.
pub struct TimeSyscall {
    boot_time: SystemTime,
}

impl TimeSyscall {
    pub fn new() -> Self {
        Self {
            boot_time: SystemTime::now(),
        }
    }
}

impl<Mac: SupportMachine> Syscalls<Mac> for TimeSyscall {
    fn initialize(&mut self, _machine: &mut Mac) -> Result<(), ckb_vm::error::Error> {
        Ok(())
    }

    fn ecall(&mut self, machine: &mut Mac) -> Result<bool, ckb_vm::error::Error> {
        let code = &machine.registers()[A7];
        // Zlib::crc32("time") % 10000
        if code.to_i32() != 9285 {
            return Ok(false);
        }

        let now = SystemTime::now();
        let d = now
            .duration_since(self.boot_time.clone())
            .expect("clock goes backwards");

        machine.set_register(A0, Mac::REG::from_u64(d.as_nanos() as u64));

        Ok(true)
    }
}
