use super::{AotMemory, Hint};
use ckb_vm::{
    ckb_vm_definitions::RISCV_PAGESIZE,
    memory::{round_page_down, round_page_up, FLAG_EXECUTABLE},
    Bytes, Error,
};
use region::{self, alloc, query_range, Allocation, Protection, Region};

pub struct MmapMemory {
    pub allocation: Allocation,
    pub cached_region: Option<Region>,
    memory: *mut u8,
}

impl MmapMemory {
    pub fn create(memory_size: usize) -> Result<Self, Error> {
        if memory_size % RISCV_PAGESIZE != 0 {
            return Err(Error::External(
                "Memory size must be a multiple of 4KB!".to_string(),
            ));
        }

        let mut allocation = alloc(memory_size, Protection::READ_WRITE)
            .map_err(|e| Error::External(format!("region alloc error: {}", e)))?;
        let memory = allocation.as_mut_ptr();

        Ok(Self {
            allocation,
            memory,
            cached_region: None,
        })
    }

    // TODO: right now we leverage host OS to mark the pages as executable,
    // is it secure to mark RISC-V code executable on a x86/arm CPU?
    fn inner_execute_check(&mut self, addr: u64, len: u64) -> Result<(), Error> {
        let len = len as usize;
        let host_addr = self.memory.wrapping_offset(addr as isize);
        if let Some(region) = self.cached_region {
            let r = region.as_range();
            if host_addr as usize >= r.start && host_addr as usize + len <= r.end {
                return Ok(());
            }
        }
        let ranges: Vec<Region> = query_range(host_addr, len)
            .map_err(e)?
            .try_fold(vec![], |mut acc, r| match r {
                Ok(r) => {
                    acc.push(r);
                    Ok(acc)
                }
                Err(e) => Err(e),
            })
            .map_err(e)?;
        if ranges.iter().any(|r| !r.is_executable()) {
            return Err(Error::External(format!(
                "addr: {:x} executing on non executable pages!",
                addr
            )));
        }
        if ranges.len() == 1 {
            self.cached_region = Some(ranges[0]);
        }
        Ok(())
    }
}

impl AotMemory for MmapMemory {
    fn memory_ptr(&mut self) -> *mut u8 {
        self.memory
    }

    fn check_permissions(&mut self, hints: &[Hint]) -> Result<(), Error> {
        // For mmap memory load / store check is a no-op since OS will help
        // us check memory permissions
        for hint in hints {
            if hint.is_execute() {
                self.inner_execute_check(hint.offset, hint.size)?;
            }
        }
        Ok(())
    }

    fn memory_size(&self) -> usize {
        self.allocation.len()
    }

    fn init_pages(
        &mut self,
        addr: u64,
        size: u64,
        flags: u8,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error> {
        if round_page_down(addr) != addr || round_page_up(size) != size {
            return Err(Error::MemPageUnalignedAccess);
        }
        let memory_size = self.allocation.len();
        if addr > memory_size as u64
            || size > memory_size as u64
            || addr + size > memory_size as u64
            || offset_from_addr > size
        {
            return Err(Error::MemOutOfBound);
        }
        let host_addr = self.memory.wrapping_offset(addr as isize);
        let mut ranges = query_range(host_addr, size as usize).map_err(e)?;
        if ranges.any(|r| r.is_ok() && r.unwrap().is_executable()) {
            return Err(Error::MemWriteOnFreezedPage);
        }
        self.fill_page_data(addr, size, source, offset_from_addr)?;
        let protection = if flags & FLAG_EXECUTABLE != 0 {
            Protection::READ_EXECUTE
        } else {
            Protection::READ_WRITE
        };
        unsafe { region::protect(host_addr, size as usize, protection) }.map_err(e)?;
        self.cached_region = None;
        Ok(())
    }
}

fn e(error: region::Error) -> Error {
    Error::External(format!("region error: {}", error))
}
