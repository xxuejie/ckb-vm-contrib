use super::{AotMemory, Hint};
use ckb_vm::{
    ckb_vm_definitions::{MEMORY_FRAMESIZE, RISCV_PAGESIZE},
    memory::{round_page_down, round_page_up, FLAG_EXECUTABLE, FLAG_FREEZED, FLAG_WXORX_BIT},
    Bytes, Error,
};
use std::alloc::{alloc, dealloc, Layout};

// Plain memory that does manual bound checking, suitable for library usage.
pub struct PlainMemory {
    memory: *mut u8,
    flags: Vec<u8>,
    frames: Vec<u8>,
    layout: Layout,
}

impl PlainMemory {
    pub fn create(memory_size: usize) -> Result<Self, Error> {
        if memory_size % RISCV_PAGESIZE != 0 {
            return Err(Error::External(format!(
                "Memory size must be a multiple of {}!",
                RISCV_PAGESIZE
            )));
        }
        if memory_size % MEMORY_FRAMESIZE != 0 {
            return Err(Error::External(format!(
                "Memory size must be a multiple of {}!",
                MEMORY_FRAMESIZE
            )));
        }

        let pages = memory_size / RISCV_PAGESIZE;
        let frames = memory_size / MEMORY_FRAMESIZE;

        let layout = Layout::array::<u8>(memory_size).unwrap();
        let memory = unsafe { alloc(layout) };

        Ok(Self {
            memory,
            flags: vec![0; pages],
            frames: vec![0; frames],
            layout,
        })
    }

    // TODO: those should be moved to Memory trait impl once Memory
    // trait is refactored
    fn fetch_flag(&mut self, page: u64) -> Result<u8, Error> {
        let page = page as usize;
        if page < self.flags.len() {
            Ok(self.flags[page])
        } else {
            Err(Error::MemOutOfBound)
        }
    }

    fn set_flag(&mut self, page: u64, flag: u8) -> Result<(), Error> {
        let page = page as usize;
        if page < self.flags.len() {
            self.flags[page as usize] |= flag;
            Ok(())
        } else {
            Err(Error::MemOutOfBound)
        }
    }
}

impl Drop for PlainMemory {
    fn drop(&mut self) {
        unsafe { dealloc(self.memory, self.layout) }
    }
}

impl AotMemory for PlainMemory {
    fn memory_size(&self) -> usize {
        self.layout.size()
    }

    fn memory_ptr(&mut self) -> *mut u8 {
        self.memory
    }

    fn check_permissions(&mut self, hints: &[Hint]) -> Result<(), Error> {
        let memory_size = self.memory_size() as u64;
        let page_size = RISCV_PAGESIZE as u64;
        let frame_size = MEMORY_FRAMESIZE as u64;

        for hint in hints {
            let addr = hint.offset;
            let size = hint.size;

            // 1. Check if memory is out of bound
            if addr > memory_size
                || size > memory_size
                || addr > addr + size
                || addr + size > memory_size
            {
                return Err(Error::MemOutOfBound);
            }
            // 2. Initialize memory if necessary
            let mut frame = addr / frame_size;
            while frame * frame_size < addr + size {
                if self.frames[frame as usize] == 0 {
                    self.store_byte(addr, size, 0)?;
                    self.frames[frame as usize] = 1;
                }
                frame += 1;
            }
            // 3. Check for write permissions
            if hint.is_write() {
                let mut page = addr / page_size;
                while page * page_size < addr + size {
                    let flag = self.fetch_flag(page)?;
                    if flag & FLAG_WXORX_BIT == FLAG_EXECUTABLE {
                        return Err(Error::MemWriteOnExecutablePage);
                    }
                    page += 1;
                }
            }
            // 4. Check for executable permissions
            if hint.is_execute() {
                let mut page = addr / page_size;
                while page * page_size < addr + size {
                    let flag = self.fetch_flag(page)?;
                    if flag & FLAG_WXORX_BIT != FLAG_EXECUTABLE {
                        return Err(Error::MemWriteOnExecutablePage);
                    }
                    page += 1;
                }
            }
        }
        Ok(())
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
        if addr > self.memory_size() as u64
            || size > self.memory_size() as u64
            || addr + size > self.memory_size() as u64
            || offset_from_addr > size
        {
            return Err(Error::MemOutOfBound);
        }
        // We benchmarked the code piece here, using while loop this way is
        // actually faster than a for..in solution. The difference is roughly
        // 3% so we are keeping this version.
        let mut current_addr = addr;
        while current_addr < addr + size {
            let page = current_addr / RISCV_PAGESIZE as u64;
            if self.fetch_flag(page)? & FLAG_FREEZED != 0 {
                return Err(Error::MemWriteOnFreezedPage);
            }
            current_addr += RISCV_PAGESIZE as u64;
        }
        self.fill_page_data(addr, size, source, offset_from_addr)?;
        current_addr = addr;
        while current_addr < addr + size {
            let page = current_addr / RISCV_PAGESIZE as u64;
            self.set_flag(page, flags)?;
            current_addr += RISCV_PAGESIZE as u64;
        }
        Ok(())
    }
}
