pub mod mmap;
pub mod plain;

use ckb_vm::{memory::memset, Bytes, Error};
use std::cmp::min;
use std::slice::from_raw_parts_mut;

pub const HINT_FLAG_WRITE: u64 = 0x1;
pub const HINT_FLAG_EXECUTABLE: u64 = 0x2;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Hint {
    pub offset: u64,
    pub size: u64,
    pub flags: u64,
}

impl Hint {
    pub fn is_write(&self) -> bool {
        (self.flags & HINT_FLAG_WRITE) != 0
    }

    pub fn is_execute(&self) -> bool {
        (self.flags & HINT_FLAG_EXECUTABLE) != 0
    }
}

pub trait AotMemory {
    fn memory_size(&self) -> usize;
    fn memory_ptr(&mut self) -> *mut u8;
    fn check_permissions(&mut self, hints: &[Hint]) -> Result<(), Error>;

    fn execute_check(&mut self, addr: u64, len: u64) -> Result<*const u8, Error> {
        let hints = [Hint {
            offset: addr,
            size: len,
            flags: HINT_FLAG_EXECUTABLE,
        }];
        self.check_permissions(&hints)?;
        Ok(self.memory_ptr().wrapping_offset(addr as isize))
    }

    // TODO: temporary code till Memory is refactored
    fn init_pages(
        &mut self,
        addr: u64,
        size: u64,
        flags: u8,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error>;

    fn store_byte(&mut self, addr: u64, size: u64, value: u8) -> Result<(), Error> {
        self.check_permissions(&[Hint {
            offset: addr,
            size,
            flags: HINT_FLAG_WRITE,
        }])?;
        let host_addr = self.memory_ptr().wrapping_offset(addr as isize);
        let mut dst = unsafe { from_raw_parts_mut(host_addr, size as usize) };
        memset(&mut dst, value);
        Ok(())
    }

    fn store_bytes(&mut self, addr: u64, value: &[u8]) -> Result<(), Error> {
        self.check_permissions(&[Hint {
            offset: addr,
            size: value.len() as u64,
            flags: HINT_FLAG_WRITE,
        }])?;
        let host_addr = self.memory_ptr().wrapping_offset(addr as isize);
        let dst = unsafe { from_raw_parts_mut(host_addr, value.len()) };
        dst.copy_from_slice(value);
        Ok(())
    }

    fn fill_page_data(
        &mut self,
        addr: u64,
        size: u64,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error> {
        let mut written_size = 0;
        if offset_from_addr > 0 {
            let real_size = min(size, offset_from_addr);
            self.store_byte(addr, real_size, 0)?;
            written_size += real_size;
        }
        if let Some(source) = source {
            let real_size = min(size - written_size, source.len() as u64);
            if real_size > 0 {
                self.store_bytes(addr + written_size, &source[0..real_size as usize])?;
                written_size += real_size;
            }
        }
        if written_size < size {
            self.store_byte(addr + written_size, size - written_size, 0)?;
        }
        Ok(())
    }
}
