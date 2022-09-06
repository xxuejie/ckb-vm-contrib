use super::{runner::LlvmAotCoreMachineData, utils::cs};
use ckb_vm::Error;
use libc::{c_void, dlclose, dlerror, dlopen, dlsym, RTLD_NOW};
use std::ffi::{CStr, CString};
use std::mem::transmute;
use std::ptr;
use std::slice::from_raw_parts;

pub type EntryFunctionType =
    unsafe extern "C" fn(machine: *mut LlvmAotCoreMachineData, host_target: u64) -> u8;

#[repr(C)]
pub struct AddressTableEntry {
    pub riscv_addr: u64,
    pub host_addr: u64,
}

pub struct AotSymbols<'a> {
    pub code_hash: &'a [u8],
    pub address_table: &'a [AddressTableEntry],
    pub entry_function: EntryFunctionType,
}

/// Data structure for loading AotSymbols from dynamic shared library.
pub struct DlSymbols<'a> {
    pub aot_symbols: AotSymbols<'a>,

    handle: *mut c_void,
}

impl<'a> Drop for DlSymbols<'a> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { dlclose(self.handle) };
        }
    }
}

impl<'a> DlSymbols<'a> {
    pub fn new(path: &str, symbol_prefix: &str) -> Result<Self, Error> {
        let c_path = CString::new(path).map_err(|_e| {
            Error::External(
                "File path cannot be represented as null terminated string!".to_string(),
            )
        })?;

        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(build_error_from_dlerror());
        }

        let code_hash_sym = unsafe {
            dlsym(
                handle,
                cs(&format!("{}____code_hash____", symbol_prefix))?.as_ptr(),
            )
        };
        if code_hash_sym.is_null() {
            let e = build_error_from_dlerror();
            unsafe { dlclose(handle) };
            return Err(e);
        }
        let address_table_sym = unsafe {
            dlsym(
                handle,
                cs(&format!("{}____address_table____", symbol_prefix))?.as_ptr(),
            )
        };
        if address_table_sym.is_null() {
            let e = build_error_from_dlerror();
            unsafe { dlclose(handle) };
            return Err(e);
        }
        let entry_sym = unsafe {
            dlsym(
                handle,
                cs(&format!("{}____entry____", symbol_prefix))?.as_ptr(),
            )
        };
        if entry_sym.is_null() {
            let e = build_error_from_dlerror();
            unsafe { dlclose(handle) };
            return Err(e);
        }

        let aot_symbols = create_aot_symbols(code_hash_sym, address_table_sym, entry_sym);

        Ok(Self {
            aot_symbols,
            handle,
        })
    }
}

/// Create AotSymbols data structure from pointers, there will be a lot of unsafe
/// used here.
fn create_aot_symbols<'a>(
    code_hash_sym: *const c_void,
    address_table_sym: *const c_void,
    entry_sym: *const c_void,
) -> AotSymbols<'a> {
    let code_hash = unsafe { from_raw_parts(code_hash_sym as *const u8, 32) };
    let address_table_addr = address_table_sym as *const u64;
    let address_table_size = unsafe { ptr::read(address_table_addr) } as usize;
    let address_table = unsafe {
        from_raw_parts(
            address_table_addr.offset(1) as *const AddressTableEntry,
            address_table_size,
        )
    };
    let entry_function = unsafe { transmute::<_, EntryFunctionType>(entry_sym) };

    AotSymbols {
        code_hash,
        address_table,
        entry_function,
    }
}

fn build_error_from_dlerror() -> Error {
    let e = unsafe { dlerror() };
    Error::External(if e.is_null() {
        "Unknown error loading shared library!".to_string()
    } else {
        match unsafe { CStr::from_ptr(e) }.to_str() {
            Ok(s) => format!("dlopen error: {}", s),
            Err(_e) => "dlerror message cannot be represented in UTF-8!".to_string(),
        }
    })
}

/// Handly macro to use when you statically link the generated AOT module into your binary.
#[macro_export]
macro_rules! derive_aot_symbols_from_static_globals {
    ($symbol_prefix:ident) => ($crate::llvm_aot::paste! {
        extern "C" {
            static [< $symbol_prefix ____code_hash____ >]: u8;
            static [< $symbol_prefix ____address_table____ >]: u64;
            fn [< $symbol_prefix ____entry____ >](machine: *mut $crate::llvm_aot::LlvmAotCoreMachineData, host_target: u64) -> u8;
        }

        pub fn create_aot_symbols() -> $crate::llvm_aot::AotSymbols<'static> {
            let code_hash = unsafe {
                core::slice::from_raw_parts(&[< $symbol_prefix ____code_hash____ >] as *const u8, 32)
            };

            let address_table_addr = unsafe {
                &[< $symbol_prefix ____address_table____ >] as *const u64
            };
            let size = unsafe { core::ptr::read(address_table_addr) } as usize;
            let address_table = unsafe {
                core::slice::from_raw_parts(
                    address_table_addr.offset(1) as *const $crate::llvm_aot::AddressTableEntry,
                    size,
                )
            };

            $crate::llvm_aot::AotSymbols {
                code_hash,
                address_table,
                entry_function: [< $symbol_prefix ____entry____ >],
            }
        }
    })
}

pub use derive_aot_symbols_from_static_globals;
