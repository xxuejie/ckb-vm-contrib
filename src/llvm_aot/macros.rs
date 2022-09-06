macro_rules! u {
    ($call:expr) => {
        unsafe { $call }
    };
}

macro_rules! assert_llvm_call {
    ($call:expr, $msg:expr) => {
        if u!($call) != 0 {
            return Err(ckb_vm::Error::External(format!(
                "LLVM call failed: {}",
                $msg,
            )));
        }
    };
}

macro_rules! assert_llvm_create {
    ($call:expr, $msg:expr) => {{
        let __p__ = u!($call);
        if __p__.is_null() {
            return Err(ckb_vm::Error::External(format!(
                "LLVM call failed: {}",
                $msg,
            )));
        }
        __p__
    }};
}

pub fn build_error_message(msg: *mut std::os::raw::c_char) -> String {
    if msg.is_null() {
        "no error message provided!".to_string()
    } else {
        let m = match u!(std::ffi::CStr::from_ptr(msg)).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => "error message is not in utf-8 format!".to_string(),
        };
        u!(llvm_sys::core::LLVMDisposeMessage(msg));
        m
    }
}

macro_rules! assert_llvm_err {
    ($func:expr, $($args:expr),+) => {
        {
            let mut __err__ = u!(core::mem::zeroed());
            if u!($func($($args),+, &mut __err__)) != 0 {
                return Err(ckb_vm::Error::External(format!("LLVMError: {}", build_error_message(__err__))));
            }
        }
    }
}

macro_rules! assert_llvm_named_err {
    ($call:expr, $err:expr) => {
        if u!($call) != 0 {
            return Err(ckb_vm::Error::External(format!(
                "LLVMError: {}",
                build_error_message($err)
            )));
        }
    };
}

pub(crate) use assert_llvm_call;
pub(crate) use assert_llvm_create;
pub(crate) use assert_llvm_err;
pub(crate) use assert_llvm_named_err;
pub(crate) use u;
