use ckb_vm::Error;
use std::ffi::CString;

pub fn cs(s: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|e| {
        Error::External(format!(
            "String {} cannot be converted to CString: {}",
            s, e
        ))
    })
}
