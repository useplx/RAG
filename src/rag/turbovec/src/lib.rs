
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

use turbovec::IdMapIndex;

pub const TV_OK: i32 = 0;
pub const TV_ERR_NULL: i32 = -1;
pub const TV_ERR_ARGUMENT: i32 = -2;
pub const TV_ERR_STATE: i32 = -3;
pub const TV_ERR_PANIC: i32 = -4;

thread_local! {
    static LAST_ERROR: RefCell<std::ffi::CString> =
        RefCell::new(std::ffi::CString::new("").unwrap());
}

fn set_error(message: impl Into<Vec<u8>>) {
    let text = std::ffi::CString::new(message)
        .unwrap_or_else(|_| std::ffi::CString::new("unrepresentable error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = text);
}

#[no_mangle]
pub extern "C" fn tv_last_error() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

fn guard<F: FnOnce() -> i32>(body: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => {
            set_error("panic no turbovec");
            TV_ERR_PANIC
        }
    }
}

#[no_mangle]
pub extern "C" fn tv_create(dim: u32, bit_width: u32) -> *mut IdMapIndex {
    let result = catch_unwind(|| {
        match IdMapIndex::new(dim as usize, bit_width as usize) {
            Ok(index) => Box::into_raw(Box::new(index)),
            Err(error) => {
                set_error(format!("{error:?}"));
                std::ptr::null_mut()
            }
        }
    });
    result.unwrap_or_else(|_| {
        set_error("panic em tv_create");
        std::ptr::null_mut()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tv_open(bytes: *const u8, len: usize) -> *mut IdMapIndex {
    if bytes.is_null() {
        set_error("tv_open received a null pointer");
        return std::ptr::null_mut();
    }
    let payload = std::slice::from_raw_parts(bytes, len);
    let result = catch_unwind(AssertUnwindSafe(|| match IdMapIndex::from_bytes(payload) {
        Ok(index) => {
            index.prepare();
            Box::into_raw(Box::new(index))
        }
        Err(error) => {
            set_error(format!("{error}"));
            std::ptr::null_mut()
        }
    }));
    result.unwrap_or_else(|_| {
        set_error("panic em tv_open");
        std::ptr::null_mut()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tv_close(index: *mut IdMapIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tv_add(
    index: *mut IdMapIndex,
    vectors: *const f32,
    count: u32,
    dim: u32,
    ids: *const u64,
) -> i32 {
    if index.is_null() || vectors.is_null() || ids.is_null() {
        set_error("tv_add received a null pointer");
        return TV_ERR_NULL;
    }
    let index = &mut *index;
    let total = (count as usize).saturating_mul(dim as usize);
    let values = std::slice::from_raw_parts(vectors, total);
    let keys = std::slice::from_raw_parts(ids, count as usize);

    guard(|| {
        let outcome = match index.dim_opt() {
            None => index.add_with_ids_2d(values, dim as usize, keys),
            Some(existing) if existing != dim as usize => {
                set_error(format!(
                    "dimension {dim} does not match the index ({existing})"
                ));
                return TV_ERR_ARGUMENT;
            }
            Some(_) => index.add_with_ids(values, keys),
        };

        match outcome {
            Ok(()) => TV_OK,
            Err(error) => {
                set_error(format!("{error:?}"));
                TV_ERR_ARGUMENT
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tv_remove(index: *mut IdMapIndex, id: u64) -> i32 {
    if index.is_null() {
        set_error("tv_remove received a null pointer");
        return TV_ERR_NULL;
    }
    let index = &mut *index;
    guard(|| i32::from(index.remove(id)))
}

#[no_mangle]
pub unsafe extern "C" fn tv_search(
    index: *const IdMapIndex,
    query: *const f32,
    dim: u32,
    k: u32,
    out_scores: *mut f32,
    out_ids: *mut u64,
    out_count: *mut u32,
) -> i32 {
    if index.is_null() || query.is_null() || out_scores.is_null()
        || out_ids.is_null() || out_count.is_null()
    {
        set_error("tv_search received a null pointer");
        return TV_ERR_NULL;
    }
    let index = &*index;
    let values = std::slice::from_raw_parts(query, dim as usize);

    guard(|| {
        let Some(existing) = index.dim_opt() else {
            set_error("empty index: no vectors were inserted");
            return TV_ERR_STATE;
        };
        if existing != dim as usize {
            set_error(format!(
                "query dimension ({dim}) does not match the index ({existing})"
            ));
            return TV_ERR_ARGUMENT;
        }

        match index.try_search(values, k as usize) {
            Ok(results) => {
                let scores = results.scores_for_query(0);
                let ids = results.ids_for_query(0);
                let written = scores.len().min(k as usize);
                std::ptr::copy_nonoverlapping(scores.as_ptr(), out_scores, written);
                std::ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, written);
                *out_count = written as u32;
                TV_OK
            }
            Err(error) => {
                set_error(format!("{error:?}"));
                TV_ERR_ARGUMENT
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn tv_to_bytes(
    index: *const IdMapIndex,
    out: *mut u8,
    capacity: usize,
) -> usize {
    if index.is_null() {
        set_error("tv_to_bytes received a null pointer");
        return 0;
    }
    let index = &*index;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let bytes = index.to_bytes();
        let needed = bytes.len();
        if capacity >= needed && !out.is_null() {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, needed);
        }
        needed
    }));
    result.unwrap_or_else(|_| {
        set_error("panic em tv_to_bytes");
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn tv_len(index: *const IdMapIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn tv_dim(index: *const IdMapIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).dim_opt().unwrap_or(0) as u32
}

#[no_mangle]
pub unsafe extern "C" fn tv_bit_width(index: *const IdMapIndex) -> u32 {
    if index.is_null() {
        return 0;
    }
    (*index).bit_width() as u32
}
