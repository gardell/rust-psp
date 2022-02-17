#![recursion_limit = "256"]
#![no_std]
#![feature(asm_experimental_arch)]
#![no_main]

psp::module!("vfpu_test", 1, 1);

fn vfpu_add(a: i32, b: i32) -> i32 {
    let ret_val;

    unsafe {
        psp::vfpu_asm! (
            // Convert `a` to float
            "mtc1 {a}, {ftmp}",
            "nop",
            "cvt.s.w {ftmp}, {ftmp}",
            "mfc1 {a}, {ftmp}",
            "nop",

            // Convert `b` to float
            "mtc1 {b}, {ftmp}",
            "nop",
            "cvt.s.w {ftmp}, {ftmp}",
            "mfc1 {b}, {ftmp}",
            "nop",

            // Perform addition
            "mtv {a}, S000",
            "mtv {b}, S001",
            "vadd.s S000, S000, S001",
            "mfv {ret}, S000",

            // Convert result to `i32`
            "mtc1 {ret}, {ftmp}",
            "nop",
            "cvt.w.s {ftmp}, {ftmp}",
            "mfc1 {ret}, {ftmp}",
            "nop",

            ftmp = out(freg) _,
            a = inout(reg) a => _,
            b = inout(reg) b => _,
            ret = out(reg) ret_val,
            options(nostack, nomem),
        );
    }

    ret_val
}

fn psp_main() {
    psp::enable_home_button();

    // Enable the VFPU
    unsafe {
        use psp::sys::{self, ThreadAttributes};
        sys::sceKernelChangeCurrentThreadAttr(0, ThreadAttributes::VFPU);
    }

    psp::dprintln!("Testing VFPU...");
    psp::dprintln!("VFPU 123 + 4 = {}", vfpu_add(123, 4));

    let matrix = psp::sys::ScePspFMatrix4 {
        x: psp::sys::ScePspFVector4 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0,
        },
        y: psp::sys::ScePspFVector4 {
            x: 5.0,
            y: 6.0,
            z: 7.0,
            w: 8.0,
        },
        z: psp::sys::ScePspFVector4 {
            x: 9.0,
            y: 10.0,
            z: 11.0,
            w: 12.0,
        },
        w: psp::sys::ScePspFVector4 {
            x: 13.0,
            y: 14.0,
            z: 15.0,
            w: 16.0,
        },
    };

    let vector = psp::sys::ScePspFVector4 {
        x: 17.0,
        y: 18.0,
        z: 19.0,
        w: 20.0,
    };

    let mut result: psp::sys::ScePspFVector4 = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
    let mut result_transposed: psp::sys::ScePspFVector4 = unsafe { core::mem::MaybeUninit::uninit().assume_init() };

    unsafe {
        psp::vfpu_asm!(
            "lv.q C000, 0({matrix})",
            "lv.q C010, 16({matrix})",
            "lv.q C020, 32({matrix})",
            "lv.q C030, 48({matrix})",

            "lv.q C100, 0({vector})",

            "vtfm4.q C110, M000, C100",

            "sv.q C110, 0({result})",

            "vtfm4.q C110, E000, C100",

            "sv.q C110, 0({result_transposed})",

            result = in(reg) (&mut result),
            result_transposed = in(reg) (&mut result_transposed),
            matrix = in(reg) (&matrix),
            vector = in(reg) (&vector),
            options(nostack),
        );
    }

    assert_eq!((result.x, result.y, result.z, result.w), (538.0, 612.0, 686.0, 760.0));
    assert_eq!((result_transposed.x, result_transposed.y, result_transposed.z, result_transposed.w), (190.0, 486.0, 782.0, 1078.0));
}
