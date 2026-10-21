//go:build darwin

// darwin-only, like mem_test.go: these need a real Metal device, and elsewhere
// they would only be skips. An arm64 Mac runs them from a cross-compiled test binary.

package metal_test

import (
	"testing"

	"github.com/jitllm/jitllm/backend/metal"
)

// TestPackedDotIntrinsic asks the Metal compiler, rather than the internet,
// whether Apple exposes a packed int8 dot product.
//
// The answer decides whether msl's kernel does 4x the arithmetic of the other
// backends'. It never fails: it records what the toolchain accepts.
func TestPackedDotIntrinsic(t *testing.T) {
	c, err := metal.Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()

	for _, cand := range []struct{ name, expr string }{
		{"dot4I8Packed", "dot4I8Packed(a, b, c)"},
		{"dot_4x8packed_i8i8_i32", "dot_4x8packed_i8i8_i32(a, b, c)"},
		{"simd_dot_i8", "simd_dot_i8(a, b, c)"},
		{"integer_dot_product", "integer_dot_product(a, b, c)"},
		{"explicit unpack (what msl/ emits)", "c + int(as_type<char4>(a).x) * int(as_type<char4>(b).x)"},
	} {
		src := `#include <metal_stdlib>
using namespace metal;
kernel void probe(device uint* p [[buffer(0)]], uint t [[thread_position_in_threadgroup]]) {
    uint a = p[0], b = p[1]; int c = 0;
    p[t] = uint(` + cand.expr + `);
}`
		k, err := c.Compile(src, "probe")
		if err != nil {
			t.Logf("  %-36s NOT available", cand.name)
			continue
		}
		k.Close()
		t.Logf("  %-36s AVAILABLE", cand.name)
	}
}

// TestSimdgroupMatrix asks what an Apple GPU offers that msl/ is NOT using.
//
// msl/ emits a register-tiled int8 kernel with no threadgroup memory and no
// matrix units, so its measured rate is a floor for that kernel shape, not the
// hardware's ceiling. Apple's matrix units are float, so the shape that suits
// this GPU is dequantize-to-fp16 and multiply (as MLX does). This records
// which primitives the toolchain compiles.
func TestSimdgroupMatrix(t *testing.T) {
	c, err := metal.Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()

	for _, cand := range []struct{ name, body string }{
		{"simdgroup_float8x8 multiply_accumulate", `
    simdgroup_float8x8 A, B, C;
    simdgroup_load(A, fa, 8);
    simdgroup_load(B, fa, 8);
    simdgroup_multiply_accumulate(C, A, B, C);
    simdgroup_store(C, fa, 8);`},
		{"simdgroup_half8x8 multiply_accumulate", `
    simdgroup_half8x8 A, B, C;
    simdgroup_load(A, ha, 8);
    simdgroup_load(B, ha, 8);
    simdgroup_multiply_accumulate(C, A, B, C);
    simdgroup_store(C, ha, 8);`},
		{"threadgroup memory + barrier", `
    threadgroup float tile[64];
    tile[t] = fa[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fa[t] = tile[63 - t];`},
		{"simd_shuffle / simd_sum", `
    fa[t] = simd_sum(fa[t]) + simd_shuffle(fa[t], 1);`},
	} {
		src := `#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device float* fa [[buffer(0)]],
                  device half* ha [[buffer(1)]],
                  uint t [[thread_position_in_threadgroup]]) {` + cand.body + `
}`
		k, err := c.Compile(src, "probe")
		if err != nil {
			t.Logf("  %-40s NOT available", cand.name)
			continue
		}
		k.Close()
		t.Logf("  %-40s AVAILABLE", cand.name)
	}
}

// TestSimdgroupMatrixForms asks the Metal compiler WHICH simdgroup_matrix
// forms exist, rather than which ones a header is rumoured to have.
//
// The earlier probe shows 8x8 float and half matrices exist, but not whether
// half operands may accumulate into float, nor whether other shapes exist;
// both decide what ir.TileType may admit here, and a wrong guess computes a
// plausible answer in the wrong precision. ir.TileType.Valid() refuses a half
// or int8 accumulator, so without the mixed form f16 tiles would be declined.
// It never fails; it records the toolchain's answer.
func TestSimdgroupMatrixForms(t *testing.T) {
	c, err := metal.Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()

	for _, cand := range []struct{ name, body string }{
		{"f32 8x8 load/mac/store", `
    simdgroup_matrix<float,8,8> a, b, d;
    simdgroup_load(a, f, 8); simdgroup_load(b, f, 8);
    d = simdgroup_matrix<float,8,8>(0);
    simdgroup_multiply_accumulate(d, a, b, d);
    simdgroup_store(d, f, 8);`},
		{"f16 8x8, half accumulator", `
    simdgroup_matrix<half,8,8> a, b, d;
    simdgroup_load(a, h, 8); simdgroup_load(b, h, 8);
    d = simdgroup_matrix<half,8,8>(0);
    simdgroup_multiply_accumulate(d, a, b, d);
    simdgroup_store(d, h, 8);`},
		{"MIXED: half operands, float accumulator", `
    simdgroup_matrix<half,8,8> a, b;
    simdgroup_matrix<float,8,8> d;
    simdgroup_load(a, h, 8); simdgroup_load(b, h, 8);
    d = simdgroup_matrix<float,8,8>(0);
    simdgroup_multiply_accumulate(d, a, b, d);
    simdgroup_store(d, f, 8);`},
		{"f32 8x16 (a non-square shape)", `
    simdgroup_matrix<float,8,16> a; simdgroup_load(a, f, 16); simdgroup_store(a, f, 16);`},
		{"f32 16x16", `
    simdgroup_matrix<float,16,16> a; simdgroup_load(a, f, 16); simdgroup_store(a, f, 16);`},
		{"int8 8x8 (char)", `
    simdgroup_matrix<char,8,8> a; simdgroup_load(a, i8, 8); simdgroup_store(a, i8, 8);`},
		{"int32 8x8 accumulator", `
    simdgroup_matrix<int,8,8> a; simdgroup_load(a, i32, 8); simdgroup_store(a, i32, 8);`},
		{"transposed load (the B operand's layout)", `
    simdgroup_matrix<float,8,8> a;
    simdgroup_load(a, f, 8, ulong2(0, 0), true);
    simdgroup_store(a, f, 8);`},
	} {
		src := `#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void probe(device float* f [[buffer(0)]], device half* h [[buffer(1)]],
                  device char* i8 [[buffer(2)]], device int* i32 [[buffer(3)]]) {` +
			cand.body + `
}`
		k, err := c.Compile(src, "probe")
		if err != nil {
			t.Logf("  %-42s NOT available", cand.name)
			continue
		}
		k.Close()
		t.Logf("  %-42s AVAILABLE", cand.name)
	}
}
