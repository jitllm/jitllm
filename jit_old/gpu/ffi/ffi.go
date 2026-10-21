// Package ffi binds C functions without cgo, on top of goffi.
//
// goffi is libffi-shaped: prepare a CallInterface, then call with an array of
// argument pointers and a return slot. This wraps it in typed Go functions,
// such as func(*CUdevptr, uint64) CUresult, so the signature is checked by the
// Go compiler rather than a hand-written descriptor list.
//
// goffi replaced purego (about three times cheaper per call, and well under
// 1% of a token, so it is not a throughput change). The two cannot coexist:
// each vendors its own internal/fakecgo, and linking both gives "duplicated
// definition of symbol _cgo_init".
package ffi

import (
	"fmt"
	"reflect"
	"sync/atomic"
	"unsafe"

	gffi "github.com/go-webgpu/goffi/ffi"
	"github.com/go-webgpu/goffi/types"
)

// None is the return type of a C function that returns void.
type None struct{}

// Lib is an open shared library.
type Lib struct{ h unsafe.Pointer }

// Open loads the first of names that exists.
func Open(names ...string) (*Lib, error) {
	var last error
	for _, n := range names {
		h, err := gffi.LoadLibrary(n)
		if err == nil {
			return &Lib{h: h}, nil
		}
		last = err
	}
	return nil, fmt.Errorf("ffi: none of %v could be loaded: %w", names, last)
}

func (l *Lib) Close() error { return gffi.FreeLibrary(l.h) }

// Has reports whether the library exports name.
//
// Binding panics on a missing symbol (a wrong signature or a typo is a startup
// bug), so an optional entry point, such as CUDA's graph API on an old driver,
// is checked with Has first.
func (l *Lib) Has(name string) bool {
	_, err := gffi.GetSymbol(l.h, name)
	return err == nil
}

type prepared struct {
	cif types.CallInterface
	fn  unsafe.Pointer
}

// call hands the prepared signature, the return slot and the argument array to
// goffi. Every one of them is in a heap frame (see frames), never on a stack.
func (p *prepared) call(rv unsafe.Pointer, av []unsafe.Pointer) {
	gffi.CallFunction(&p.cif, p.fn, rv, av)
}

// A call's arguments, its return slot and its argument array live in a frame
// on the heap, one of a few each binding holds, and are copied in and out by
// value.
//
// goffi takes the address of every argument and of the return slot, and its
// CallFunction reaches the trampoline through an interface
// (arch.Registry.Caller), so the compiler must assume it keeps them: built in
// the wrapper's frame, each call would allocate them. Hiding them from escape
// analysis (a uintptr round trip) kept them on the stack, and that is unsound,
// not only on paper: goffi turns some of those addresses into integers before
// the C call -- a struct over 16 bytes passed by reference on arm64 (Metal's
// MTLSize), a struct returned through x8 or a hidden first argument, any
// pointer-valued argument -- and then runs splittable Go (its argument
// packing, a sync.Pool, a defer) before runtime.cgocall leaves the goroutine
// stack. A stack moved in that window, grown or shrunk by the collector,
// leaves C reading and writing the old one. TestCallSurvivesStackGrowthAndGC
// catches exactly that.
//
// A heap object does not move, and a Go pointer an argument carries stays
// visible to the collector in it. A frame is taken by a compare-and-swap; a
// call finding every frame busy allocates its own, so concurrency is never
// refused, only paid for past nframes concurrent calls of one entry point.
// TestCallArgumentsStayOnTheStack gates the uncontended path at zero
// allocations.
const nframes = 8

type frameInit[F any] interface {
	*F
	init()
}

type frames[F any] struct {
	f    [nframes]F
	busy [nframes]atomic.Bool
}

func newFrames[F any, PF frameInit[F]]() *frames[F] {
	s := new(frames[F])
	for i := range s.f {
		PF(&s.f[i]).init()
	}
	return s
}

// take returns a free frame and its index, or a fresh frame and -1.
func take[F any, PF frameInit[F]](s *frames[F]) (PF, int) {
	for i := range s.busy {
		if !s.busy[i].Load() && s.busy[i].CompareAndSwap(false, true) {
			return &s.f[i], i
		}
	}
	f := PF(new(F))
	f.init()
	return f, -1
}

func (s *frames[F]) release(i int) {
	if i >= 0 {
		s.busy[i].Store(false)
	}
}

// desc maps a Go type to goffi's descriptor.
//
// Only what the GPU runtimes use. An unknown type panics at bind time, once at
// startup, rather than making a silently wrong call.
func desc(t reflect.Type) *types.TypeDescriptor {
	if t == nil {
		return types.VoidTypeDescriptor
	}
	switch t.Kind() {
	case reflect.Struct:
		if t.NumField() == 0 {
			return types.VoidTypeDescriptor
		}
		// Structs by value are needed: Metal passes two MTLSize (24 bytes,
		// where arm64 switches to a pointer to a copy). goffi applies the
		// ABI rule from the member layout.
		m := make([]*types.TypeDescriptor, t.NumField())
		for i := range m {
			m[i] = desc(t.Field(i).Type)
		}
		return &types.TypeDescriptor{Kind: types.StructType, Members: m}
	// The 8-bit cases exist for Objective-C's BOOL: arm64 returns it in the low
	// bits of x0 with the rest unspecified, so a wider binding reads garbage.
	// goffi writes back exactly the descriptor's size.
	case reflect.Uint8:
		return types.UInt8TypeDescriptor
	case reflect.Int8:
		return types.SInt8TypeDescriptor
	case reflect.Int32:
		return types.SInt32TypeDescriptor
	case reflect.Uint32:
		return types.UInt32TypeDescriptor
	case reflect.Int64:
		return types.SInt64TypeDescriptor
	case reflect.Uint64, reflect.Uintptr:
		return types.UInt64TypeDescriptor
	case reflect.Float32:
		return types.FloatTypeDescriptor
	case reflect.Float64:
		return types.DoubleTypeDescriptor
	case reflect.Ptr, reflect.UnsafePointer:
		return types.PointerTypeDescriptor
	}
	panic("ffi: no descriptor for " + t.String())
}

func descOf[T any]() *types.TypeDescriptor {
	var v T
	return desc(reflect.TypeOf(v))
}

func prep(l *Lib, name string, ret *types.TypeDescriptor, args ...*types.TypeDescriptor) *prepared {
	sym, err := gffi.GetSymbol(l.h, name)
	if err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	p := &prepared{fn: sym}
	if err := gffi.PrepareCallInterface(&p.cif, types.DefaultConvention(), ret, args); err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	return p
}

// Fn0 binds a C function of 0 argument(s).
func Fn0[R any](l *Lib, name string) func() R {
	return bind0[R](prep(l, name, descOf[R]()))
}

// Fn1 binds a C function of 1 argument(s).
func Fn1[R any, A0 any](l *Lib, name string) func(A0) R {
	return bind1[R, A0](prep(l, name, descOf[R](), descOf[A0]()))
}

// Fn2 binds a C function of 2 argument(s).
func Fn2[R any, A0 any, A1 any](l *Lib, name string) func(A0, A1) R {
	return bind2[R, A0, A1](prep(l, name, descOf[R](), descOf[A0](), descOf[A1]()))
}

// Fn3 binds a C function of 3 argument(s).
func Fn3[R any, A0 any, A1 any, A2 any](l *Lib, name string) func(A0, A1, A2) R {
	return bind3[R, A0, A1, A2](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2]()))
}

// Fn4 binds a C function of 4 argument(s).
func Fn4[R any, A0 any, A1 any, A2 any, A3 any](l *Lib, name string) func(A0, A1, A2, A3) R {
	return bind4[R, A0, A1, A2, A3](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3]()))
}

// Fn5 binds a C function of 5 argument(s).
func Fn5[R any, A0 any, A1 any, A2 any, A3 any, A4 any](l *Lib, name string) func(A0, A1, A2, A3, A4) R {
	return bind5[R, A0, A1, A2, A3, A4](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4]()))
}

// Fn6 binds a C function of 6 argument(s).
func Fn6[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5) R {
	return bind6[R, A0, A1, A2, A3, A4, A5](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5]()))
}

// Fn7 binds a C function of 7 argument(s).
func Fn7[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6) R {
	return bind7[R, A0, A1, A2, A3, A4, A5, A6](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6]()))
}

// Fn8 binds a C function of 8 argument(s).
func Fn8[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7) R {
	return bind8[R, A0, A1, A2, A3, A4, A5, A6, A7](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7]()))
}

// Fn9 binds a C function of 9 argument(s).
func Fn9[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8) R {
	return bind9[R, A0, A1, A2, A3, A4, A5, A6, A7, A8](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8]()))
}

// Fn10 binds a C function of 10 argument(s).
func Fn10[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) R {
	return bind10[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8](), descOf[A9]()))
}

// Fn11 binds a C function of 11 argument(s).
func Fn11[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any, A10 any](l *Lib, name string) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) R {
	return bind11[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10](prep(l, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3](), descOf[A4](), descOf[A5](), descOf[A6](), descOf[A7](), descOf[A8](), descOf[A9](), descOf[A10]()))
}

// Entry points that are not exported by name.

// Fn3At is Fn4At for a 3-argument entry point, such as
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
func Fn3At[R any, A0 any, A1 any, A2 any](fn unsafe.Pointer, name string) func(A0, A1, A2) R {
	return bind3[R, A0, A1, A2](prepAt(fn, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2]()))
}

// Fn4At binds a C function of 4 arguments held as an address rather than a
// symbol name. The Vulkan loader exports core entry points but not extension
// ones, which are reachable only through its dispatch table
// (vkGetDeviceProcAddr). The caller vouches for the pointer; a nil fn panics
// here rather than jumping to zero later.
func Fn4At[R any, A0 any, A1 any, A2 any, A3 any](fn unsafe.Pointer, name string) func(A0, A1, A2, A3) R {
	return bind4[R, A0, A1, A2, A3](prepAt(fn, name, descOf[R](), descOf[A0](), descOf[A1](), descOf[A2](), descOf[A3]()))
}

type frame0[R any] struct {
	r  R
	ps [0]unsafe.Pointer
}

func (f *frame0[R]) init() { f.ps = [0]unsafe.Pointer{} }

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame0[R]) clear() {
	var zr R
	f.r = zr
}

func bind0[R any](p *prepared) func() R {
	s := newFrames[frame0[R]]()
	return func() R {
		f, i := take(s)
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame1[R any, A0 any] struct {
	r  R
	a0 A0
	ps [1]unsafe.Pointer
}

func (f *frame1[R, A0]) init() { f.ps = [1]unsafe.Pointer{unsafe.Pointer(&f.a0)} }

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame1[R, A0]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
}

func bind1[R any, A0 any](p *prepared) func(A0) R {
	s := newFrames[frame1[R, A0]]()
	return func(a0 A0) R {
		f, i := take(s)
		f.a0 = a0
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame2[R any, A0 any, A1 any] struct {
	r  R
	a0 A0
	a1 A1
	ps [2]unsafe.Pointer
}

func (f *frame2[R, A0, A1]) init() {
	f.ps = [2]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame2[R, A0, A1]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
}

func bind2[R any, A0 any, A1 any](p *prepared) func(A0, A1) R {
	s := newFrames[frame2[R, A0, A1]]()
	return func(a0 A0, a1 A1) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame3[R any, A0 any, A1 any, A2 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	ps [3]unsafe.Pointer
}

func (f *frame3[R, A0, A1, A2]) init() {
	f.ps = [3]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame3[R, A0, A1, A2]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
}

func bind3[R any, A0 any, A1 any, A2 any](p *prepared) func(A0, A1, A2) R {
	s := newFrames[frame3[R, A0, A1, A2]]()
	return func(a0 A0, a1 A1, a2 A2) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame4[R any, A0 any, A1 any, A2 any, A3 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	ps [4]unsafe.Pointer
}

func (f *frame4[R, A0, A1, A2, A3]) init() {
	f.ps = [4]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame4[R, A0, A1, A2, A3]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
}

func bind4[R any, A0 any, A1 any, A2 any, A3 any](p *prepared) func(A0, A1, A2, A3) R {
	s := newFrames[frame4[R, A0, A1, A2, A3]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame5[R any, A0 any, A1 any, A2 any, A3 any, A4 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	ps [5]unsafe.Pointer
}

func (f *frame5[R, A0, A1, A2, A3, A4]) init() {
	f.ps = [5]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame5[R, A0, A1, A2, A3, A4]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
}

func bind5[R any, A0 any, A1 any, A2 any, A3 any, A4 any](p *prepared) func(A0, A1, A2, A3, A4) R {
	s := newFrames[frame5[R, A0, A1, A2, A3, A4]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame6[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	a5 A5
	ps [6]unsafe.Pointer
}

func (f *frame6[R, A0, A1, A2, A3, A4, A5]) init() {
	f.ps = [6]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame6[R, A0, A1, A2, A3, A4, A5]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
}

func bind6[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any](p *prepared) func(A0, A1, A2, A3, A4, A5) R {
	s := newFrames[frame6[R, A0, A1, A2, A3, A4, A5]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame7[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	a5 A5
	a6 A6
	ps [7]unsafe.Pointer
}

func (f *frame7[R, A0, A1, A2, A3, A4, A5, A6]) init() {
	f.ps = [7]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5), unsafe.Pointer(&f.a6)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame7[R, A0, A1, A2, A3, A4, A5, A6]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
	var z6 A6
	f.a6 = z6
}

func bind7[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any](p *prepared) func(A0, A1, A2, A3, A4, A5, A6) R {
	s := newFrames[frame7[R, A0, A1, A2, A3, A4, A5, A6]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		f.a6 = a6
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame8[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	a5 A5
	a6 A6
	a7 A7
	ps [8]unsafe.Pointer
}

func (f *frame8[R, A0, A1, A2, A3, A4, A5, A6, A7]) init() {
	f.ps = [8]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5), unsafe.Pointer(&f.a6), unsafe.Pointer(&f.a7)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame8[R, A0, A1, A2, A3, A4, A5, A6, A7]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
	var z6 A6
	f.a6 = z6
	var z7 A7
	f.a7 = z7
}

func bind8[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any](p *prepared) func(A0, A1, A2, A3, A4, A5, A6, A7) R {
	s := newFrames[frame8[R, A0, A1, A2, A3, A4, A5, A6, A7]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		f.a6 = a6
		f.a7 = a7
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame9[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	a5 A5
	a6 A6
	a7 A7
	a8 A8
	ps [9]unsafe.Pointer
}

func (f *frame9[R, A0, A1, A2, A3, A4, A5, A6, A7, A8]) init() {
	f.ps = [9]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5), unsafe.Pointer(&f.a6), unsafe.Pointer(&f.a7), unsafe.Pointer(&f.a8)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame9[R, A0, A1, A2, A3, A4, A5, A6, A7, A8]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
	var z6 A6
	f.a6 = z6
	var z7 A7
	f.a7 = z7
	var z8 A8
	f.a8 = z8
}

func bind9[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any](p *prepared) func(A0, A1, A2, A3, A4, A5, A6, A7, A8) R {
	s := newFrames[frame9[R, A0, A1, A2, A3, A4, A5, A6, A7, A8]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		f.a6 = a6
		f.a7 = a7
		f.a8 = a8
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame10[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any] struct {
	r  R
	a0 A0
	a1 A1
	a2 A2
	a3 A3
	a4 A4
	a5 A5
	a6 A6
	a7 A7
	a8 A8
	a9 A9
	ps [10]unsafe.Pointer
}

func (f *frame10[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]) init() {
	f.ps = [10]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5), unsafe.Pointer(&f.a6), unsafe.Pointer(&f.a7), unsafe.Pointer(&f.a8), unsafe.Pointer(&f.a9)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame10[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
	var z6 A6
	f.a6 = z6
	var z7 A7
	f.a7 = z7
	var z8 A8
	f.a8 = z8
	var z9 A9
	f.a9 = z9
}

func bind10[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any](p *prepared) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) R {
	s := newFrames[frame10[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8, a9 A9) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		f.a6 = a6
		f.a7 = a7
		f.a8 = a8
		f.a9 = a9
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

type frame11[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any, A10 any] struct {
	r   R
	a0  A0
	a1  A1
	a2  A2
	a3  A3
	a4  A4
	a5  A5
	a6  A6
	a7  A7
	a8  A8
	a9  A9
	a10 A10
	ps  [11]unsafe.Pointer
}

func (f *frame11[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]) init() {
	f.ps = [11]unsafe.Pointer{unsafe.Pointer(&f.a0), unsafe.Pointer(&f.a1), unsafe.Pointer(&f.a2), unsafe.Pointer(&f.a3), unsafe.Pointer(&f.a4), unsafe.Pointer(&f.a5), unsafe.Pointer(&f.a6), unsafe.Pointer(&f.a7), unsafe.Pointer(&f.a8), unsafe.Pointer(&f.a9), unsafe.Pointer(&f.a10)}
}

// clear drops the values so a parked frame holds no caller pointer.
func (f *frame11[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]) clear() {
	var zr R
	f.r = zr
	var z0 A0
	f.a0 = z0
	var z1 A1
	f.a1 = z1
	var z2 A2
	f.a2 = z2
	var z3 A3
	f.a3 = z3
	var z4 A4
	f.a4 = z4
	var z5 A5
	f.a5 = z5
	var z6 A6
	f.a6 = z6
	var z7 A7
	f.a7 = z7
	var z8 A8
	f.a8 = z8
	var z9 A9
	f.a9 = z9
	var z10 A10
	f.a10 = z10
}

func bind11[R any, A0 any, A1 any, A2 any, A3 any, A4 any, A5 any, A6 any, A7 any, A8 any, A9 any, A10 any](p *prepared) func(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) R {
	s := newFrames[frame11[R, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]]()
	return func(a0 A0, a1 A1, a2 A2, a3 A3, a4 A4, a5 A5, a6 A6, a7 A7, a8 A8, a9 A9, a10 A10) R {
		f, i := take(s)
		f.a0 = a0
		f.a1 = a1
		f.a2 = a2
		f.a3 = a3
		f.a4 = a4
		f.a5 = a5
		f.a6 = a6
		f.a7 = a7
		f.a8 = a8
		f.a9 = a9
		f.a10 = a10
		p.call(unsafe.Pointer(&f.r), f.ps[:])
		r := f.r
		f.clear()
		s.release(i)
		return r
	}
}

// prepAt is prep for an entry point already resolved to an address. name is
// for the panic message only.
func prepAt(fn unsafe.Pointer, name string, ret *types.TypeDescriptor, args ...*types.TypeDescriptor) *prepared {
	if fn == nil {
		panic("ffi: " + name + ": nil entry point")
	}
	p := &prepared{fn: fn}
	if err := gffi.PrepareCallInterface(&p.cif, types.DefaultConvention(), ret, args); err != nil {
		panic(fmt.Sprintf("ffi: %s: %v", name, err))
	}
	return p
}
