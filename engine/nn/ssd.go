//go:build amd64 || arm64

package nn

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/jit/cpu"
)

// AddSSD generates Mamba-2's selective state update for one state size (the
// container's SSM.StateSize), provisioned per model like AddDelta.
func (f *JIT) AddSSD(n int) {
	if f.ssd != nil {
		return
	}
	f.ssd, f.ssdN = mustEmit("gated_ssd_n"+itoaN(n))(f.em.GatedSSD(n)), n
}

// GatedSSD runs one head of Mamba-2's selective state update: rows rows of n
// (the head's head_dim by the state size), with b the group's B (the key), c
// its C (the query) and x the head's value.
//
//	state[j] = state[j]*decay + (x[j]*dt)*b
//	o[j]     = dot(state[j], c) + d*x[j]
//
// state is read and written.
func (f *JIT) GatedSSD(o, state, b, c, x []float32, decay, dt, d float32, rows, n int) {
	if f.ssdN != n {
		panic(fmt.Sprintf("jit: GatedSSD at state size %d, provisioned for %d", n, f.ssdN))
	}
	if rows <= 0 || len(o) < rows || len(x) < rows || len(state) < rows*n || len(b) < n || len(c) < n {
		lengthPanic("gated_ssd", rows*n, min(len(o), len(x), len(b), len(c)))
	}
	sc := [3]float32{decay, dt, d}
	args := cpu.Args{
		Out:    &o[0],
		W:      (*byte)(unsafe.Pointer(&state[0])),
		AScale: &b[0],
		Rows:   int64(rows),
		Scr:    (*byte)(unsafe.Pointer(&sc[0])),
		Q32:    &c[0],
		Q2:     &x[0],
	}
	f.ssd.Call(&args)
}

// ssdGateOnce guards Mamba-2's gate kernel, shape-free like the delta gate.
var ()

// AddSelScan generates Mamba-1's selective scan for one state size,
// provisioned per model like AddSSD.
func (f *JIT) AddSelScan(n int) {
	if f.selScan != nil {
		return
	}
	f.selScan, f.selScanN = mustEmit("sel_scan_n"+itoaN(n))(f.em.SelScan(n)), n
}

// SelScan runs Mamba-1's selective scan over rows channels: each channel is a
// head of one row whose decay is a vector over the state of n.
//
//	state[j] = state[j]*exp(dt[j]*a[j]) + (x[j]*dt[j])*b
//	o[j]     = dot(state[j], c) + d[j]*x[j]
//
// a and state are rows*n; o, x, dt and d are rows. state is read and written.
func (f *JIT) SelScan(o, state, b, c, x, dt, a, d []float32, rows, n int) {
	if f.selScanN != n {
		panic(fmt.Sprintf("jit: SelScan at state size %d, provisioned for %d", n, f.selScanN))
	}
	if rows <= 0 || len(o) < rows || len(x) < rows || len(dt) < rows || len(d) < rows ||
		len(state) < rows*n || len(a) < rows*n || len(b) < n || len(c) < n {
		lengthPanic("sel_scan", rows*n, min(len(o), len(x), len(dt), len(d), len(state)/n, len(a)/n))
	}
	args := cpu.Args{
		Out:     &o[0],
		W:       (*byte)(unsafe.Pointer(&state[0])),
		AScale:  &b[0],
		Rows:    int64(rows),
		Scr:     (*byte)(unsafe.Pointer(&gateConsts[0])),
		AHalf:   &a[0],
		Q32:     &c[0],
		Q2:      &x[0],
		AScale2: &dt[0],
		PD:      (*byte)(unsafe.Pointer(&d[0])),
	}
	f.selScan.Call(&args)
}
