package tier

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
)

// A quantized matvec served alone (matVec) is the composer's: the call site
// asks it for the slice y = W x (graph.MatVec) at the tier's activation
// window, and launches what it forms -- the activation's int8 form and the
// contraction, generated from ops and shaped by its passes -- on the float
// row it uploads. The row is never quantized in Go.

// formedKey is a lone matvec's call-site shape: the composer asked, its
// weight's format and shape at an activation window. It names the request's
// jit.Key
// (devShared.formKeys) so the slice is not built again to be keyed; the
// kernels are found by the Key.
type formedKey struct {
	r            *jit.Registry
	t            jlm.Type
	rows, k, win int
}

// formed is a slice's kernels compiled on the device, in launch order, and
// the composer's own buffers they share: its scratch, and the blocks of
// constants its kernels read.
type formed struct {
	codes   []formedCode
	scratch []backend.Buf
}

type formedCode struct {
	k             backend.Kernel
	groups, width int
	args          []formedArg
	bufs          []backend.Buf // the call's arguments, reused
}

// formedArg is what a composed kernel's parameter binds: the uploaded row,
// the output, a region of the weight, or one of the composer's buffers.
type formedArg struct {
	kind uint8
	i    int
}

const (
	argRow uint8 = iota
	argOut
	argWeight
	argScratch
)

// formRole is the role the slice states its weight's format under: the call
// site's matrix, whatever role it plays in its model.
const formRole = jlm.RoleOutput

// formedMatVec is the composed matvec of format t at rows x k, compiled once
// on this device. Callers hold g.mu.
func (g *devTier) formedMatVec(t jlm.Type, rows, k int) (*formed, error) {
	if g.composer == nil {
		g.composer = jit.Default()
	}
	sk := formedKey{g.composer, t, rows, k, g.actWin}
	if key, ok := g.formKeys[sk]; ok {
		if f := g.forms[key]; f != nil {
			return f, nil
		}
		return nil, fmt.Errorf("tier: the composed matvec of %v at %dx%d did not form earlier (%s)", t, rows, k, g.LastErr)
	}
	key, codes, err := g.composer.Form(jit.Request{
		Graph:   graph.MatVec(formRole, rows, k),
		Formats: map[jlm.Role]jlm.Type{formRole: t},
		Step:    jit.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: g.actWin},
		Device:  jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: g.pickLanes()},
	})
	if g.formKeys == nil {
		g.formKeys, g.forms = map[formedKey]jit.Key{}, map[jit.Key]*formed{}
	}
	// A slice that does not form or compile is remembered, as a compile
	// failure is (kernel): it is not tried again on every matvec.
	g.formKeys[sk] = key
	if err != nil {
		return nil, err
	}
	if f, ok := g.forms[key]; ok {
		if f == nil {
			return nil, fmt.Errorf("tier: the composed matvec of %v at %dx%d did not compile earlier", t, rows, k)
		}
		return f, nil
	}
	g.forms[key] = nil
	f := &formed{}
	scratch := map[jit.Operand]int{}
	for _, c := range codes {
		if c.GPU == nil {
			f.close()
			return nil, fmt.Errorf("tier: the composed matvec's %s has no GPU code", c.Name)
		}
		kern, err := g.dev.Compile(c.GPU)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("tier: compiling the composed %s: %w", c.Name, err)
		}
		fc := formedCode{k: kern, groups: c.Groups, width: c.GPU.Group[0]}
		f.codes = append(f.codes, fc)
		fcp := &f.codes[len(f.codes)-1]
		for _, b := range c.Binds {
			o := b.Operand
			switch {
			case o.Kind == jit.OperandValue && o.Value == 0:
				fcp.args = append(fcp.args, formedArg{kind: argRow})
			case o.Kind == jit.OperandValue && o.Value == 1:
				fcp.args = append(fcp.args, formedArg{kind: argOut})
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				fcp.args = append(fcp.args, formedArg{kind: argWeight, i: o.Region})
			case b.Words != nil:
				// A block of constants the kernel reads (a format's code
				// table): uploaded once with its words.
				buf, err := g.dev.Alloc(4 * len(b.Words))
				if err != nil {
					f.close()
					return nil, err
				}
				if err := buf.Write(u32b(b.Words)); err != nil {
					buf.Free()
					f.close()
					return nil, err
				}
				fcp.args = append(fcp.args, formedArg{kind: argScratch, i: len(f.scratch)})
				f.scratch = append(f.scratch, buf)
			case o.Kind == jit.OperandScratch:
				at, ok := scratch[o]
				if !ok {
					buf, err := g.dev.Alloc(int(max(b.Bytes, 4)))
					if err != nil {
						f.close()
						return nil, err
					}
					at = len(f.scratch)
					scratch[o] = at
					f.scratch = append(f.scratch, buf)
				}
				fcp.args = append(fcp.args, formedArg{kind: argScratch, i: at})
			default:
				f.close()
				return nil, fmt.Errorf("tier: the composed matvec's %s binds %+v, which a lone matvec does not provide", c.Name, o)
			}
		}
		fcp.bufs = make([]backend.Buf, len(fcp.args))
	}
	g.forms[key] = f
	return f, nil
}

func (f *formed) close() {
	for _, c := range f.codes {
		c.k.Close()
	}
	for _, b := range f.scratch {
		b.Free()
	}
	f.codes, f.scratch = nil, nil
}

// formsClose frees every composed matvec. Callers hold g.mu.
func (g *devTier) formsClose() {
	for k, f := range g.forms {
		if f != nil {
			f.close()
		}
		delete(g.forms, k)
	}
	clear(g.formKeys)
}

// fmCall is one composed matvec's device side, handed to formedSession
// through the devTier rather than captured by a closure (see mvCall).
type fmCall struct {
	f     *formed
	r     *resident
	x     []float32
	nrows int
	err   error
}

// formedSession uploads the row, launches the composed kernels and reads the
// output back.
func (g *devTier) formedSession(sess backend.Session) {
	c := &g.fmc
	if c.err = sess.Write(g.xfBuf, f32b(c.x)); c.err != nil {
		return
	}
	lc := launcher{to: &g.mvTo}
	g.mvTo.s = sess
	defer func() { g.mvTo.s = nil }()
	w := [3]backend.Buf{c.r.qs, c.r.d, c.r.sc}
	for i := range c.f.codes {
		fc := &c.f.codes[i]
		for j, a := range fc.args {
			switch a.kind {
			case argRow:
				fc.bufs[j] = g.xfBuf
			case argOut:
				fc.bufs[j] = g.outBuf
			case argWeight:
				fc.bufs[j] = w[a.i]
			case argScratch:
				fc.bufs[j] = c.f.scratch[a.i]
			}
		}
		if c.err = lc.launch(fc.k, fc.groups, fc.width, fc.bufs...); c.err != nil {
			return
		}
	}
	c.err = sess.Read(g.outBuf, g.hostRaw[:c.nrows*4])
}

// formedMatVecCall serves a quantized matvec from its composed kernels.
// Callers hold g.mu.
func (g *devTier) formedMatVecCall(out []float32, t quant.Type, r *resident, w []byte, x []float32, nrows, k int) decline {
	jt, ok := jlm.TypeOf(t)
	if !ok {
		g.NoKernel++
		return noKernel
	}
	t0 := time.Now()
	f, err := g.formedMatVec(jt, nrows, k)
	if err != nil {
		g.Failed++
		g.LastErr = err.Error()
		return failed
	}
	if !g.sizeOut(nrows) || !g.sizeXF(k) {
		return noRoom
	}
	t1 := time.Now()
	g.fmc = fmCall{f: f, r: r, x: x[:k], nrows: nrows}
	if g.fmcFn == nil {
		g.fmcFn = g.formedSession
	}
	g.dev.Session(g.fmcFn)
	err = g.fmc.err
	g.fmc = fmCall{}
	if err != nil {
		g.Failed++
		g.LastErr = err.Error()
		return failed
	}
	t3 := time.Now()
	copy(out[:nrows], g.hostOut[:nrows])
	g.TStage += t1.Sub(t0)
	g.TLaunch += t3.Sub(t1)
	g.TConv += time.Since(t3)
	if g.Verify {
		g.verify(out, t, w, x, nrows, k)
	}
	g.Served++
	g.Formed++
	return served
}
