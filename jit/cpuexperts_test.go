package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestComposedCPUExpertPages executes an expert matvec over a bank held in
// pages (Request.Paged): fewer sheets than the bank's experts, in another
// order, each expert reached through the step's table "experts". Every
// format, at one and two tokens, with and without the int8 form and with a
// bias row an expert, must give the whole bank's answer to the bit, the
// whole bank's held to the oracle; with two held experts' words of the table
// swapped, it must not.
func TestComposedCPUExpertPages(t *testing.T) {
	const (
		E, held, P, K, rows = 8, 5, 3, 256, 32
	)
	rng := rand.New(rand.NewSource(11))
	// The held sheets in slot order, and the table: each held expert's slot,
	// the others an empty word the selection never reads.
	slotExpert := []int{6, 1, 4, 0, 3}
	table := make([]uint32, E)
	for e := range table {
		table[e] = math.MaxUint32
	}
	for s, e := range slotExpert {
		table[e] = uint32(s)
	}
	ran, unformed, arms := 0, 0, 0
	for _, typ := range []quant.Type{quant.F32, quant.Q8_0, quant.Q4_K, quant.Q6_K, quant.MXFP4} {
		q, ok := kernels.QuantOf(typ)
		if typ == quant.F32 {
			q, ok = kernels.Float32, true
		}
		jt, jok := jlm.TypeOf(typ)
		if !ok || !jok {
			t.Fatalf("%v has no device or container type", typ)
		}
		sheets := make([][]byte, E)
		packed := make([][3][]uint32, E)
		for e := range sheets {
			sheets[e] = cpuTestWeights(typ, q, rows, K, rng)
			qs, d, sc, err := kernels.PackWeights(q, sheets[e], rows, K)
			if err != nil {
				t.Fatalf("%v: %v", typ, err)
			}
			packed[e] = [3][]uint32{qs, d, sc}
		}
		bank := func(order []int) [3][]uint32 {
			var b [3][]uint32
			for _, e := range order {
				for i := range b {
					b[i] = append(b[i], packed[e][i]...)
				}
			}
			for i := range b {
				if len(b[i]) == 0 {
					b[i] = []uint32{0}
				}
			}
			return b
		}
		whole := bank([]int{0, 1, 2, 3, 4, 5, 6, 7})
		pages := bank(slotExpert)
		// The bank as the host's expert pages hold it: each sheet's planes
		// a page apart, the words between them garbage.
		stride := 7
		for i := range packed[0] {
			stride = max(stride, len(packed[0][i])+7)
		}
		var strided [3][]uint32
		for i := range strided {
			strided[i] = make([]uint32, E*stride)
			for j := range strided[i] {
				strided[i][j] = rng.Uint32()
			}
			for e := range E {
				copy(strided[i][e*stride:], packed[e][i])
			}
		}
		bias := make([]float32, E*rows)
		for i := range bias {
			bias[i] = float32(rng.NormFloat64())
		}
		for _, arm := range []struct {
			name string
			r    *Registry
		}{{"float", Default().WithoutInt8(DeviceCPU)}, {"int8", withCPUInt8()}} {
			for _, tokens := range []int{1, 2} {
				for _, biased := range []bool{false, true} {
					name := fmt.Sprintf("%v/%s/%d tokens/bias %v", typ, arm.name, tokens, biased)
					arms++
					b := graph.New()
					x := b.In("x", P*K)
					sel := b.In("sel", P)
					b.G.Values[sel].Elem = graph.ElemU32
					w := []graph.Binding{{Role: jlm.RoleExpGateBank, K: K, Rows: rows, Sheets: E}}
					formats := map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: jt}
					if biased {
						w = append(w, graph.Binding{Role: jlm.RoleExpGateBias, K: rows, Rows: E})
						formats[jlm.RoleExpGateBias] = jlm.TypeF32
					}
					y := b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * rows}}, w,
						graph.Attrs{}, nil, nil)
					b.Out(y[0])
					xs := make([]float32, tokens*P*K)
					for i := range xs {
						xs[i] = float32(rng.NormFloat64())
					}
					sels := make([]uint32, tokens*P)
					for i := range sels {
						sels[i] = uint32(slotExpert[rng.Intn(held)])
					}
					ref := xs
					if typ != quant.F32 && arm.r.Profiles[DeviceCPU].Int8 != nil {
						ref = int8Row(xs, 32)
					}
					want := make([]float32, tokens*P*rows)
					for tk := range tokens {
						for j := range P {
							at := tk*P + j
							e := int(sels[at])
							if err := oracle.MatVec(want[at*rows:(at+1)*rows], typ, sheets[e], ref[at*K:(at+1)*K], rows, K); err != nil {
								t.Fatal(err)
							}
							if biased {
								for r := range rows {
									want[at*rows+r] += bias[e*rows+r]
								}
							}
						}
					}
					// The violation: the first selected expert's page swapped
					// with another held expert's.
					swapped := append([]uint32(nil), table...)
					other := slotExpert[0]
					if int(sels[0]) == other {
						other = slotExpert[1]
					}
					swapped[sels[0]], swapped[other] = swapped[other], swapped[sels[0]]
					run := func(paged, strides map[jlm.Role]int, planes [3][]uint32, tab []uint32) []float32 {
						t.Helper()
						_, codes, err := arm.r.Form(Request{Graph: b.G, Formats: formats, Paged: paged, Strides: strides,
							Step: Step{Rows: tokens, Page: 64, Keys: 1, ActWindow: 32}, Device: cpuDevice()})
						switch {
						case err != nil && paged == nil:
							// The whole bank's form is the reference: every
							// arm forms it, whatever the format, the int8
							// form or the tokens hold live at once.
							t.Errorf("%s: the whole bank does not form: %v", name, err)
							unformed++
							return nil
						case err != nil:
							t.Fatalf("%s: the paged bank does not form where the whole bank does: %v", name, err)
						}
						out := make([]uint32, tokens*P*rows+16)
						nan := math.Float32bits(float32(math.NaN()))
						for i := range out {
							out[i] = nan
						}
						sw := append([]uint32(nil), sels...)
						vecs := map[jlm.Role][]float32{jlm.RoleExpGateBias: bias}
						_, err = runCPUBound(codes, xs, 1, planes, vecs, func(o Operand) []uint32 {
							switch {
							case o.Kind == OperandValue && o.Value == sel:
								return sw
							case o.Kind == OperandValue && o.Value == y[0]:
								return out
							case o.Kind == OperandTable && o.Name == tableExperts.Name:
								return tab
							}
							return nil
						})
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						for i, v := range out[tokens*P*rows:] {
							if v != nan {
								t.Fatalf("%s: the kernels wrote past the output's end, at element %d", name, i)
							}
						}
						return unsafe.Slice((*float32)(unsafe.Pointer(&out[0])), tokens*P*rows)
					}
					got := run(nil, nil, whole, nil)
					if got == nil {
						continue
					}
					if e := cpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
						t.Errorf("%s: the whole bank is %.3e from the oracle", name, e)
						continue
					}
					pg := map[jlm.Role]int{jlm.RoleExpGateBank: held}
					paged := run(pg, nil, pages, table)
					if paged == nil {
						continue
					}
					for i := range got {
						if math.Float32bits(paged[i]) != math.Float32bits(got[i]) {
							t.Errorf("%s: the paged bank's element %d is %v, the whole bank's %v", name, i, paged[i], got[i])
							break
						}
					}
					bad := run(pg, nil, pages, swapped)
					if e := cpuNMSE(bad, got); !(e > 1e-6) {
						t.Errorf("%s: with two experts' table words swapped the answer moved %.3e: the gate cannot see a page read from the wrong expert", name, e)
					}
					// The strided bank: every sheet read from its own start
					// (a float sheet's rows by another index expression, within
					// a rounding of the packed bank's), and with the stride a
					// word short, not.
					st := map[jlm.Role]int{jlm.RoleExpGateBank: stride}
					if sv := run(nil, st, strided, nil); sv != nil {
						if e := cpuNMSE(sv, got); math.IsNaN(e) || e > 1e-12 {
							t.Errorf("%s: the strided bank is %.3e from the packed bank's", name, e)
						}
					}
					st[jlm.RoleExpGateBank] = stride - 1
					if sv := run(nil, st, strided, nil); sv != nil && !(cpuNMSE(sv, got) > 1e-6) && !math.IsNaN(cpuNMSE(sv, got)) {
						t.Errorf("%s: with the stride a word short the answer moved %.3e: the gate cannot see a sheet read from its neighbour's place", name, cpuNMSE(sv, got))
					}
					ran++
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no paged bank ran: this gate proved nothing")
	}
	t.Logf("%d paged arms equal to the whole bank's; %d arms the whole bank does not form on the CPU", ran, unformed)
	if ran != arms {
		t.Errorf("%d of %d arms ran", ran, arms)
	}
}
