package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedEmbedReadsRowZeroPastTheTable runs the composer's input rows
// on every backend present: rows of an embedding over the step's tokens, on
// both sides of 2^31 and past the table, and of a position table over its
// positions, past the table. A token past the table, the all-ones word a row
// naming no token holds, and a position past the table read row 0, as the CPU's
// form does (composer.TestComposedCPUEmbedsARowNamingNoToken), and every
// other row its own; a guard after the output survives. A card that read
// past the table would read the guard row after it, or a word far before it.
func TestComposedEmbedReadsRowZeroPastTheTable(t *testing.T) {
	gpuLock(t)
	devs := backend.OpenForTest()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, positional := range []bool{false, true} {
			name := "tokens"
			if positional {
				name = "positions"
			}
			t.Run(fmt.Sprintf("%s/%s", d.API(), name), func(t *testing.T) { composedEmbedBound(t, d, positional) })
		}
	}
}

func composedEmbedBound(t *testing.T, d backend.Device, positional bool) {
	const R, K, N = 7, 13, 9
	role := jlm.RoleTokenEmbd
	if positional {
		role = jlm.RolePosEmbd
	}
	b := graph.New()
	rows := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "rows", Width: K}},
		[]graph.Binding{{Role: role, K: K, Rows: N}}, graph.Attrs{Positional: positional}, nil, nil)[0]
	b.Out(rows)
	g := b.G
	lanes := 1
	if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
		lanes = ir.SubgroupLanes
	}
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{role: jlm.TypeF32},
		Step:   composer.Step{Rows: R, Page: 1, Keys: 1},
		Device: composer.Device{Kind: composer.DeviceGPU, Slots: d.Slots(), Lanes: lanes, ExactDiv: composer.ExactDivOn(d.API())}})
	if err != nil {
		t.Fatal(err)
	}
	// A guard row after the table: a gather of row N reads its values.
	table := make([]float32, (N+1)*K)
	for i := range table {
		table[i] = float32(i) + 0.5
	}
	for i := range K {
		table[N*K+i] = -2
	}
	words := []uint32{3, 0xFFFFFFFF, 0x80000000, 8, N, N + 2, 0}
	if positional {
		// A position is a small word, below 2^31, as the composer takes the
		// step's tables' words to be.
		words = []uint32{3, 1 << 20, N + 5, 8, N, N + 2, 0}
	}
	gg := newGPU(t, d)
	defer gg.free()
	// rows.n is every row of the form, as the tier binds it for a part
	// whose call states none.
	tables := map[string]backend.Buf{role.String(): gg.up(f32bytes(table)), "rows.n": gg.up(u32bytes([]uint32{^uint32(0)}))}
	if positional {
		// The step's positions: the live row count, then a row's position.
		tables["rope.pos"] = gg.up(u32bytes(append([]uint32{R}, words...)))
	} else {
		tables["tokens"] = gg.up(u32bytes(words))
	}
	guarded := guardedTable(R * K)
	out := gg.up(u32bytes(guarded))
	tables[g.Values[rows].Name] = out
	for _, c := range codes {
		k, err := d.Compile(c.GPU)
		if err != nil {
			t.Fatalf("compiling %s: %v", c.Name, err)
		}
		var bufs []backend.Buf
		for _, b := range c.Binds {
			o := b.Operand
			switch {
			case b.Words != nil:
				bufs = append(bufs, gg.up(u32bytes(b.Words)))
			case tables[o.Name] != nil:
				bufs = append(bufs, tables[o.Name])
			default:
				k.Close()
				t.Fatalf("%s binds %+v, which the embedding's call site does not hold", c.Name, o)
			}
		}
		err = k.Launch(c.Groups, c.GPU.Group[0], bufs...)
		k.Close()
		if err != nil {
			t.Fatalf("launching %s: %v", c.Name, err)
		}
	}
	raw := make([]byte, 4*len(guarded))
	if err := out.Read(raw); err != nil {
		t.Fatal(err)
	}
	for i := range guarded {
		guarded[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	got := unguard(t, guarded, R*K)
	for r, w := range words {
		row := int(w)
		if w >= N {
			row = 0
		}
		for i := range K {
			if want, v := table[row*K+i], got[r*K+i]; math.Float32bits(want) != math.Float32bits(v) {
				t.Fatalf("row %d (word %#x) element %d: %v, the table's row %d holds %v", r, w, i, v, row, want)
			}
		}
	}
	t.Logf("%d rows, words across 2^31 and past the table, each its row or row 0", R)
}
