package composer_test

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestTheStatedTileIsTheOneThatCompiles holds backend.LowersTile, which
// compiles nothing, to what the device answers when it compiles the
// composer's tiled forms: a binary16 projection on the matrix unit's tiles,
// and an attention on them (the flash pass), its sums' rows scaled in place
// in a third arm. A device that states the tile must compile every kernel
// of each arm, one that states none must refuse one; both directions run on
// every device present, and on a CUDA card lowered for sm_70.
func TestTheStatedTileIsTheOneThatCompiles(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if d, ok := sm70(t); ok {
		devs = append(devs, d)
	}
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no GPU device here")
	}
	arms := []struct {
		name  string
		scale bool
		form  func(*testing.T, composer.Device) []composer.Code
		op    ir.Kind
	}{
		{"projection", false, tiledProjection, ir.OpTileMMA},
		{"attention", false, tiledAttention, ir.OpTileMMA},
		{"attention-scaled", true, tiledAttention, ir.OpTileScale},
	}
	asked, yes := 0, 0
	for i, d := range devs {
		for _, a := range arms {
			s := matrixTile()
			s.Scale = a.scale
			stated, why := backend.LowersTile(d, s)
			dev := gpuDevice(d)
			dev.Matrix = [3]int{s.M, s.N, s.K}
			dev.Fragment = ir.MMAShape{}
			dev.ScaleRows = a.scale
			codes := a.form(t, dev)
			if !slices.ContainsFunc(codes, func(c composer.Code) bool {
				return c.GPU != nil && slices.ContainsFunc(c.GPU.Ops, func(o ir.Op) bool { return o.Kind == a.op })
			}) {
				t.Fatalf("dev%d %s %s: no kernel formed holds a %s: the arm asks the device nothing", i, d.API(), a.name, a.op)
			}
			op, err := compileLone(d, codes)
			compiled := err == nil
			if compiled {
				op.close()
			}
			t.Logf("dev%d %s %s %s: states %v (%s), compiles %v (%v)", i, d.API(), d.Name(), a.name, stated, why, compiled, err)
			if stated != compiled {
				t.Errorf("dev%d %s %s: backend.LowersTile states %v (%s) and its tiled form compiles %v (%v)",
					i, d.API(), a.name, stated, why, compiled, err)
			}
			asked++
			if stated {
				yes++
			}
		}
	}
	t.Logf("%d answers held to a compile, %d of them yes", asked, yes)
}

// tiledProjection is a binary16 projection of 256 into 100 at a prompt
// chunk of 100 rows, formed for dev.
func tiledProjection(t *testing.T, dev composer.Device) []composer.Code {
	dev.Tile = splitTiles(gpuMatrixTiles())[0]
	b := graph.New()
	in := b.In("x", 256)
	b.Out(b.MatVec(in, jlm.RoleAttnQ, 100))
	_, codes, err := poisonedRegistry().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jlm.TypeF16},
		Step: composer.Step{Rows: 100, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatalf("forming the projection: %v", err)
	}
	return codes
}

// tiledAttention is an attention of 8 heads over 2 of 64 dimensions at a
// prompt chunk of 64 rows, formed for dev.
func tiledAttention(t *testing.T, dev composer.Device) []composer.Code {
	const heads, kvh, dim, page, rows = 8, 2, 64, 64, 64
	keys := (2100 + 37 + 64*page - 1) / (64 * page) * 64 * page
	b := graph.New()
	n := b.KVAppend(b.In("k", kvh*dim), b.In("v", kvh*dim), 0)
	b.Out(b.Attend(b.In("q", heads*dim), n, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: dim, Scale: 1 / math.Sqrt(dim)}))
	_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: rows, Page: page, Keys: keys, Seqs: 1},
		Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
	if err != nil {
		t.Fatalf("forming the attention: %v", err)
	}
	return codes
}
