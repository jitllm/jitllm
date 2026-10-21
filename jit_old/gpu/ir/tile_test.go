package ir

import (
	"strings"
	"testing"
)

// TestTileValidationRefusesEachViolation is the gate on the collective value's
// contract, and every case here is a program that would LOWER and be wrong.
//
// A tile is not a register: it is one matrix spread across a subgroup with an
// undefined lane mapping, so reaching an add, a shuffle or an ordinary store
// would lower and give whatever the driver happened to do.
func TestTileValidationRefusesEachViolation(t *testing.T) {
	accT := TileType{Rows: 16, Cols: 8, Elem: TileF32, Use: TileAcc}
	aT := TileType{Rows: 16, Cols: 16, Elem: TileF16, Use: TileA}
	bT := TileType{Rows: 16, Cols: 8, Elem: TileF16, Use: TileB}

	// The well-formed program every case below breaks in one place.
	good := func() *Kernel {
		b := New("tile", [3]int{32, 1, 1})
		pA, pB := b.Param("pA", U32), b.Param("pB", U32)
		pOut := b.Param("pOut", F32)
		z, st := b.Const(U32, 0), b.Const(U32, 16)
		a := b.TileLoad(aT, pA, z, st, false)
		bb := b.TileLoad(bT, pB, z, st, false)
		c := b.TileSplat(accT, b.Const(F32, 0))
		d := b.TileMMA(a, bb, c)
		b.TileStore(pOut, z, b.Const(U32, 8), d, false)
		return b.Done()
	}
	if err := good().Validate(); err != nil {
		t.Fatalf("the well-formed kernel does not validate: %v", err)
	}

	for _, c := range []struct {
		name, want string
		break_     func(k *Kernel)
	}{
		{"tile into a scalar add", "as scalar argument", func(k *Kernel) {
			// The MMA result, added like a number.
			for i, o := range k.Ops {
				if o.Kind == OpTileMMA {
					k.Ops = append(k.Ops[:i+1], append([]Op{{
						Kind: OpAdd, Type: F32, Args: [3]Value{Value(i + 1), Value(i + 1)},
					}}, k.Ops[i+1:]...)...)
					return
				}
			}
		}},
		{"lane-dependent offset", "lane-dependent", func(k *Kernel) {
			// tid as the load offset: every lane would name a different tile.
			k.Ops = append([]Op{{Kind: OpTID, Type: U32}}, k.Ops...)
			for i := range k.Ops {
				for n := range k.Ops[i].Args {
					if k.Ops[i].Args[n] != 0 {
						k.Ops[i].Args[n]++
					}
				}
				for n := range k.Ops[i].Vargs {
					if k.Ops[i].Vargs[n] != 0 {
						k.Ops[i].Vargs[n]++
					}
				}
			}
			for i, o := range k.Ops {
				if o.Kind == OpTileLoad {
					k.Ops[i].Args[1] = 1 // the tid
					return
				}
			}
		}},
		{"mismatched K", "do not compose", func(k *Kernel) {
			for i, o := range k.Ops {
				if o.Kind == OpTileLoad && o.Tile.Use == TileB {
					tt := *o.Tile
					tt.Rows = 8 // A's K is 16
					k.Ops[i].Tile = &tt
					return
				}
			}
		}},
		{"operand roles swapped", "want A/B/Acc", func(k *Kernel) {
			for i, o := range k.Ops {
				if o.Kind == OpTileMMA {
					k.Ops[i].Args[0], k.Ops[i].Args[1] = o.Args[1], o.Args[0]
					return
				}
			}
		}},
		{"narrow accumulator", "not a well-formed tile type", func(k *Kernel) {
			for i, o := range k.Ops {
				if o.Kind == OpTileSplat {
					tt := *o.Tile
					tt.Elem = TileF16 // f16 operands must accumulate in f32
					k.Ops[i].Tile = &tt
					return
				}
			}
		}},
		{"no subgroup width", "needs 32", func(k *Kernel) { k.Lanes = 1 }},
		{"load through a non-parameter", "neither a parameter nor a shared array", func(k *Kernel) {
			for i, o := range k.Ops {
				if o.Kind == OpTileLoad {
					k.Ops[i].Args[0] = o.Args[1] // the constant zero
					return
				}
			}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			k := good()
			c.break_(k)
			err := k.Validate()
			if err == nil {
				t.Fatalf("Validate accepted %q", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused %q with %q, which does not name the problem (%q)",
					c.name, err, c.want)
			}
			t.Logf("%s", err)
		})
	}
}

// TestTileLoopFormsAndTheirViolations is the gate on the three forms a staged
// GEMM needs (kernels.GemmTile): a tile carried across a Loop (TilePhi), a tile
// read out of a shared array, and an offset built from SubgroupIndex. Each is
// accepted in its well-formed shape and refused one step away from it -- the
// uniformity exemption in particular must cover tid/32 and NOT tid/16, which
// differs between the two halves of a subgroup.
func TestTileLoopFormsAndTheirViolations(t *testing.T) {
	aT := TileType{Rows: 8, Cols: 8, Elem: TileF16, Use: TileA}
	bT := TileType{Rows: 8, Cols: 8, Elem: TileF16, Use: TileB}
	cT := TileType{Rows: 8, Cols: 8, Elem: TileF32, Use: TileAcc}
	type knob struct {
		shift  int64 // the subgroup index's shift: 5 is uniform, 4 is not
		update bool  // close the tile phi with a wrongly typed tile
		scalar bool  // carry the tile through a SCALAR phi
	}
	build := func(k knob) *Kernel {
		b := New("tileloop", [3]int{64, 1, 1})
		pOut := b.ParamTile("pOut", TileF32)
		sh := b.Shared("sh", U32, 64*8)
		sg := b.Shr(U32, b.TID(), b.Const(U32, k.shift))
		b.k.Lanes = SubgroupLanes
		off := b.Mul(U32, sg, b.Const(U32, 64))
		st := b.Const(U32, 8)
		init := b.TileSplat(cT, b.ConstF32(0))
		b.Loop(4)
		var acc Value
		if k.scalar {
			acc = b.Phi(F32, init)
		} else {
			acc = b.TilePhi(init)
		}
		a := b.TileLoad(aT, sh, off, st, false)
		bb := b.TileLoad(bT, sh, off, st, true)
		in := acc
		if k.scalar {
			in = init // the builder refuses a scalar operand outright; the phi is what carries it
		}
		d := b.TileMMA(a, bb, in)
		if k.update {
			d = b.TileLoad(TileType{Rows: 8, Cols: 16, Elem: TileF32, Use: TileAcc}, sh, off, st, false)
		}
		b.SetPhi(acc, d)
		b.EndLoop()
		out := acc
		if k.scalar {
			out = d
		}
		b.TileStore(pOut, off, st, out, false)
		return b.Done()
	}
	if err := build(knob{shift: 5}).Validate(); err != nil {
		t.Fatalf("the well-formed loop does not validate: %v", err)
	}
	for _, c := range []struct {
		name, want string
		k          knob
	}{
		{"half a subgroup's index", "lane-dependent", knob{shift: 4}},
		{"a phi closed with another tile type", "tile phi's update", knob{shift: 5, update: true}},
		{"a tile through a scalar phi", "as scalar argument", knob{shift: 5, scalar: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := build(c.k).Validate()
			if err == nil {
				t.Fatalf("Validate accepted %q", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused %q with %q, which does not name the problem (%q)", c.name, err, c.want)
			}
			t.Logf("%s", err)
		})
	}
}
