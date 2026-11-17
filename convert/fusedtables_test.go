package convert

import (
	"maps"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// stateTable runs arch's GGUF table from the library on in, block tensors
// named blk.N., its sizes read off keys and its recurrent blocks those kinds
// mark.
func stateTable(t *testing.T, arch string, in []jlm.Tensor, keys map[string]any, kinds []jlm.LayerKind) ([]jlm.Tensor, error) {
	t.Helper()
	files, err := libraryFiles()
	if err != nil {
		t.Fatal(err)
	}
	var st *mjStates
	for _, mj := range files {
		if s, ok := mj.Container[arch]; ok && st == nil {
			st = s
		}
	}
	if st == nil || st.Tensors == nil {
		t.Fatalf("the library states no %s tensor table", arch)
	}
	names, err := readTensorNames()
	if err != nil {
		t.Fatal(err)
	}
	stated := &Statement{blockTensors: map[string][]statedCut{}, linearTensors: map[string][]statedCut{}}
	for _, x := range []struct {
		table map[string]mjTargets
		to    map[string][]statedCut
	}{{st.Tensors.Block, stated.blockTensors}, {st.Tensors.BlockLinear, stated.linearTensors}} {
		for n, ts := range x.table {
			if x.to[n], err = statedCuts(names, n, ts); err != nil {
				t.Fatal(err)
			}
		}
	}
	if stated.sizes, err = statedSizes(st.Tensors.Sizes); err != nil {
		t.Fatal(err)
	}
	c := &Config{Config: &jlm.Config{LayerKinds: kinds}, Stated: stated}
	s := &jlm.Source{Tensors: slices.Clone(in)}
	err = statedTensors(c, s, func(p string) any { return keys[p] })
	return s.Tensors, err
}

// fusedKeys are a small attention block's GGUF keys: four heads of 16 over a
// 64-wide residual, two key-value heads, rotary over the whole head, two
// blocks of which the first slides, an FFN of 24, and three experts of 8.
func fusedKeys(over map[string]any) map[string]any {
	k := map[string]any{
		"embedding_length": 64.0, "block_count": 2.0, "attention.head_count": 4.0, "attention.head_count_kv": 2.0,
		"attention.key_length": 16.0, "attention.value_length": 16.0, "rope.dimension_count": 16.0,
		"attention.key_length_swa": 16.0, "attention.sliding_window_pattern": []any{1.0, 0.0},
		"feed_forward_length": 24.0, "expert_count": 3.0, "expert_feed_forward_length": 8.0,
	}
	maps.Copy(k, over)
	return k
}

// numbered is block b's f32 tensor of dims (GGUF order, innermost first),
// each element its expert * 1e6 + row * 1000 + column, so a part's values
// name the rows they came from.
func numbered(name string, b int32, dims ...uint64) jlm.Tensor {
	cols, rows, experts := dims[0], uint64(1), uint64(1)
	if len(dims) > 1 {
		rows = dims[1]
	}
	if len(dims) > 2 {
		experts = dims[2]
	}
	if len(dims) == 1 {
		cols, rows = 1, dims[0]
	}
	vals := make([]float32, 0, cols*rows*experts)
	for x := range experts {
		for r := range rows {
			for c := range cols {
				vals = append(vals, float32(x*1_000_000+r*1000+c))
			}
		}
	}
	e := jlm.Tensor{Name: name, Block: b, Index: -1}
	setF32Dims(&e, vals, dims...)
	return e
}

// holdsRows fails t unless p is rows [from, from+n) of each expert of a
// numbered source, saying what it found where it is not.
func holdsRows(t *testing.T, p jlm.Tensor, from, n uint64) {
	t.Helper()
	vals, err := vecOf(&p)
	if err != nil {
		t.Fatal(err)
	}
	cols, rows, experts := p.Dims[0], uint64(1), uint64(1)
	switch p.NDim {
	case 1:
		cols, rows = 1, p.Dims[0]
	case 2:
		rows = p.Dims[1]
	case 3:
		rows, experts = p.Dims[1], p.Dims[2]
	}
	if rows != n || uint64(len(vals)) != cols*rows*experts {
		t.Fatalf("%s (%v) is %v, %d values: want %d rows", p.Name, p.Role, p.Dims[:p.NDim], len(vals), n)
	}
	for x := range experts {
		for r := range rows {
			for c := range cols {
				want := float32(x*1_000_000 + (from+r)*1000 + c)
				if got := vals[(x*rows+r)*cols+c]; got != want {
					t.Fatalf("%s (%v) expert %d row %d column %d is %g, want source row %d (%g)",
						p.Name, p.Role, x, r, c, got, from+r, want)
				}
			}
		}
	}
}

func tableSize(t *testing.T, st *mjStates, name string, keys map[string]any, block int32) uint64 {
	t.Helper()
	x, err := parseExpr(st.Tensors.Sizes[name])
	if err != nil {
		t.Fatal(err)
	}
	v, err := evalNum(x, func(p string) any {
		if p == "block" {
			return float64(block)
		}
		return keys[p]
	})
	if err != nil {
		t.Fatalf("size %s: %v", name, err)
	}
	return uint64(v)
}

// TestEveryFusedTableCutsItsRowsInOrder: every library table that cuts a
// fused attn_qkv cuts it, and its bias, into attn_q, attn_k and attn_v in
// that order, the source's rows back to back with none left over, at the
// widths its sizes state; and refuses a source a row short of them.
func TestEveryFusedTableCutsItsRowsInOrder(t *testing.T) {
	files, err := libraryFiles()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, mj := range files {
		for _, arch := range slices.Sorted(maps.Keys(mj.Container)) {
			st := mj.Container[arch]
			if st.Tensors == nil || st.Tensors.Block["attn_qkv.weight"] == nil {
				continue
			}
			n++
			t.Run(arch, func(t *testing.T) {
				keys := fusedKeys(nil)
				w := []uint64{tableSize(t, st, "q", keys, 0), tableSize(t, st, "k", keys, 0), tableSize(t, st, "v", keys, 0)}
				all := w[0] + w[1] + w[2]
				for _, x := range []struct {
					name  string
					dims  []uint64
					short []uint64
					roles []jlm.Role
				}{
					{"blk.0.attn_qkv.weight", []uint64{64, all}, []uint64{64, all - 1}, []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV}},
					{"blk.0.attn_qkv.bias", []uint64{all}, []uint64{all - 1}, []jlm.Role{jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias}},
				} {
					out, err := stateTable(t, arch, []jlm.Tensor{numbered(x.name, 0, x.dims...)}, keys, nil)
					if err != nil {
						t.Fatal(err)
					}
					if len(out) != 3 {
						t.Fatalf("%s came out as %d tensors, want 3", x.name, len(out))
					}
					from := uint64(0)
					for i, p := range out {
						if p.Role != x.roles[i] {
							t.Fatalf("%s's part %d is %v, want %v", x.name, i, p.Role, x.roles[i])
						}
						holdsRows(t, p, from, w[i])
						from += w[i]
					}
					if out, err := stateTable(t, arch, []jlm.Tensor{numbered(x.name, 0, x.short...)}, keys, nil); err == nil {
						t.Errorf("%s a row short came out as %d tensors", x.name, len(out))
					}
				}
			})
		}
	}
	if n < 50 {
		t.Fatalf("%d tables cut a fused attn_qkv: this gate covers less than the library states", n)
	}
}

// TestFusedTablesAreTheirClassesWidths holds one table of each form to the
// widths each class's fused attn_qkv holds (q, k and v by the heads, or the
// class's own): keys whose heads disagree with the residual tell the
// forms apart, so a table of the wrong form fails here.
func TestFusedTablesAreTheirClassesWidths(t *testing.T) {
	odd := map[string]any{"attention.key_length": 20.0, "attention.value_length": 12.0}
	rot := map[string]any{"rope.dimension_count": 8.0}
	for _, tc := range []struct {
		arch  string
		over  map[string]any
		block int32
		want  [3]uint64
	}{
		{"phi3", odd, 0, [3]uint64{64, 24, 24}},        // n_embd, n_embd_gqa, n_embd_gqa
		{"falcon", odd, 0, [3]uint64{64, 24, 24}},      // {n_embd, n_embd + 2*n_embd_gqa}
		{"llama", odd, 0, [3]uint64{80, 40, 24}},       // n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa
		{"qwen3", odd, 0, [3]uint64{80, 24, 24}},       // n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa
		{"seed_oss", odd, 0, [3]uint64{80, 40, 40}},    // n_head * head_dim, n_head_kv * head_dim
		{"dots1", odd, 0, [3]uint64{80, 80, 80}},       // n_embd_head_k * n_head, thrice
		{"gpt-oss", rot, 0, [3]uint64{32, 16, 16}},     // n_head * n_rot, n_head_kv * n_rot
		{"qwen35", odd, 0, [3]uint64{160, 40, 24}},     // the gated query: twice the heads'
		{"modern-bert", odd, 0, [3]uint64{64, 64, 64}}, // {n_embd, 3 * n_embd}
		{"lfm2", map[string]any{"attention.key_length": 20.0, "attention.value_length": 12.0,
			"attention.head_count_kv": []any{0.0, 4.0}}, 1, [3]uint64{64, 80, 48}}, // n_embd, the block's k and v
		{"granitehybrid", map[string]any{"attention.head_count": []any{0.0, 4.0},
			"attention.head_count_kv": []any{0.0, 2.0}}, 1, [3]uint64{64, 32, 32}}, // the block's heads
		{"gemma4", map[string]any{"attention.key_length": 32.0, "attention.value_length": 32.0}, 0, [3]uint64{64, 32, 32}},
		{"gemma4", map[string]any{"attention.key_length": 32.0, "attention.value_length": 32.0}, 1, [3]uint64{128, 64, 64}},
	} {
		keys := fusedKeys(tc.over)
		name := "blk." + itoa(int(tc.block)) + ".attn_qkv.weight"
		all := tc.want[0] + tc.want[1] + tc.want[2]
		out, err := stateTable(t, tc.arch, []jlm.Tensor{numbered(name, tc.block, 64, all)}, keys, nil)
		if err != nil {
			t.Errorf("%s block %d: %v", tc.arch, tc.block, err)
			continue
		}
		if len(out) != 3 {
			t.Errorf("%s block %d: %d parts", tc.arch, tc.block, len(out))
			continue
		}
		from := uint64(0)
		for i, p := range out {
			holdsRows(t, p, from, tc.want[i])
			from += tc.want[i]
		}
	}
}

// TestARecurrentBlocksOwnQKVIsLeftWhole: the gated-delta classes' recurrent
// blocks store attn_qkv as their own projection, which their tables leave as
// it is while cutting an attention block's.
func TestARecurrentBlocksOwnQKVIsLeftWhole(t *testing.T) {
	for _, arch := range []string{"qwen35", "qwen35moe", "qwen3next", "qwen4exp"} {
		e := numbered("blk.0.attn_qkv.weight", 0, 64, 97)
		e.Role = jlm.RoleAttnQKV
		out, err := stateTable(t, arch, []jlm.Tensor{e}, fusedKeys(nil), []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerFullAttn})
		if err != nil || len(out) != 1 || out[0].Role != jlm.RoleAttnQKV || out[0].Name != e.Name {
			t.Errorf("%s: a recurrent block's attn_qkv came out as %d tensors (%v)", arch, len(out), err)
			continue
		}
		holdsRows(t, out[0], 0, 97)
	}
}

// TestFusedGateUpIsHalves: phi3's and glm4's ffn_up, and modern-bert's on its
// trunk, is the gate's rows then the up's, each the block's FFN wide; a
// decision head's ffn_up past modern-bert's trunk is its own, left whole.
func TestFusedGateUpIsHalves(t *testing.T) {
	for _, arch := range []string{"phi3", "glm4", "modern-bert"} {
		keys := fusedKeys(map[string]any{"decision.block_count": 1.0, "feed_forward_length": []any{24.0, 40.0}})
		out, err := stateTable(t, arch, []jlm.Tensor{numbered("blk.0.ffn_up.weight", 0, 64, 48)}, keys, nil)
		if err != nil || len(out) != 2 || out[0].Role != jlm.RoleFFNGate || out[1].Role != jlm.RoleFFNUp {
			t.Fatalf("%s: ffn_up came out as %d tensors (%v)", arch, len(out), err)
		}
		holdsRows(t, out[0], 0, 24)
		holdsRows(t, out[1], 24, 24)
		if _, err := stateTable(t, arch, []jlm.Tensor{numbered("blk.0.ffn_up.weight", 0, 64, 47)}, keys, nil); err == nil {
			t.Errorf("%s: an ffn_up a row short was cut", arch)
		}
	}
	keys := fusedKeys(map[string]any{"decision.block_count": 1.0, "feed_forward_length": []any{24.0, 40.0}})
	out, err := stateTable(t, "modern-bert", []jlm.Tensor{numbered("blk.1.ffn_up.weight", 1, 64, 40)}, keys, nil)
	if err != nil || len(out) != 1 {
		t.Fatalf("modern-bert's head ffn_up came out as %d tensors (%v)", len(out), err)
	}
	holdsRows(t, out[0], 0, 40)
}

// TestFusedExpertBankIsEachExpertsHalves: a fused ffn_gate_up_exps is, per
// expert, the gate's rows then the up's; every class whose loader reads one
// cuts it into the gate bank and the up bank.
func TestFusedExpertBankIsEachExpertsHalves(t *testing.T) {
	for _, arch := range []string{"gemma4", "qwen35moe", "qwen3next", "qwen4exp", "deepseek2", "deepseek32"} {
		keys := fusedKeys(nil)
		out, err := stateTable(t, arch, []jlm.Tensor{numbered("blk.0.ffn_gate_up_exps.weight", 0, 64, 16, 3)}, keys, nil)
		if err != nil || len(out) != 2 || out[0].Role != jlm.RoleExpGateBank || out[1].Role != jlm.RoleExpUpBank {
			t.Fatalf("%s: the bank came out as %d tensors (%v)", arch, len(out), err)
		}
		holdsRows(t, out[0], 0, 8)
		holdsRows(t, out[1], 8, 8)
		if _, err := stateTable(t, arch, []jlm.Tensor{numbered("blk.0.ffn_gate_up_exps.weight", 0, 64, 14, 3)}, keys, nil); err == nil {
			t.Errorf("%s: a bank of other widths was cut", arch)
		}
	}
}
