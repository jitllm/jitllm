package kernels_test

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

var updateAttn = flag.Bool("update-attn", false, "rewrite testdata/attn_ir.sha")

// attnSweep is every paged decode attention kernel at the plans the tier
// compiles: FlashDecodeKV over head widths, GQA groups, warps, splits, binary16
// values, sinks, a softcap, the K-immediate and vector-V forms, several rows;
// the staged scores, softmax and accumulate at their chunks and lanes (MLA
// too); and the wide merge with and without sinks and a run table. A refused
// shape is recorded as refused.
func attnSweep() map[string]string {
	out := map[string]string{}
	add := func(name string, k *ir.Kernel, err error) {
		if err != nil {
			out[name] = "refused"
			return
		}
		out[name] = irHash(k)
	}
	base := []kernels.FlashShape{
		{Heads: 8, KVHeads: 8, Dim: 64, Rows: 1, Page: 64, Scale: 0.125},
		{Heads: 16, KVHeads: 4, Dim: 128, Rows: 1, Page: 128, Scale: 0.0884},
		{Heads: 32, KVHeads: 8, Dim: 256, Rows: 1, Page: 64, Scale: 0.0625},
		{Heads: 16, KVHeads: 2, Dim: 64, Rows: 4, Page: 64, Scale: 0.125},
	}
	for i, s := range base {
		for _, v := range []struct {
			name string
			f    func(s kernels.FlashShape) kernels.FlashShape
		}{
			{"plain", func(s kernels.FlashShape) kernels.FlashShape { return s }},
			{"split4", func(s kernels.FlashShape) kernels.FlashShape { s.Splits = 4; return s }},
			{"f16", func(s kernels.FlashShape) kernels.FlashShape { s.F16 = true; return s }},
			{"sink", func(s kernels.FlashShape) kernels.FlashShape { s.Sink = true; return s }},
			{"softcap", func(s kernels.FlashShape) kernels.FlashShape { s.Softcap = 30; return s }},
			{"group1", func(s kernels.FlashShape) kernels.FlashShape { s.Group = 1; return s }},
			{"warps4", func(s kernels.FlashShape) kernels.FlashShape { s.Warps = 4; return s }},
			{"kimm", func(s kernels.FlashShape) kernels.FlashShape { s.KImm = true; return s }},
			{"vecv", func(s kernels.FlashShape) kernels.FlashShape { s.VecV = true; return s }},
		} {
			sh := v.f(s)
			k, err := kernels.FlashDecodeKV(sh)
			add(fmt.Sprintf("flashkv/%d/%s", i, v.name), k, err)
			m, err := kernels.FlashAttentionMergeWide(kernels.FlashShape{Heads: sh.Heads, Dim: sh.Dim, Rows: sh.Rows, Splits: 4, Sink: sh.Sink})
			add(fmt.Sprintf("merge/%d/%s", i, v.name), m, err)
		}
		for _, keys := range []int{512, 4096} {
			st := kernels.PagedStagedAt(s, keys, 8)
			for _, lanes := range []int{32, 64} {
				k, err := kernels.PagedStagedScores(st, lanes)
				add(fmt.Sprintf("staged/scores/%d/%d/%d", i, keys, lanes), k, err)
				k, err = kernels.PagedStagedSoftmax(st, lanes)
				add(fmt.Sprintf("staged/softmax/%d/%d/%d", i, keys, lanes), k, err)
			}
			k, err := kernels.PagedStagedAcc(st)
			add(fmt.Sprintf("staged/acc/%d/%d", i, keys), k, err)
		}
	}
	mla := kernels.FlashShape{Heads: 16, KVHeads: 1, Dim: 576, Rows: 1, Page: 64, MLA: 512, Scale: 0.07}
	st := kernels.PagedStagedAt(mla, 2048, 8)
	k, err := kernels.PagedStagedScores(st, 64)
	add("staged/mla/scores", k, err)
	k, err = kernels.PagedStagedSoftmax(st, 64)
	add("staged/mla/softmax", k, err)
	k, err = kernels.PagedStagedAcc(st)
	add("staged/mla/acc", k, err)
	k, err = kernels.FlashAttentionMergeWide(kernels.FlashShape{Heads: 8, Dim: 64, Rows: 16, Splits: 4, Map: true})
	add("merge/map", k, err)
	return out
}

// TestAttnIRUnchanged holds every paged decode attention kernel's IR to
// testdata/attn_ir.sha, recorded before attention was formed in compose
// (-update-attn rewrites it).
func TestAttnIRUnchanged(t *testing.T) {
	got := attnSweep()
	var keys []string
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s %s\n", k, got[k])
	}
	path := filepath.Join("testdata", "attn_ir.sha")
	if *updateAttn {
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update-attn to create it)", err)
	}
	wl := strings.Split(strings.TrimSpace(string(want)), "\n")
	gl := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(wl) != len(gl) {
		t.Fatalf("%d kernels, the golden has %d", len(gl), len(wl))
	}
	built := 0
	for i := range wl {
		if wl[i] != gl[i] {
			t.Errorf("moved: %s (was %s)", gl[i], wl[i])
		}
		if !strings.HasSuffix(gl[i], " refused") {
			built++
		}
	}
	if built < 60 {
		t.Fatalf("only %d kernels built: the sweep does not reach the emitters", built)
	}
	t.Logf("%d attention kernels, %d built, unchanged", len(gl), built)
}
