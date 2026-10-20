package oldkernels_test

import (
	"crypto/sha256"
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

var updatePrefill = flag.Bool("update-prefill", false, "rewrite testdata/prefill_ir.sha")

// irHash is a kernel's IR, every field of every op and parameter, hashed. Two
// kernels with the same hash are the same program for every lowerer.
func irHash(k *ir.Kernel) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%v|%d|%v\n", k.Name, k.Group, k.Lanes, k.Params)
	for _, o := range k.Ops {
		fmt.Fprintf(h, "%d %d %v %d %v", o.Kind, o.Type, o.Args, o.Imm, o.Vargs)
		if o.Tile != nil {
			fmt.Fprintf(h, " %+v", *o.Tile)
		}
		fmt.Fprintln(h)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:12])
}

// contiguousPrefill is every contiguous prefill attention kernel at the
// configurations that reach a different code path: tiles, windows (sliding
// and chunked), transposed and row-major K, GQA, softcap. The paged forms
// share their builders; this pins that the contiguous forms did not move.
func contiguousPrefill(t *testing.T) map[string]*ir.Kernel {
	out := map[string]*ir.Kernel{}
	add := func(name string, k *ir.Kernel, err error) {
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out[name] = k
	}
	for _, c := range []struct{ rows, qt, kt, ks, w int }{
		{1, 1, 1, 0, 0}, {1, 1, 1, 2049, 0}, {1, 1, 1, 2049, 97}, {32, 4, 2, 2049, 0},
		{32, 2, 4, 0, 0}, {64, 8, 1, 2049, 97}, {64, 4, 4, 2049, -128}, {16, 1, 1, 2049, 0},
	} {
		k, err := kernels.AttnScoresTiledW(8, 64, 256, 2, 2064, .125, c.rows, c.qt, c.kt, c.ks, c.w)
		add(fmt.Sprintf("scorestiled/r%d/q%d/k%d/s%d/w%d", c.rows, c.qt, c.kt, c.ks, c.w), k, err)
	}
	k, err := kernels.AttnScoresTiled(16, 576, 576, 16, 2064, .04, 32, 2, 2, 0)
	add("scorestiled/mla", k, err)
	for _, c := range []struct{ rows, qt int }{{1, 1}, {32, 1}, {32, 4}, {64, 8}} {
		k, err := kernels.AttnAccTiled(8, 64, 256, 2, 2064, c.rows, c.qt)
		add(fmt.Sprintf("acctiled/r%d/q%d", c.rows, c.qt), k, err)
	}
	k, err = kernels.AttnAccTiled(16, 512, 576, 16, 2064, 32, 4)
	add("acctiled/mla", k, err)
	for _, c := range []struct{ nt, w int }{{1, 0}, {2, 0}, {4, 97}, {2, -256}} {
		k, err := kernels.AttnScoresMMAW(8, 64, 256, 2, 2064, .125, 64, 2049, c.nt, c.w)
		add(fmt.Sprintf("scoresmma/n%d/w%d", c.nt, c.w), k, err)
	}
	for _, c := range []struct{ hd, kvd, gqa, ks, mt, nt, w int }{
		{128, 1024, 4, 2049, 1, 2, 0}, {128, 1024, 4, 2049, 2, 2, 100}, {64, 512, 4, 2049, 1, 1, -64},
		{576, 576, 16, 0, 1, 2, 0}, {128, 1024, 4, 0, 2, 1, 0},
	} {
		k, err := kernels.AttnScoresMMA70(32, c.hd, c.kvd, c.gqa, 2064, .09, 64, c.ks, c.mt, c.nt, c.w)
		add(fmt.Sprintf("scoresmma70/d%d/kv%d/s%d/m%dn%d/w%d", c.hd, c.kvd, c.ks, c.mt, c.nt, c.w), k, err)
	}
	for _, c := range []struct{ hd, kvd, gqa, mt, nt int }{{128, 1024, 4, 1, 2}, {128, 1024, 4, 2, 2}, {512, 576, 16, 2, 2}, {64, 512, 8, 2, 1}} {
		k, err := kernels.AttnAccMMA70(32, c.hd, c.kvd, c.gqa, 2064, 64, c.mt, c.nt)
		add(fmt.Sprintf("accmma70/d%d/kv%d/m%dn%d", c.hd, c.kvd, c.mt, c.nt), k, err)
	}
	for _, s := range []kernels.FlashPrefill70Shape{
		{Heads: 32, KVHeads: 8, Dim: 64, Rows: 128, KStride: 2049, Scale: .125},
		{Heads: 16, KVHeads: 8, Dim: 128, Rows: 64, KStride: 2049, Scale: .088, Window: 100},
		{Heads: 8, KVHeads: 4, Dim: 128, Rows: 128, KStride: 2049, Scale: .088, Softcap: 50},
	} {
		k, err := kernels.FlashPrefill70(s)
		add(fmt.Sprintf("flash70/h%d-%d/d%d/r%d/w%d/c%g", s.Heads, s.KVHeads, s.Dim, s.Rows, s.Window, s.Softcap), k, err)
		for _, hd := range []int{s.Dim, 256} {
			st := s
			st.Dim = hd
			tl, ok := kernels.FlashTileFor(hd)
			if !ok {
				t.Fatalf("no tile at %d", hd)
			}
			k, err := kernels.FlashPrefillTile(st, tl)
			add(fmt.Sprintf("flashtile/h%d-%d/d%d/r%d/w%d/c%g/sg%d/tk%d", st.Heads, st.KVHeads, st.Dim, st.Rows, st.Window, st.Softcap, tl.SG, tl.TK), k, err)
		}
	}
	return out
}

// TestContiguousPrefillIRUnchanged hashes the contiguous prefill kernels'
// IR against testdata/prefill_ir.sha, written before their paged forms were
// added (-update-prefill rewrites it; a moved hash is a changed contiguous
// kernel, which is the finding).
func TestContiguousPrefillIRUnchanged(t *testing.T) {
	ks := contiguousPrefill(t)
	names := make([]string, 0, len(ks))
	for n := range ks {
		names = append(names, n)
	}
	sort.Strings(names)
	var b strings.Builder
	for _, n := range names {
		fmt.Fprintf(&b, "%s %s\n", n, irHash(ks[n]))
	}
	path := filepath.Join("testdata", "prefill_ir.sha")
	if *updatePrefill {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update-prefill to create it)", err)
	}
	if string(want) == b.String() {
		return
	}
	got := map[string]string{}
	for _, l := range strings.Split(b.String(), "\n") {
		if f := strings.Fields(l); len(f) == 2 {
			got[f[0]] = f[1]
		}
	}
	for _, l := range strings.Split(string(want), "\n") {
		f := strings.Fields(l)
		if len(f) != 2 {
			continue
		}
		if got[f[0]] != f[1] {
			t.Errorf("%s: IR moved (%s -> %s)", f[0], f[1], got[f[0]])
		}
		delete(got, f[0])
	}
	for n := range got {
		t.Errorf("%s: not in the golden", n)
	}
}
