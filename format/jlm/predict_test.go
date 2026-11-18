package jlm

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// predictSource is migrateSource's one-block model with a prediction block
// after it: block 1 stores the projection, and the statement states stated
// blocks.
func predictSource(stated int) *Source {
	src := migrateSource(false, false)
	k := uint64(src.Config.NEmbd)
	e := Tensor{Role: RolePredictProj, Block: 1, Index: -1, Type: TypeF32, NDim: 2, Data: make([]byte, 4*2*k*k),
		Name: RolePredictProj.String()}
	e.Dims[0], e.Dims[1] = 2*k, k
	src.Tensors = append(src.Tensors, e)
	src.Config.Blocks = BlockGraphs{Graphs: []string{"a"}, Of: make([]uint16, stated)}
	return src
}

// TestPredictionBlocksAreTheStatedBlocks: the record keeps no count of
// prediction blocks; they are the blocks the statement states past the
// trunk. A v31 or v32 container's count is held to its statement on the
// way to v33, which gives the fresh v33 container byte for byte; a count
// the statement does not cover is refused for a reconversion, and a v33
// container storing a prediction block its statement leaves out does not
// open.
func TestPredictionBlocksAreTheStatedBlocks(t *testing.T) {
	fp := Fingerprint{Host: "test"}
	dir := t.TempDir()
	fresh := filepath.Join(dir, "fresh"+Ext)
	if _, err := Write(fresh, predictSource(2), fp); err != nil {
		t.Fatal(err)
	}
	f, err := Open(fresh)
	if err != nil {
		t.Fatal(err)
	}
	if n := f.Config().PredictionBlocks(); n != 1 {
		t.Fatalf("%d prediction blocks, want the one the statement states past the trunk", n)
	}
	f.Close()
	want, err := os.ReadFile(fresh)
	if err != nil {
		t.Fatal(err)
	}

	// From v31 (a stamp to v32, then the rewrite) and from v32, whose
	// containers carry the count too.
	for _, v := range []uint32{31, 32} {
		old := filepath.Join(dir, fmt.Sprintf("old%d%s", v, Ext))
		if _, err := writeAt(old, predictSource(2), fp, v); err != nil {
			t.Fatal(err)
		}
		if from, err := Upgrade(old, nil); err != nil || from != v {
			t.Fatalf("Upgrade of v%d: from %d, %v", v, from, err)
		}
		got, err := os.ReadFile(old)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("the upgraded v%d container (%d bytes) is not the fresh one (%d bytes)%s", v, len(got), len(want), firstDiff(got, want))
		}
	}

	// The violation: a step that only stamps v33 on the v32 record.
	stamp := func(v uint32) func(string) error {
		if v == 32 {
			return func(path string) error { return setVersion(path, 33) }
		}
		return migrationFor(v, nil)
	}
	sq := filepath.Join(dir, "stamped"+Ext)
	if _, err := writeAt(sq, predictSource(2), fp, 32); err != nil {
		t.Fatal(err)
	}
	// The stamped record misreads at v33: a later step refuses it, or the
	// chain ends in other bytes than the fresh container's.
	if _, err := upgradeWith(stamp, sq); err == nil {
		if bad, err := os.ReadFile(sq); err != nil {
			t.Fatal(err)
		} else if bytes.Equal(bad, want) {
			t.Fatal("a chain that only stamped v33 gave the fresh container: the gate does not see the step")
		}
	}

	// A v31 record counting a block its statement leaves out.
	short := migrateSource(false, false)
	short.Config.Blocks = BlockGraphs{Graphs: []string{"a"}, Of: []uint16{0}}
	short.Config.legacyPredict = 1
	sp := filepath.Join(dir, "short"+Ext)
	if _, err := writeAt(sp, short, fp, 31); err != nil {
		t.Fatal(err)
	}
	var ue *UpgradeError
	if _, err := Upgrade(sp, nil); !errors.As(err, &ue) || !ue.Reconvert {
		t.Fatalf("a v31 count of 1 over a one-block statement upgraded: %v", err)
	} else {
		t.Logf("a count the statement does not cover: %v", err)
	}

	left := filepath.Join(dir, "left"+Ext)
	if _, err := Write(left, predictSource(1), fp); err != nil {
		t.Fatal(err)
	}
	if f, err := Open(left); err == nil {
		f.Close()
		t.Fatal("a container storing a prediction block its statement leaves out opened")
	} else if !strings.Contains(err.Error(), "past the 1 blocks its graphs state") {
		t.Fatalf("the refusal %q does not name the statement", err)
	} else {
		t.Logf("a prediction block left out: %v", err)
	}
}
