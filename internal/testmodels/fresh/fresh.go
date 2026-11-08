// Package fresh makes a test's container one this build reads.
//
// The engine reads nothing but containers and refuses one of another container
// version, so a container an earlier build left in the model directory fails
// every gate that names it until someone converts it by hand. The suite
// converts instead (AGENTS.md, Scope: "the test suite converts"): a container
// that does not open is reconverted from the GGUF beside it.
//
// It is its own package rather than part of testmodels because it imports the
// converter, whose own tests import testmodels. Tests only, as testmodels is
// (TestNoProductionCodeImportsTestModels covers both).
package fresh

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// Container makes the container at path one this build reads: a container that
// does not open and has X.gguf beside it is reconverted from that, and a
// merged vision container X-vlm.jlm written before the converter stated its
// projector's tower is reconverted from X.gguf and mmproj-X.gguf, and one
// whose chat template states no tool form is reconverted too. An absent
// container with its source beside it is converted from it: a missing
// artefact is a task. A container that opens and states all of that, a path
// that is not a container and one with no source are left alone, and the
// caller's load says what is wrong.
func Container(path string) error {
	if !strings.HasSuffix(path, jlm.Ext) {
		return nil
	}
	return reconvert(path)
}

// Must is Container failing t.
func Must(t testing.TB, path string) {
	t.Helper()
	if err := Container(path); err != nil {
		t.Fatal(err)
	}
}

func reconvert(path string) error {
	c, openErr := jlm.Open(path)
	if openErr == nil {
		stated := len(c.Config().Acts) > 0
		v := c.Vision()
		tower := v == nil || v.Tower != nil || !convert.StatesTower(v.Projector)
		voc := c.Vocab()
		tools := voc == nil || len(voc.Templates) == 0 || len(voc.Tools.Calls) > 0
		cfg := c.Config()
		streams := (cfg.AttnResBlock == 0 && cfg.HCLowRank == 0 || cfg.Streams.N > 1) &&
			(cfg.NGramBlock == 0 || convert.StatesPre(cfg)) &&
			(cfg.IdxHeads == 0 || cfg.Selection != nil || len(cfg.CompKinds) != 0)
		labels := cfg.DecisionLabels != nil || !convert.StatesDecisionLabels(cfg.Decision)
		rev := cfg.Revision >= jlm.StatementRevision || (cfg.SWAWindow == 0 || cfg.YarnFactor <= 1 && v == nil) &&
			(cfg.Revision >= 2 || !storesAny(c, jlm.RoleAttnSubNorm, jlm.RoleFFNSubNorm)) &&
			(cfg.Revision >= 3 || cfg.RopeSections == [4]uint32{}) &&
			(cfg.Revision >= 4 || !storesPast(c, jlm.RoleFFNUp))
		c.Close()
		switch {
		case stated && tower && tools && streams && rev && labels:
			return nil
		case !labels:
			openErr = fmt.Errorf("it states no label readout for its decision kind")
		case !rev:
			openErr = fmt.Errorf("it states nothing of its sliding layers or its rotary layout")
		case !streams:
			openErr = fmt.Errorf("it states no residual streams")
		case stated && tower:
			openErr = fmt.Errorf("its chat template states no tool form")
		case !stated:
			// Written before the converter stated activations: it opens, and
			// the front end refuses its blocks.
			openErr = fmt.Errorf("it states no activations")
		default:
			openErr = fmt.Errorf("it states no tower for its %v projector", v.Projector)
		}
	}
	base := strings.TrimSuffix(path, jlm.Ext)
	src, mm := base+".gguf", ""
	if text, ok := strings.CutSuffix(base, "-vlm"); ok {
		src = text + ".gguf"
		mm = filepath.Join(filepath.Dir(text), "mmproj-"+filepath.Base(text)+".gguf")
		if _, err := os.Stat(mm); err != nil {
			return nil
		}
	}
	if _, err := os.Stat(src); err != nil {
		return nil
	}
	// A private name and a rename: another test binary, or another test of
	// this one, must not open a half-written container or write the same
	// file. The private name keeps the extension, as the converter writes
	// nothing else.
	tmp := filepath.Join(filepath.Dir(path),
		fmt.Sprintf(".%s.%d.%d%s", strings.TrimSuffix(filepath.Base(path), jlm.Ext), os.Getpid(), time.Now().UnixNano(), jlm.Ext))
	if _, err := convert.FromGGUFs(src, mm, tmp, jlm.Fingerprint{Host: "test"}); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("%s does not open (%v), and reconverting it from %s failed: %w",
			filepath.Base(path), openErr, filepath.Base(src), err)
	}
	return os.Rename(tmp, path)
}

// storesPast reports that a block of c past the trunk's stores r.
func storesPast(c *jlm.File, r jlm.Role) bool {
	for bi := c.Config().NLayer; bi < c.H.NBlocks; bi++ {
		if c.Has(r, int32(bi), -1) {
			return true
		}
	}
	return false
}

// storesAny reports that some block of c stores one of rs.
func storesAny(c *jlm.File, rs ...jlm.Role) bool {
	for bi := range c.Config().NLayer {
		for _, r := range rs {
			if c.Has(r, int32(bi), -1) {
				return true
			}
		}
	}
	return false
}
