package convert

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryTensorHasARole walks every model in the model directory and requires that each
// tensor maps onto a role and a type the container defines.
//
// It is the writing-side gate on "a container needs no knowledge of the
// source format": a name the converter cannot place would otherwise be
// dropped, and a dropped tensor is a model that loads and is wrong. So
// identify() refuses, and this says which names would make it refuse.
//
// A model whose architecture is unsupported is expected to have tensors with no
// role (state-space blocks, for instance): those are reported, not failed,
// because the converter refuses them earlier and for a better reason.
func TestEveryTensorHasARole(t *testing.T) {
	paths := testmodels.Glob("*.gguf")
	if len(paths) == 0 {
		t.Skip("MODEL MISSING: no *.gguf in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	seenRole := map[jlm.Role]int{}
	unplaced := map[string]string{} // name -> first model that has it
	files, tensors := 0, 0
	// One file can be singled out, and there is deliberately no size cap:
	// gguf.Open reads only the header, so size is never what makes this slow
	// (a slow run means a busy box), and a cap would skip the large models.
	pick := os.Getenv("JITLLM_ROLE_MODEL")
	for _, p := range paths {
		if pick != "" && !strings.Contains(p, pick) {
			continue
		}
		f, err := gguf.Open(p)
		if err != nil {
			continue
		}
		files++
		_, cfgErr := configOf(f)
		// configOf cannot read a bare mmproj (no text config), and the clip
		// override must not swallow a real refusal: an unimplemented projector
		// is a scope decision (ErrNotImplemented), not a missing role.
		supported := cfgErr == nil
		var vis *jlm.Vision
		if f.Arch() == "clip" {
			// A bare mmproj has no text config: visionOf is where the projector
			// list lives and where an unimplemented one is refused.
			var visErr error
			vis, visErr = visionOf(f)
			supported = !errors.Is(visErr, ErrNotImplemented)
		}
		renames := towerNames(f, vis)
		for i := range f.Tensors {
			n := f.Tensors[i].Name
			if r, ok := renames[n]; ok {
				n = r
			}
			tensors++
			// Qwen3.5's and Qwen3.8-Flash-Next's split gates never reach the
			// name table: sourceOf fuses them into ssm_ba first
			// (fuseBetaAlpha, gated by its own test).
			if (f.Arch() == "qwen35" || f.Arch() == "qwen35moe" || f.Arch() == "qwen4exp") && splitBetaAlpha(n) {
				continue
			}
			// A tower's tensors are named as conversion names them
			// (towerRoleOf), and the ones it reads into another or not at all
			// are accounted for by the same predicates sourceOf uses. The
			// newer towers' fused q|k|v and deepstack mergers are resolved by
			// familyTowerTensors (gated by the towers' own end-to-end gates).
			var role jlm.Role
			var err error
			switch {
			case vis != nil && (towerUnread(vis, n) || towerGathered(n) || familyTowerName(n)):
				continue
			case vis != nil:
				role, _, _, err = towerRoleOf(vis, n)
			default:
				role, _, _, err = identify(n)
			}
			if err != nil {
				// A tensor-level ErrNotImplemented is a scope decision too, even
				// for an architecture configOf accepts.
				if errors.Is(err, ErrNotImplemented) {
					unplaced[n] = filepath.Base(p)
				} else if supported {
					t.Errorf("%s: %v", filepath.Base(p), err)
				} else {
					unplaced[n] = filepath.Base(p)
				}
				continue
			}
			// A state-space in_proj is split at conversion (unfuseSSD, unfuseLFM2,
			// unfuseMamba1): its role names the source and stays reserved, since
			// no container carries one.
			if !role.Valid() && role != jlm.RoleSSMInProj {
				t.Errorf("%s: %q mapped to %v, which the schema does not name", filepath.Base(p), n, role)
			}
			seenRole[role]++
			// The type check is scoped the same way: an unsupported
			// architecture may also carry a weight type this engine has no
			// kernel for, and the refusal for that belongs to configOf.
			// DeepSeek V4's token table is I32 ids in the file and F32 in the
			// container: sourceOf rewrites it (ds4HashTable) rather than
			// storing the file's type. A ternary type with no code (TQ1_0,
			// I2_S) is checked as what the converter re-packs it to.
			ty := f.Tensors[i].Type
			if _, rt, ok, err := ternaryGGUF(ty, f.Bytes(&f.Tensors[i]), elemsOf(f.Tensors[i].Dims)); err != nil {
				t.Errorf("%s: %q: %v", filepath.Base(p), n, err)
			} else if ok {
				ty = rt
			}
			if _, err := typeOf(ty); err != nil && supported && role != jlm.RoleHashExperts {
				t.Errorf("%s: %q: %v", filepath.Base(p), n, err)
			}
		}
		f.Close()
	}
	if files == 0 || tensors == 0 {
		t.Fatal("no tensors examined -- this gate proved nothing")
	}
	t.Logf("%d files, %d tensors, %d distinct roles used", files, tensors, len(seenRole))
	for n, m := range unplaced {
		t.Logf("no role for %-34s (%s, architecture not implemented)", n, m)
	}

	// And the inverse mapping is checked: a type code that does not
	// round-trip is a container the engine cannot run.
	for ty := jlm.TypeF32; ty <= jlm.TypeQ6S; ty++ {
		st, ok := sourceType(ty)
		if !ok {
			t.Errorf("%v has no source type", ty)
			continue
		}
		back, err := typeOf(st)
		if err != nil || back != ty {
			t.Errorf("%v -> %v -> %v, %v", ty, st, back, err)
		}
	}
}
