package convert

import (
	"embed"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

//go:embed tensornames.json
var tensorNamesFS embed.FS

// tensorNames is the source format's tensor-name vocabulary
// (tensornames.json), each name's role resolved. A container identifies a
// tensor by (Role, Block, Index), so a conversion reads it once. An
// unrecognised name is a refusal, not a skip, because a tensor dropped at
// conversion is a model that loads and is wrong.
type tensorNames map[string]jlm.Role

// readTensorNames reads tensornames.json, refusing a role the container does
// not name.
func readTensorNames() (tensorNames, error) {
	b, err := tensorNamesFS.ReadFile("tensornames.json")
	if err != nil {
		return nil, err
	}
	var named map[string]string
	if err := json.Unmarshal(b, &named); err != nil {
		return nil, fmt.Errorf("convert: tensornames.json: %w", err)
	}
	roles := rolesByName()
	v := make(tensorNames, len(named))
	for n, rn := range named {
		r, ok := roles[rn]
		if !ok {
			return nil, fmt.Errorf("convert: tensornames.json: %q is role %q, which the container does not know", n, rn)
		}
		v[n] = r
	}
	return v, nil
}

// roleOf is name's role: the vocabulary's, a fused input projection's, or a
// decision head's.
func (v tensorNames) roleOf(name string) (jlm.Role, bool) {
	if r, ok := v[name]; ok {
		return r, true
	}
	// A fused input projection is split at conversion (unfuseScan,
	// unfuseShortConv); its role is reserved and never stored.
	if name == "ssm_in.weight" || name == "shortconv.in_proj.weight" {
		return jlm.RoleSSMInProj, true
	}
	return decisionRoleOf(name)
}

// decisionRoleOf is a decision head's tensor, named as its role is
// (format/jlm/decisionroles.go), with ".weight" on all but the biases and
// the scales.
func decisionRoleOf(name string) (jlm.Role, bool) {
	for r := jlm.RoleDecHiddenNorm; r <= jlm.RoleDecFFNDownBias; r++ {
		if !r.Valid() {
			continue
		}
		n := r.String()
		if strings.HasSuffix(n, ".bias") || r == jlm.RoleDecScales {
			if name == n {
				return r, true
			}
			continue
		}
		if len(name) == len(n)+len(".weight") && strings.HasPrefix(name, n) && strings.HasSuffix(name, ".weight") {
			return r, true
		}
	}
	return 0, false
}

// indexedRole is the per-expert form: "ffn_gate.3.weight" is expert 3's gate.
func indexedRole(name string) (jlm.Role, bool) {
	switch name {
	case "ffn_gate":
		return jlm.RoleExpGate, true
	case "ffn_up":
		return jlm.RoleExpUp, true
	case "ffn_down":
		return jlm.RoleExpDown, true
	}
	return 0, false
}

// retargets is the roles whose name, in a source of the GGUF architecture
// arch, means what another role does (the library's retarget statement,
// each with its reason): qwen3next's post_attention_norm is the FFN's input
// norm, where gemma2/gemma3's tensor of that name is applied to the
// attention output. An entry that holds of one layout names the tensor that
// marks it, and applies where stored says the source holds it (nil: no
// source to ask). Resolving them here keeps the source's ambiguity out of
// the container.
func retargets(arch string, stored func(string) bool) (map[jlm.Role]jlm.Role, error) {
	st, _, err := ggufStates(arch)
	if err != nil || st == nil || len(st.Retarget) == 0 {
		return nil, err
	}
	named := rolesByName()
	out := map[jlm.Role]jlm.Role{}
	for _, m := range st.Retarget {
		if m.When != "" && (stored == nil || !stored(m.When)) {
			continue
		}
		out[named[m.From]] = named[m.To]
	}
	return out, nil
}

// rolesByName is every container role by its name.
func rolesByName() map[string]jlm.Role {
	named := map[string]jlm.Role{}
	for i := 1; i <= math.MaxUint16; i++ {
		if r := jlm.Role(i); r.Valid() {
			named[r.String()] = r
		}
	}
	return named
}

// identify turns a source tensor name into the triple a container stores.
// Index is -1 unless the name carries one.
func (v tensorNames) identify(name string) (role jlm.Role, block int32, index int32, err error) {
	block, index = jlm.DenseBlock, -1
	rest := name
	// "v." marks the vision tower; its blocks are "v.blk.N.".
	vision := strings.HasPrefix(rest, "v.")
	// A decision head's blocks are "dec.blk.N.": model-level tensors
	// with the head block in Index, so they never meet a backbone block.
	if r, ok := strings.CutPrefix(rest, "dec.blk."); ok {
		i := strings.IndexByte(r, '.')
		n, e := strconv.ParseInt(r[:max(i, 0)], 10, 32)
		if i <= 0 || e != nil || n < 0 {
			return 0, 0, 0, fmt.Errorf("convert: %q: no head block index", name)
		}
		if role, ok := v.roleOf("dec." + r[i+1:]); ok {
			if role < jlm.RoleDecSelfNorm || role > jlm.RoleDecFFNDownBias || n >= 64 {
				return 0, 0, 0, fmt.Errorf("convert: %q: no head block role", name)
			}
			return jlm.RoleDecBlock(int(n), role), jlm.DenseBlock, -1, nil
		}
		return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary", name)
	}
	for _, p := range []string{"v.blk.", "blk."} {
		if strings.HasPrefix(rest, p) {
			r := rest[len(p):]
			i := strings.IndexByte(r, '.')
			if i <= 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: no block index", name)
			}
			n, e := strconv.Atoi(r[:i])
			if e != nil || n < 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: block index %q", name, r[:i])
			}
			block, rest = int32(n), r[i+1:]
			if vision {
				rest = "v." + rest
			}
			break
		}
	}
	if r, ok := v.roleOf(rest); ok {
		return r, block, index, nil
	}
	// The second temporal plane of Qwen2-VL's conv3d patch embedding, which the
	// GGUF spells as a second 4-D tensor. It is carried as index 1, folded into
	// index -1 by prepareTower (the sum a still image reads), and kept as
	// RoleVPatchEmbd2 for a video's second frame (foldPatchPlanes).
	if rest == "v.patch_embd.weight.1" {
		return jlm.RoleVPatchEmbd, block, 1, nil
	}
	// The indexed per-expert form.
	if f := strings.Split(rest, "."); len(f) == 3 && f[2] == "weight" {
		if r, ok := indexedRole(f[0]); ok {
			n, e := strconv.Atoi(f[1])
			if e == nil && n >= 0 {
				return r, block, int32(n), nil
			}
		}
	}
	return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
		"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
}

// typeOf maps a source weight type onto the container's own code.
//
// A type the container does not define is a refusal rather than a stored
// source number.
func typeOf(t quant.Type) (jlm.Type, error) {
	ty, ok := jlm.TypeOf(t)
	if !ok {
		return 0, fmt.Errorf("convert: weight type %s has no code in this container's vocabulary", t)
	}
	return ty, nil
}

// sourceType is typeOf's inverse, for the gate below.
func sourceType(t jlm.Type) (quant.Type, bool) { return jlm.SourceType(t) }
