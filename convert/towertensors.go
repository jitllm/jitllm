package convert

import (
	"embed"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

//go:embed towertensors.json
var towerTensorsFS embed.FS

// towerTensors is towertensors.json: the tower tensors a file carries and no
// graph reads, each with the reason, and each projector's own names.
type towerTensors struct {
	Unread     map[string]string         `json:"unread"`
	Projectors map[string]towerProjector `json:"projectors"`
}

// towerProjector is a projector's own tensor vocabulary: the roles of names
// the shared table reads otherwise or not at all (one position means
// different things in different projectors: InternVL's mm.model.mlp.0 is a
// LayerNorm, llava's mm.0 a matrix), the renames into the shared table's
// names (a prefix or a part of the name), and the names it carries that
// nothing reads, by name or prefix. Indexed are its
// names of a numbered part outside the blocks: a name Prefix, N, ".", then a
// part Roles names takes that role at index N, N one of the tower's blocks
// (a tap, Qwen3-VL's deepstack mergers).
type towerProjector struct {
	Roles   map[string]string `json:"roles"`
	Indexed []struct {
		Prefix string            `json:"prefix"`
		Roles  map[string]string `json:"roles"`
	} `json:"indexed"`
	Renames []struct {
		Prefix string `json:"prefix"`
		Infix  string `json:"infix"`
		To     string `json:"to"`
	} `json:"renames"`
	Unread         []string `json:"unread"`
	UnreadPrefixes []string `json:"unread_prefixes"`
}

// towerVocab is a tower file's names: towertensors.json's shared list and
// its projector's entry, the roles resolved.
type towerVocab struct {
	unread map[string]string
	p      towerProjector
	roles  map[string]jlm.Role
	// indexed is p.Indexed's roles resolved, by prefix and part; blocks the
	// tower's block count, which bounds an index.
	indexed map[string]map[string]jlm.Role
	blocks  uint32
	text    tensorNames
}

// towerVocabOf is v's projector's tower names, over the text vocabulary.
func towerVocabOf(v *jlm.Vision, text tensorNames) (*towerVocab, error) {
	b, err := towerTensorsFS.ReadFile("towertensors.json")
	if err != nil {
		return nil, err
	}
	var t towerTensors
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, fmt.Errorf("convert: towertensors.json: %w", err)
	}
	n := &towerVocab{unread: t.Unread, p: t.Projectors[v.Projector.String()], roles: map[string]jlm.Role{},
		indexed: map[string]map[string]jlm.Role{}, blocks: v.NLayer, text: text}
	named := rolesByName()
	for _, ix := range n.p.Indexed {
		if ix.Prefix == "" || !strings.HasSuffix(ix.Prefix, ".") || len(ix.Roles) == 0 {
			return nil, fmt.Errorf("convert: towertensors.json: %s states an indexed prefix %q with no roles or no trailing dot", v.Projector, ix.Prefix)
		}
		n.indexed[ix.Prefix] = map[string]jlm.Role{}
		for part, role := range ix.Roles {
			r, ok := named[role]
			if !ok {
				return nil, fmt.Errorf("convert: towertensors.json: %s's %s%s is role %q, which the container does not know", v.Projector, ix.Prefix, part, role)
			}
			n.indexed[ix.Prefix][part] = r
		}
	}
	for name, role := range n.p.Roles {
		r, ok := named[role]
		if !ok {
			return nil, fmt.Errorf("convert: towertensors.json: %s's %s is role %q, which the container does not know", v.Projector, name, role)
		}
		n.roles[name] = r
	}
	return n, nil
}

// roleOf is the role a tower file's tensor takes: its projector's own name
// first, then the shared table's after the projector's renames. sourceOf
// and the role gate both read it, so the gate sees the names conversion
// sees.
func (n *towerVocab) roleOf(name string) (jlm.Role, int32, int32, error) {
	if r, ok := n.roles[name]; ok {
		return r, jlm.DenseBlock, -1, nil
	}
	for prefix, parts := range n.indexed {
		rest, ok := strings.CutPrefix(name, prefix)
		if !ok {
			continue
		}
		num, part, ok := strings.Cut(rest, ".")
		i, err := strconv.Atoi(num)
		if !ok || err != nil || i < 0 || num != strconv.Itoa(i) {
			return 0, 0, 0, fmt.Errorf("convert: %s: no index after %s", name, prefix)
		}
		r, known := parts[part]
		if !known {
			return 0, 0, 0, fmt.Errorf("convert: %s: %s names no %s part", name, prefix, part)
		}
		if uint32(i) >= n.blocks {
			return 0, 0, 0, fmt.Errorf("convert: %s taps block %d of %d", name, i, n.blocks)
		}
		return r, jlm.DenseBlock, int32(i), nil
	}
	for _, rn := range n.p.Renames {
		switch {
		case rn.Prefix != "" && strings.HasPrefix(name, rn.Prefix):
			return n.text.identify(rn.To + strings.TrimPrefix(name, rn.Prefix))
		case rn.Infix != "" && strings.Contains(name, rn.Infix):
			return n.text.identify(strings.Replace(name, rn.Infix, rn.To, 1))
		}
	}
	return n.text.identify(name)
}

// named reports whether the vocabulary names a role for a tower file's tensor.
func (n *towerVocab) named(name string) bool {
	_, _, _, err := n.roleOf(name)
	return err == nil
}

// unreadName reports a tower file's tensor that no graph here reads: the
// shared list's, and its projector's (an audio encoder shipped in the same
// file, a row the reference declares and never reads).
func (n *towerVocab) unreadName(name string) bool {
	if _, ok := n.unread[name]; ok {
		return true
	}
	for _, u := range n.p.Unread {
		if name == u {
			return true
		}
	}
	for _, p := range n.p.UnreadPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}
