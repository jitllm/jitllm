package convert

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// The transforms a model.json may name. Each is one shape of reshaping a
// source tensor on its way into the container, with its own gate; a family
// that needs a new shape adds one here, not a family entry.

// mjElem rewrites a tensor's values, its shape unchanged; it may change the
// tensor's type.
type mjElem func(e *jlm.Tensor, data []byte) ([]byte, error)

// mjElems rewrites a tensor's values by name.
func mjElems() map[string]mjElem {
	return map[string]mjElem{
		// plus_one stores 1 + weight: an RMSNorm that multiplies by (1 + weight).
		"plus_one": func(e *jlm.Tensor, data []byte) ([]byte, error) {
			if err := plusOneNorm(e, data); err != nil {
				return nil, err
			}
			return e.Data, nil
		},
		// neg_exp stores -exp(x): a log decay as the rate the gate kernel reads.
		"neg_exp": negExp,
	}
}

// mjShapes changes a tensor's shape, its bytes in place.
func mjShapes() map[string]hfSplit {
	return map[string]hfSplit{
		"squeeze": squeezeUnitAxes,
	}
}

// mjGatherFunc builds one tensor from a block's sources, each read as float32
// rows: the result's values and its [ne0, ne1].
type mjGatherFunc func(c *Config, src [][]float32) ([]float32, [2]uint64, error)

// mjGathers builds a tensor from several sources by name.
func mjGathers() map[string]mjGatherFunc {
	return map[string]mjGatherFunc{
		"interleave_groups": interleaveGroups,
	}
}

func negExp(e *jlm.Tensor, data []byte) ([]byte, error) {
	if e.Type != jlm.TypeF32 {
		return nil, fmt.Errorf("convert: %s is %v; neg_exp is float32 arithmetic", e.Name, e.Type)
	}
	out := make([]byte, 0, len(data))
	for i := 0; i+4 <= len(data); i += 4 {
		a := math.Float32frombits(binary.LittleEndian.Uint32(data[i:]))
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(float32(-math.Exp(float64(a)))))
	}
	return out, nil
}

// squeezeUnitAxes drops every axis of extent one: torch's [channels, 1, taps]
// convolution is the container's [taps, channels]. No byte moves.
func squeezeUnitAxes(e jlm.Tensor, data []byte, c *Config) ([]jlm.Tensor, error) {
	var dims [4]uint64
	n := 0
	for i := 0; i < int(e.NDim); i++ {
		if e.Dims[i] != 1 {
			dims[n] = e.Dims[i]
			n++
		}
	}
	if n == 0 {
		dims[0], n = 1, 1
	}
	e.NDim, e.Dims, e.Data = uint8(n), dims, data
	return []jlm.Tensor{e}, nil
}

// interleaveGroups joins two projections of one row per value head (a
// recurrent block's beta and alpha) as the engine reads them: per key head,
// its value heads' rows from the first, then theirs from the second. Each
// value head keeps its own pair whatever the heads' order.
func interleaveGroups(c *Config, src [][]float32) ([]float32, [2]uint64, error) {
	nv, nk := int(c.SSM.NHeadV), int(c.SSM.Groups)
	if len(src) != 2 || nk == 0 || nv%nk != 0 {
		return nil, [2]uint64{}, fmt.Errorf("interleave_groups joins two sources over %d value heads in %d groups", nv, nk)
	}
	if len(src[0]) != len(src[1]) || len(src[0])%nv != 0 || len(src[0]) == 0 {
		return nil, [2]uint64{}, fmt.Errorf("interleave_groups: sources of %d and %d values over %d value heads",
			len(src[0]), len(src[1]), nv)
	}
	k, rep := len(src[0])/nv, nv/nk
	out := make([]float32, 0, 2*nv*k)
	for g := 0; g < nk; g++ {
		out = append(out, src[0][g*rep*k:(g+1)*rep*k]...)
		out = append(out, src[1][g*rep*k:(g+1)*rep*k]...)
	}
	return out, [2]uint64{uint64(k), uint64(2 * nv)}, nil
}

// hfGatherStep is one model.json gather: the role it makes, its sources by
// block-relative name, and the transform.
type hfGatherStep struct {
	role jlm.Role
	from []string
	f    mjGatherFunc
}

// gatherBlocks runs each step on every block that carries its first source.
// A source left over -- a second without its first -- is left to identify,
// which refuses it by name.
func gatherBlocks(c *Config, prefix string, steps []hfGatherStep, names []string,
	read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	var out []jlm.Tensor
	used := map[string]bool{}
	for _, g := range steps {
		for _, n := range names {
			rest, ok := strings.CutPrefix(n, prefix)
			if !ok {
				continue
			}
			block, tail, err := leadingIndex(n, rest)
			if err != nil || tail != g.from[0] {
				continue
			}
			src := make([][]float32, len(g.from))
			bp := prefix + itoa(int(block)) + "."
			for i, f := range g.from {
				if src[i], err = read(bp + f); err != nil {
					return nil, nil, err
				}
				used[bp+f] = true
			}
			v, dims, err := g.f(c, src)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", n, err)
			}
			e := jlm.Tensor{Role: g.role, Block: block, Index: -1, NDim: 2, Type: jlm.TypeF32,
				Name: bp + g.from[0] + "(gathered)", Data: f32AsBytes(v)}
			e.Dims[0], e.Dims[1] = dims[0], dims[1]
			out = append(out, e)
		}
	}
	return out, used, nil
}
