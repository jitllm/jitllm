package convert

import (
	"sort"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// A tower file may name its blocks by a stage and a place in the stage,
// "v.blk.S.I.*". The container's blocks are one index, stage by stage
// (towerNames), each carrying its place (stagePlaces), where the projector's
// entry states blocks "staged".

// stagedBlockName parses "v.blk.S.I.rest", a block tensor named by its stage
// S and its place I in the stage.
func stagedBlockName(name string) (stage, idx int, rest string, ok bool) {
	r, found := strings.CutPrefix(name, "v.blk.")
	if !found {
		return 0, 0, "", false
	}
	f := strings.SplitN(r, ".", 3)
	if len(f) != 3 {
		return 0, 0, "", false
	}
	s, e1 := strconv.Atoi(f[0])
	i, e2 := strconv.Atoi(f[1])
	if e1 != nil || e2 != nil || s < 0 || i < 0 {
		return 0, 0, "", false
	}
	return s, i, f[2], true
}

// stagedBlocks is every block a file names by stage and place, in order: by
// stage, then by place in the stage.
func stagedBlocks(f *meta.File) [][2]int {
	seen := map[[2]int]bool{}
	var out [][2]int
	for i := range f.Tensors {
		s, b, _, ok := stagedBlockName(f.Tensors[i].Name)
		if !ok || seen[[2]int{s, b}] {
			continue
		}
		seen[[2]int{s, b}] = true
		out = append(out, [2]int{s, b})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a][0] != out[b][0] {
			return out[a][0] < out[b][0]
		}
		return out[a][1] < out[b][1]
	})
	return out
}

// towerNames is the renames a tower file's tensors take before the name
// table: where p states blocks "staged", every block named by a stage and a
// place in it becomes one index, stage by stage. nil for every other tower.
// sourceOf and the role gate both read it.
func towerNames(f *meta.File, p *projStates) map[string]string {
	if p == nil || p.Blocks != "staged" {
		return nil
	}
	flat := map[[2]int]int{}
	for i, sb := range stagedBlocks(f) {
		flat[sb] = i
	}
	out := map[string]string{}
	for i := range f.Tensors {
		n := f.Tensors[i].Name
		if s, b, rest, ok := stagedBlockName(n); ok {
			out[n] = "v.blk." + strconv.Itoa(flat[[2]int{s, b}]) + "." + rest
		}
	}
	return out
}

// stagePlaces appends each block's place, {stage, place in the stage}, so the
// tower's program can say which blocks open a stage.
func stagePlaces(s *jlm.Source, places [][2]int) {
	for i, p := range places {
		e := jlm.Tensor{Role: jlm.RoleVMNGeom, Block: int32(i), Index: -1,
			Name: "v.blk." + strconv.Itoa(i) + ".mn_geom"}
		setF32Dims(&e, []float32{float32(p[0]), float32(p[1])}, 2)
		s.Tensors = append(s.Tensors, e)
	}
}
