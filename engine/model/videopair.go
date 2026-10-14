package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
)

// A video's frames enter a Qwen-VL tower two at a time: its patch embedding
// is a conv3d whose kernel spans temporal_patch_size = 2 frames, a plane of
// weights each. A still picture reads one frame against both planes, which
// the container folds into their sum; a video's pair reads two, so the second
// plane is kept beside the sum (convert.foldPatchPlanes) and a patch row is
//
//	x0*(W0+W1) + (x1-x0)*W1 = x0*W0 + x1*W1
//
// -- the sum's product the still path already makes, plus the second plane
// against the frames' difference. Every pixel passes through generated code:
// the gather (Copy32JIT), the difference and the add (Axpy32JIT), the
// matmul.

// patchPair adds the second frame's term to the patch rows x, the first
// frame's already in them from gather (np rows of k pk, pw pixels across).
func (s *State) patchPair(x, gather []float32, pw, np, pk int) error {
	v := s.vis
	t, c := v.t, v.t.Cfg
	if t.patchW1.e == nil {
		return fmt.Errorf("model: this container's tower has no second temporal plane, so it cannot read a " +
			"video's frame pairs; convert the model again with its mmproj")
	}
	if len(v.px2) != np*c.PatchSz*c.PatchSz*3 {
		return fmt.Errorf("model: a frame pair of %d and %d pixel values", np*c.PatchSz*c.PatchSz*3, len(v.px2))
	}
	if len(v.gather2) < np*pk {
		v.gather2 = make([]float32, np*pk)
	}
	g2 := v.gather2[:np*pk]
	P := c.PatchSz
	gw := pw / P
	for r := 0; r < np; r++ {
		p := s.patchAt(r)
		py, pxi := p/gw, p%gw
		at := (py*P*pw + pxi*P) * 3
		for ch := 0; ch < 3; ch++ {
			nn.Copy32JIT(g2[r*pk+ch*P*P:r*pk+(ch+1)*P*P], P, 1, v.px2[at+ch:], 3*pw, 3, P, P)
		}
	}
	nn.Axpy32JIT(g2, gather[:np*pk], -1)
	E := c.NEmbd
	if len(v.pair) < np*E {
		v.pair = make([]float32, np*E)
	}
	pair := v.pair[:np*E]
	s.jit.NewInput()
	if err := s.mm(pair, t.patchW1, g2, np); err != nil {
		return err
	}
	nn.Axpy32JIT(x, pair, 1)
	return nil
}

// framePair names a video's temporal pair of preprocessed frames of a
// gh x gw patch grid: one Picture, its key over both frames.
func (c TowerConfig) framePair(px, px2 []float32, gh, gw int) (*Picture, error) {
	if len(px) != len(px2) {
		return nil, fmt.Errorf("model: a frame pair of %d and %d pixel values", len(px), len(px2))
	}
	pc, err := c.picture(px, gh, gw)
	if err != nil {
		return nil, err
	}
	pc.Pixels2 = px2
	pc.Key = c.imageKey(append(append(make([]float32, 0, 2*len(px)), px...), px2...), gh, gw)
	return pc, nil
}
