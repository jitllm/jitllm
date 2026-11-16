package convert

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// reshapePointConv lays a 1x1 convolution [1 1 in out] out as the matrix
// [k=in, rows=out] it is: the bytes already are, a 1x1 kernel having no
// spatial axis to move.
func reshapePointConv(t *jlm.Tensor) error {
	if t.NDim == 2 {
		return nil
	}
	if t.NDim != 4 || t.Dims[0] != 1 || t.Dims[1] != 1 {
		return fmt.Errorf("convert: %s is %v, want a [1 1 in out] convolution", t.Name, t.Dims[:t.NDim])
	}
	in, out := t.Dims[2], t.Dims[3]
	t.NDim = 2
	t.Dims = [4]uint64{in, out}
	return nil
}

// normFrameRows writes the begin and end rows as after_rms leaves them:
// the merger normalises the picture's rows with them, and a row's RMSNorm
// is its own, so the head normalises the grid's rows and frames them with
// rows normalised here (in float64, the reference's f32 rounding at the
// end).
func normFrameRows(s *jlm.Source, eps float32) error {
	var w []float32
	rows := map[jlm.Role]*jlm.Tensor{}
	for i := range s.Tensors {
		switch t := &s.Tensors[i]; t.Role {
		case jlm.RoleVProjNorm:
			v, err := normF32(t)
			if err != nil {
				return err
			}
			w = v
		case jlm.RoleVImgBegin, jlm.RoleVImgEnd:
			rows[t.Role] = t
		}
	}
	if w == nil || len(rows) != 2 {
		return fmt.Errorf("convert: hunyuanvl: after_rms and the begin and end rows are %d and %d", len(w), len(rows))
	}
	for _, t := range rows {
		v, err := normF32(t)
		if err != nil {
			return err
		}
		if len(v) != len(w) {
			return fmt.Errorf("convert: %s holds %d values, after_rms %d", t.Name, len(v), len(w))
		}
		var ss float64
		for _, x := range v {
			ss += float64(x) * float64(x)
		}
		r := 1 / math.Sqrt(ss/float64(len(v))+float64(eps))
		for j, x := range v {
			v[j] = float32(float64(x)*r) * w[j]
		}
		setF32Dims(t, v, uint64(len(v)))
	}
	return nil
}
