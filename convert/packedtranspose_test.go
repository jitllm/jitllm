package convert

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestATransposeMovesAPackedTypesRows: a transpose of a packed tensor that
// leaves its fastest axis in place moves whole rows of blocks, byte for byte,
// and one that moves that axis is refused.
func TestATransposeMovesAPackedTypesRows(t *testing.T) {
	const k = 64
	vals := make([]float32, 2*3*k)
	for i := range vals {
		vals[i] = float32(i%97) * 0.25
	}
	data := quantizeQ8Rows(vals, k, 6)
	row := len(data) / 6
	v := mjVal{shape: []uint64{2, 3, k}, typ: jlm.TypeQ8, data: data}
	out, err := mjTranspose(v, []int64{1, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.shape) != 3 || out.shape[0] != 3 || out.shape[1] != 2 || out.shape[2] != k {
		t.Fatalf("shape %v", out.shape)
	}
	// out[i][j] is in[j][i]: in row j*3+i.
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			got := out.data[(i*2+j)*row : (i*2+j+1)*row]
			want := data[(j*3+i)*row : (j*3+i+1)*row]
			if !bytes.Equal(got, want) {
				t.Fatalf("row [%d][%d] is not source row %d", i, j, j*3+i)
			}
		}
	}
	if _, err := mjTranspose(v, []int64{0, 2, 1}); err == nil {
		t.Error("a transpose moving a packed tensor's blocks was run")
	}
}
