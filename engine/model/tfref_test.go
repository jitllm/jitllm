//go:build jitllmbench

package model_test

import (
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestTeacherForcedAgainstReference feeds a fixed id sequence and compares the
// logits at every row from the prompt's last on with a reference's own -- a
// transformers run over the same ids -- argmax, margin and NMSE. It is the
// instrument for a real checkpoint whose llama.cpp graph is not the model's
// (RULE 7m): which engine is right at a confident disagreement is the
// reference's call.
//
//	JITLLM_REF_MODEL    a GGUF or container
//	JITLLM_REF_IDS      the whole sequence, prompt first, comma-separated
//	JITLLM_REF_NPROMPT  how many of them are the prompt
//	JITLLM_REF_LOGITS   the reference: raw little-endian f32, one vocabulary
//	                    row per position from the prompt's last on
//	JITLLM_TEST_DEVICES the device to place every block on, or unset for the host
//	JITLLM_REF_KVF16    1 for the f16 KV cache, else f32
func TestTeacherForcedAgainstReference(t *testing.T) {
	path := os.Getenv("JITLLM_REF_MODEL")
	if path == "" {
		t.Skip("set JITLLM_REF_MODEL")
	}
	var ids []int32
	for _, f := range strings.Split(os.Getenv("JITLLM_REF_IDS"), ",") {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, int32(v))
	}
	np, err := strconv.Atoi(os.Getenv("JITLLM_REF_NPROMPT"))
	if err != nil || np < 1 || np > len(ids) {
		t.Fatalf("JITLLM_REF_NPROMPT %q for %d ids", os.Getenv("JITLLM_REF_NPROMPT"), len(ids))
	}
	raw, err := os.ReadFile(os.Getenv("JITLLM_REF_LOGITS"))
	if err != nil {
		t.Fatal(err)
	}
	// A directory is a HuggingFace checkpoint, converted beside itself.
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		dst := strings.TrimSuffix(path, "/") + jlm.Ext
		if _, err := os.Stat(dst); err != nil {
			if _, err := convert.FromSafetensors(path, dst, jlm.Fingerprint{Host: "test"}); err != nil {
				t.Fatal(err)
			}
		}
		path = dst
	}
	m, err := model.Open(jlmOf(t, path), model.WithJITOptions(nn.WithGEMMExact(true), nn.WithTune(nn.TuneOff)),
		model.WithKVF16(os.Getenv("JITLLM_REF_KVF16") == "1"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	nv := m.Cfg.NVocab
	rows := len(ids) - np + 1
	if len(raw) != 4*rows*nv {
		t.Fatalf("the reference holds %d bytes, want %d rows of %d", len(raw), rows, nv)
	}
	ref := func(r int) []float32 {
		out := make([]float32, nv)
		for j := range out {
			out[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(r*nv+j):]))
		}
		return out
	}
	st := m.NewState(len(ids) + 1)
	defer st.Close()
	if spec := testdevices.One(""); spec != "" {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil {
			t.Fatal(err)
		}
		defer gpu.Close()
		st.SetDeviceLayers(gpu, -1)
		t.Logf("%d of %d blocks on %s", st.GPULayers(), m.Cfg.NLayer, spec)
	}
	top2 := func(v []float32) (int, float32) {
		b, s := 0, float32(math.Inf(-1))
		for j := range v {
			if m.Vocab.IsEOG(int32(j)) {
				continue
			}
			if v[j] > v[b] || m.Vocab.IsEOG(int32(b)) {
				b = j
			}
		}
		for j := range v {
			if j != b && !m.Vocab.IsEOG(int32(j)) && v[j] > s {
				s = v[j]
			}
		}
		return b, v[b] - s
	}
	lg, err := st.Prefill(ids[:np])
	if err != nil {
		t.Fatal(err)
	}
	agree, confident, worst := 0, 0, 0.0
	for r := 0; r < rows; r++ {
		want := ref(r)
		var num, den float64
		for j := range want {
			d := float64(lg[j] - want[j])
			num, den = num+d*d, den+float64(want[j])*float64(want[j])
		}
		nmse := num / den
		worst = max(worst, nmse)
		ja, jm := top2(lg)
		ra, rm := top2(want)
		fed := "-"
		if np+r < len(ids) {
			fed = strconv.Itoa(int(ids[np+r]))
		}
		switch {
		case ja == ra:
			agree++
		case jm > 0.5 && rm > 0.5:
			confident++
		}
		t.Logf("pos %2d fed %6s  ref %6d %q (margin %.3f)  jitllm %6d %q (margin %.3f)  NMSE %.2e",
			r, fed, ra, m.Vocab.Text(int32(ra)), rm, ja, m.Vocab.Text(int32(ja)), jm, nmse)
		if np+r >= len(ids) {
			break
		}
		if lg, err = st.Forward(ids[np+r]); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d of %d argmaxes agree with the reference, %d confident disagreements, worst logit NMSE %.3e",
		agree, rows, confident, worst)
}
