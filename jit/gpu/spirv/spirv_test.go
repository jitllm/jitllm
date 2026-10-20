package spirv_test

import (
	"encoding/binary"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/spirv"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestShuffleEncoding decodes the emitted module and states the shuffle's
// binary form independently of the code that wrote it.
//
// spirv-val is not installed everywhere, and this backend's first bugs were
// operand order. A shuffle carries an execution scope and a mask, both <id>s,
// so swapping them assembles and shuffles by the wrong distance.
func TestShuffleEncoding(t *testing.T) {
	k, err := mainkernels.SoftmaxRows(8, 64, 32, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := spirv.Emit(k)
	if err != nil {
		t.Fatal(err)
	}
	w := words(t, b)

	const (
		opCapability   = 17
		opTypeFloat    = 22
		opConstant     = 43
		opShuffleXor   = 346 // OpGroupNonUniformShuffleXor
		capNonUniform  = 61  // GroupNonUniform
		capShuffle     = 65  // GroupNonUniformShuffle
		scopeSubgroup  = 3
		headerWords    = 5
		shuffleOperand = 6 // Result Type, Result, Execution, Value, Mask
	)
	caps := map[uint32]bool{}
	constOf := map[uint32]uint32{} // result id -> literal value
	f32 := uint32(0)
	var masks []uint32
	for i := headerWords; i < len(w); {
		n, op := int(w[i]>>16), w[i]&0xFFFF
		if n == 0 {
			t.Fatalf("word %d: instruction of length 0", i)
		}
		switch op {
		case opCapability:
			caps[w[i+1]] = true
		case opTypeFloat:
			f32 = w[i+1]
		case opConstant:
			constOf[w[i+2]] = w[i+3]
		case opShuffleXor:
			if n != shuffleOperand {
				t.Fatalf("shuffle at word %d has %d words, want %d", i, n, shuffleOperand)
			}
			if w[i+1] != f32 {
				t.Fatalf("shuffle result type is %d, want the f32 type %d", w[i+1], f32)
			}
			if v, ok := constOf[w[i+3]]; !ok || v != scopeSubgroup {
				t.Fatalf("shuffle scope operand is id %d (= %d), want a constant Subgroup(%d)",
					w[i+3], v, scopeSubgroup)
			}
			masks = append(masks, constOf[w[i+5]])
		}
		i += n
	}
	if !caps[capNonUniform] || !caps[capShuffle] {
		t.Fatalf("capabilities %v declared; want both GroupNonUniform(%d) and Shuffle(%d)",
			caps, capNonUniform, capShuffle)
	}
	want := []uint32{1, 2, 4, 8, 16, 1, 2, 4, 8, 16} // one butterfly for the max, one for the sum
	if len(masks) != len(want) {
		t.Fatalf("got %d shuffles %v, want %d", len(masks), masks, len(want))
	}
	for i, m := range masks {
		if m != want[i] {
			t.Fatalf("shuffle %d has mask %d, want %d", i, m, want[i])
		}
	}

	// And the fallback kernel must not ask the device for a capability it does
	// not use: that is the whole point of having a fallback.
	sc, err := mainkernels.SoftmaxRows(8, 64, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err = spirv.Emit(sc)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := headerWords, words(t, b); i < len(w); {
		n, op := int(w[i]>>16), w[i]&0xFFFF
		if op == opCapability && (w[i+1] == capNonUniform || w[i+1] == capShuffle) {
			t.Fatalf("the scalar softmax declares subgroup capability %d", w[i+1])
		}
		i += n
	}
}

// TestDescriptorAnchor: a module whose first buffer access sits inside a loop
// with a runtime count reads a descriptor at its entry, ahead of every loop,
// and keeps it alive through that loop's count; one whose first access is not
// inside such a loop is emitted without it. See anchorFor: without it llvmpipe
// faults on mainkernels.MatVecSegments, which spirv-val and the validation layer
// both pass.
func TestDescriptorAnchor(t *testing.T) {
	const (
		headerWords   = 5
		opArrayLength = 68
		opIEqual      = 170
		opSelect      = 169
		opULessThan   = 176
		opLoopMerge   = 246
	)
	sh := mainkernels.MatVecShape{T: kernels.Q4_K, K: 512, Rows: 64}
	seg, err := mainkernels.MatVecSegments([]mainkernels.MatVecShape{sh, sh}, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := mainkernels.MatVec(sh)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name   string
		anchor bool
	}{{"segments", true}, {"plain", false}} {
		k := seg
		if !c.anchor {
			k = plain
		}
		b, err := spirv.Emit(k)
		if err != nil {
			t.Fatal(err)
		}
		w := words(t, b)
		lengths, loops := 0, 0
		var length, never, sel uint32
		anchored := false
		for i := headerWords; i < len(w); {
			n, op := int(w[i]>>16), w[i]&0xFFFF
			switch op {
			case opArrayLength:
				if loops > 0 {
					t.Fatalf("%s: the descriptor is read inside a loop, where it anchors nothing", c.name)
				}
				lengths++
				length = w[i+2]
			case opIEqual:
				if length != 0 && w[i+3] == length {
					never = w[i+2]
				}
			case opSelect:
				if never != 0 && w[i+3] == never {
					sel = w[i+2]
				}
			case opULessThan:
				if sel != 0 && w[i+4] == sel && loops == 0 {
					anchored = true // the first loop's count is the select
				}
			case opLoopMerge:
				loops++
			}
			i += n
		}
		if c.anchor && (lengths != 1 || !anchored) {
			t.Fatalf("%s: %d descriptor read(s), feeding the first loop's count: %v; want 1, true",
				c.name, lengths, anchored)
		}
		if !c.anchor && lengths != 0 {
			t.Fatalf("%s: %d descriptor read(s) in a module whose first access needs no anchor", c.name, lengths)
		}
	}
}

func words(t *testing.T, b []byte) []uint32 {
	t.Helper()
	if len(b)%4 != 0 || len(b) < 20 {
		t.Fatalf("module is %d bytes, not a whole SPIR-V header plus body", len(b))
	}
	w := make([]uint32, len(b)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	if w[0] != 0x07230203 {
		t.Fatalf("magic is %#x, not SPIR-V", w[0])
	}
	return w
}
