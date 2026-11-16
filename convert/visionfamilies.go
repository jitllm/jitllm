package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The per-projector half of the tower converter: what gemma3 and InternVL ship
// that the shared name table and the shared transforms do not cover. Every
// transform here is exact, or states the one rounding it adds.

// floatRows decodes a 2-D tensor to f32, k values per row.
func floatRows(t *jlm.Tensor) (vals []float32, k, rows int, err error) {
	if t.NDim != 2 {
		return nil, 0, 0, fmt.Errorf("convert: %s is %d-dimensional, want a matrix", t.Name, t.NDim)
	}
	src, ok := sourceType(t.Type)
	if !ok {
		return nil, 0, 0, fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
	}
	k, rows = int(t.Dims[0]), int(t.Dims[1])
	b, err := t.Bytes()
	if err != nil {
		return nil, 0, 0, err
	}
	vals = make([]float32, k*rows)
	if err := quant.Dequant32(src, b, vals); err != nil {
		return nil, 0, 0, fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	return vals, k, rows, nil
}

// setF32Dims is setF32 with a new shape. The tower's matrices are quantized
// afterwards by quantizeTower, whose one lossy step is then the only one.
func setF32Dims(t *jlm.Tensor, vals []float32, dims ...uint64) {
	setF32(t, vals)
	t.NDim = uint8(len(dims))
	t.Dims = [4]uint64{}
	copy(t.Dims[:], dims)
}

// vecOf decodes a vector to f32.
func vecOf(t *jlm.Tensor) ([]float32, error) { return normF32(t) }

// normF32 decodes a norm vector to f32, whatever the source stored.
func normF32(t *jlm.Tensor) ([]float32, error) {
	typ, ok := jlm.SourceType(t.Type)
	if !ok {
		return nil, fmt.Errorf("convert: %s is %v, which has no dequantizer", t.Name, t.Type)
	}
	b, err := t.Bytes()
	if err != nil {
		return nil, err
	}
	n := 1
	for d := 0; d < int(t.NDim); d++ {
		n *= int(t.Dims[d])
	}
	out := make([]float32, n)
	if err := quant.Dequant32(typ, b, out); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	return out, nil
}

// setF32 replaces a tensor's bytes with v as F32.
func setF32(t *jlm.Tensor, v []float32) {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	t.Type, t.Data, t.Load = jlm.TypeF32, b, nil
}

// imgBreakRow gives a Pixtral tower its [IMG_BREAK] row when the mmproj does
// not carry one. llama.cpp's converter copies the token's embedding into the
// mmproj only from a file whose table it can see: from a transformers
// checkpoint it skips every language_model tensor first, so the row is
// absent. The container holds the text model too, so the row is the text
// table's own, read once here: the same row the reference looks up for the
// token, at the precision the text file keeps it.
func imgBreakRow(s *jlm.Source) error {
	for i := range s.Tensors {
		if s.Tensors[i].Role == jlm.RoleVImgBreak {
			return nil
		}
	}
	id := -1
	if s.Vocab != nil {
		for i, tok := range s.Vocab.Tokens {
			if tok == "[IMG_BREAK]" {
				id = i
				break
			}
		}
	}
	if id < 0 {
		return fmt.Errorf("convert: a pixtral tower and no [IMG_BREAK] row: the mmproj carries no " +
			"v.token_embd.img_break and the text vocabulary has no [IMG_BREAK]")
	}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleTokenEmbd {
			continue
		}
		src, ok := sourceType(t.Type)
		if !ok || t.NDim != 2 || uint64(id) >= t.Dims[1] {
			return fmt.Errorf("convert: %s cannot give token %d's row", t.Name, id)
		}
		k := int(t.Dims[0])
		be, bb := int(src.BlockElems()), int(src.BlockBytes())
		if k%be != 0 {
			return fmt.Errorf("convert: %s rows are %d wide, not whole %v blocks", t.Name, k, src)
		}
		rb := k / be * bb
		b, err := t.Bytes()
		if err != nil {
			return err
		}
		row := make([]float32, k)
		if err := quant.Dequant32(src, b[id*rb:(id+1)*rb], row); err != nil {
			return fmt.Errorf("convert: %s row %d: %w", t.Name, id, err)
		}
		e := jlm.Tensor{Role: jlm.RoleVImgBreak, Block: jlm.DenseBlock, Index: -1,
			Name: "v.token_embd.img_break"}
		setF32Dims(&e, row, uint64(k))
		s.Tensors = append(s.Tensors, e)
		return nil
	}
	return fmt.Errorf("convert: a pixtral tower and no text token table to take [IMG_BREAK]'s row from")
}

// padFFN widens a tower's MLP to a whole quantization block when its width is
// not one: SigLIP-so400m's is 4304, 16 short of a block, and Qwen2.5-VL's
// 3420 (the published mmproj keeps that contraction F16 for the reason). The
// expansions -- the up projection, and a gated MLP's gate -- gain
// zero rows and a zero bias, so their extra outputs are exactly zero, every
// activation the tower uses maps zero to zero, and the contraction's zero
// columns then add exactly nothing. The result is the same function with a
// width every packed kernel, host and device, can read.
func padFFN(s *jlm.Source) error {
	v := s.Vision
	if v.NFFN%q8Elems == 0 {
		return nil
	}
	wide := (v.NFFN + q8Elems - 1) / q8Elems * q8Elems
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch t.Role {
		case jlm.RoleVFC1, jlm.RoleVFC2, jlm.RoleVFFNGate:
			vals, k, rows, err := floatRows(t)
			if err != nil {
				return err
			}
			switch {
			case uint32(rows) == v.NFFN && uint32(k) == v.NEmbd:
				// The expansion: zero rows appended.
				out := make([]float32, int(wide)*k)
				copy(out, vals)
				setF32Dims(t, out, uint64(k), uint64(wide))
			case uint32(k) == v.NFFN && uint32(rows) == v.NEmbd:
				// The contraction: zero columns appended to every row.
				out := make([]float32, int(wide)*rows)
				for r := 0; r < rows; r++ {
					copy(out[r*int(wide):], vals[r*k:(r+1)*k])
				}
				setF32Dims(t, out, uint64(wide), uint64(rows))
			default:
				return fmt.Errorf("convert: %s is %dx%d, neither NEmbd %d -> NFFN %d nor back",
					t.Name, k, rows, v.NEmbd, v.NFFN)
			}
		case jlm.RoleVFC1Bias, jlm.RoleVFC2Bias, jlm.RoleVFFNGateBias:
			bv, err := vecOf(t)
			if err != nil {
				return err
			}
			if uint32(len(bv)) == v.NFFN {
				out := make([]float32, wide)
				copy(out, bv)
				setF32Dims(t, out, uint64(wide))
			}
		}
	}
	v.NFFN = wide
	return nil
}

// tokenRow is the text table's row of token id, dequantized as the text
// model reads it.
func tokenRow(s *jlm.Source, id int) ([]float32, error) {
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Role != jlm.RoleTokenEmbd {
			continue
		}
		src, ok := sourceType(t.Type)
		if !ok || t.NDim != 2 || uint64(id) >= t.Dims[1] {
			return nil, fmt.Errorf("convert: %s cannot give token %d's row", t.Name, id)
		}
		k := int(t.Dims[0])
		be, bb := int(src.BlockElems()), int(src.BlockBytes())
		if k%be != 0 {
			return nil, fmt.Errorf("convert: %s rows are %d wide, not whole %v blocks", t.Name, k, src)
		}
		rb := k / be * bb
		b, err := t.Bytes()
		if err != nil {
			return nil, err
		}
		row := make([]float32, k)
		if err := quant.Dequant32(src, b[id*rb:(id+1)*rb], row); err != nil {
			return nil, fmt.Errorf("convert: %s row %d: %w", t.Name, id, err)
		}
		return row, nil
	}
	return nil, fmt.Errorf("convert: no text token table to take token %d's row from", id)
}

// markTable writes a picture's marker rows (jlm.RoleVTileMarks): the text
// table's row of each named token in order, a zero row where a name is
// empty.
func markTable(s *jlm.Source, names []string) error {
	ids := map[string]int{}
	if s.Vocab != nil {
		for i, tok := range s.Vocab.Tokens {
			ids[tok] = i
		}
	}
	rows := make([]int32, len(names))
	for at, name := range names {
		rows[at] = -1
		if name == "" {
			continue
		}
		id, ok := ids[name]
		if !ok && name == "\n" {
			// A newline is a byte token where the vocabulary stores bytes:
			// byte-level BPE's or sentencepiece's.
			for _, alt := range []string{"\u010a", "<0x0A>"} {
				if id, ok = ids[alt]; ok {
					break
				}
			}
		}
		if !ok {
			return fmt.Errorf("convert: the vocabulary has no %q, a picture's marker", name)
		}
		rows[at] = int32(id)
	}
	return markRows(s, rows)
}

// markRows writes a picture's marker rows (jlm.RoleVTileMarks): the text
// table's row of each token in order, a zero row for -1.
func markRows(s *jlm.Source, ids []int32) error {
	var table []float32
	k := 0
	for at, id := range ids {
		if id < 0 {
			continue
		}
		row, err := tokenRow(s, int(id))
		if err != nil {
			return err
		}
		if table == nil {
			k = len(row)
			table = make([]float32, len(ids)*k)
		}
		copy(table[at*k:(at+1)*k], row)
	}
	if table == nil {
		return fmt.Errorf("convert: a marker table of no markers")
	}
	e := jlm.Tensor{Role: jlm.RoleVTileMarks, Block: jlm.DenseBlock, Index: -1, Name: "v.token_embd.tile_marks"}
	setF32Dims(&e, table, uint64(k), uint64(len(ids)))
	s.Tensors = append(s.Tensors, e)
	return nil
}
