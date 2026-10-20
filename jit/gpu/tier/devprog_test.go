package tier

import "testing"

// TestEveryDevKindHasAName holds the names to the enums: a kind, slot or
// reason added without a name would dump as "".
//
// VIOLATION SIGNATURE. Add a kind before numKinds and leave it out of its
// String's switch, and this names it.
func TestEveryDevKindHasAName(t *testing.T) {
	for k := devKind(0); k < numKinds; k++ {
		if k.String() == "" {
			t.Errorf("device-program kind %d has no name", k)
		}
	}
	for r := progReason(1); r < numProgReasons; r++ {
		if r.String() == "" {
			t.Errorf("program reason %d has no name", r)
		}
	}
	for s := devSlot(0); s < numSlots; s++ {
		if s.String() == "" {
			t.Errorf("slot %d has no name", s)
		}
	}
	if n := len(newDevRewrites()); n > 64 {
		t.Errorf("%d rewrites: rwSet holds 64", n)
	}
}

// TestCompositeKindsOnlyShrink pins the kinds that are not the program's
// primitives (compositeKind: every kind primitiveKind does not name): a
// mechanism of one family run whole, whether by an encoder sequence or by a
// generator written for that kind alone. The device program is a compiler
// of primitives, and each of these goes as the builder lowers its graph op
// into them. A kind added to the set fails here, and so does a list entry
// no longer composite.
//
// VIOLATION SIGNATURE. Add a program kind for one family's mechanism, or
// give one a generator of its own, and this names it.
func TestCompositeKindsOnlyShrink(t *testing.T) {
	allowed := map[string]bool{
		"recurrent":         true,
		"q_temp":            true,
		"split_qg":          true,
		"ple_slice":         true,
		"rope_k_row":        true,
		"stream_route":      true,
		"stream_predict":    true,
		"stream_correct":    true,
		"stream_scale":      true,
		"stream_finish":     true,
		"gauss_topk":        true,
		"stream_norm":       true,
		"stream_gate_mix":   true,
		"place_weights":     true,
		"stream_place":      true,
		"idx_norm":          true,
		"idx_rope":          true,
		"idx_assemble":      true,
		"idx_score":         true,
		"select_blocks":     true,
		"gather_kv":         true,
		"attend_gathered":   true,
		"stream_dot":        true,
		"gated_val":         true,
		"ngram_conv":        true,
		"ngram_add":         true,
		"stream_score":      true,
		"stream_weighted":   true,
		"idx_write":         true,
		"idx_attend_latent": true,
		"row_bias":          true,
		"stream_flatten":    true,
		"flat_norm":         true,
		"mix_logits":        true,
		"sinkhorn":          true,
		"stream_collect":    true,
		"row_r_m_s":         true,
		"turn":              true,
		"cache_write":       true,
		"pool_entries":      true,
		"entry_write":       true,
		"entry_select":      true,
		"gather_entries":    true,
		"grouped_mat_vec":   true,
		"clamp_at":          true,
		"router_logits":     true,
		"router_bias":       true,
		"group_mask":        true,
		"route_weights":     true,
		"expert_bias":       true,
		"gate_logit":        true,
		"gated_add":         true,
		"group_by_expert":   true,
		"rope_tail":         true,
		"latent_write":      true,
		"latent_attend":     true,
		"expert_matvec_e":   true,
		"conv1d":            true,
		"scan":              true,
		"gated_norm":        true,
		"conv_bias":         true,
		"ssm_norm":          true,
		"matvec_res":        true,
		"matvec_gate":       true,
		"qkv":               true,
		"rope_qkv":          true,
		"gather_heads":      true,
	}
	if len(allowed) > 67 {
		t.Fatalf("%d composite kinds allowed: the list only shrinks from 67", len(allowed))
	}
	for k := devKind(0); k < numKinds; k++ {
		name := k.String()
		switch {
		case compositeKind(k) && !allowed[name]:
			t.Errorf("%s is composite and not on the list: lower it to primitives", name)
		case !compositeKind(k) && allowed[name]:
			t.Errorf("%s is on the list and no longer composite: remove it from the list", name)
		}
	}
}
