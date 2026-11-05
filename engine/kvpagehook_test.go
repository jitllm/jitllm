package engine

import "testing"

// withKVPage pins the instance's KV page size: p positions a page. It lives
// in a test file, so a release binary has no way to reach it.
func withKVPage(p int) Option { return func(l *loadOpts) { l.opt.kvPage = p } }

// TestKVPageIsPerInstance: two instances of one model, one with its KV page
// pinned, made in both orders. Each pages at its own size.
func TestKVPageIsPerInstance(t *testing.T) {
	pages := func(s *Instance) int {
		for li := range s.kv.layers {
			if p := s.kv.layers[li].p; p > 0 {
				return p
			}
		}
		t.Fatal("the fixture has no paged layer")
		return 0
	}
	m, openOpts := hybridModelOpt(t, hyOpt{})
	defer m.Close()
	probe := testInstance(t, m, 64, openOpts...)
	want := pages(probe)
	probe.Close()
	if want == kvKeyTile {
		t.Fatalf("the default page is already %d positions; the pin would not show", kvKeyTile)
	}
	pinned := append(append([]Option(nil), openOpts...), withKVPage(kvKeyTile))
	for _, pinnedFirst := range []bool{true, false} {
		var sa, sb *Instance
		if pinnedFirst {
			sa, sb = testInstance(t, m, 64, append(append([]Option(nil), openOpts...), pinned...)...), testInstance(t, m, 64, openOpts...)
		} else {
			sb, sa = testInstance(t, m, 64, openOpts...), testInstance(t, m, 64, append(append([]Option(nil), openOpts...), pinned...)...)
		}
		if p := pages(sa); p != kvKeyTile {
			t.Errorf("pinned first=%v: the pinned instance pages at %d, want %d", pinnedFirst, p, kvKeyTile)
		}
		if p := pages(sb); p != want {
			t.Errorf("pinned first=%v: the other instance pages at %d, want its default %d", pinnedFirst, p, want)
		}
		sa.Close()
		sb.Close()
	}
}
