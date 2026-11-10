package gpu

import "testing"

// TestAFormSharesBuffersByTheirLives is the gate on partForm.share: a
// buffer is laid on another's device buffer only where the other's last
// code ran before its first, and a constant, a buffer first read and a
// paged form's selection and table keep their own.
func TestAFormSharesBuffersByTheirLives(t *testing.T) {
	w := func(i int) partArg { return partArg{kind: partOwn, i: i, writes: true} }
	r := func(i int) partArg { return partArg{kind: partOwn, i: i, read: true} }
	// Codes: 0 writes a; 1 reads a, writes b; 2 reads b, writes c; 3 reads
	// c and the constant k, reads then writes the carried s; 4 reads c,
	// writes d.
	const a, b, c, k, s, d = 0, 1, 2, 3, 4, 5
	f := &partForm{
		codes: make([]partCode, 5),
		own: []partBuf{a: {n: 400}, b: {n: 800}, c: {n: 400}, k: {n: 16, words: []uint32{1, 2, 3, 4}},
			s: {n: 64}, d: {n: 200}},
		args: [][]partArg{
			{w(a)},
			{r(a), w(b)},
			{r(b), w(c)},
			{r(c), r(k), r(s), w(s)},
			{r(c), w(d)},
		},
	}
	f.share(false)
	live := [][2]int{a: {0, 1}, b: {1, 2}, c: {2, 4}, d: {4, 4}}
	for i := range live {
		for j := range live {
			if i < j && f.slot[i] == f.slot[j] && live[i][1] >= live[j][0] && live[j][1] >= live[i][0] {
				t.Fatalf("buffers %d and %d, live over codes %v and %v, share device buffer %d",
					i, j, live[i], live[j], f.slot[i])
			}
		}
	}
	for _, i := range []int{k, s} {
		for j := range f.own {
			if j != i && f.slot[j] == f.slot[i] {
				t.Fatalf("buffer %d, a constant or carried, shares device buffer %d with %d", i, f.slot[i], j)
			}
		}
	}
	if f.slot[a] != f.slot[c] && f.slot[a] != f.slot[d] && f.slot[b] != f.slot[d] {
		t.Fatalf("no buffer took one given up before it (slots %v): nothing was shared", f.slot)
	}
	var sum, held int
	for _, p := range f.phys {
		held += p.n
	}
	for _, o := range f.own {
		sum += o.n
	}
	t.Logf("%d own buffers of %d bytes on %d device buffers of %d (slots %v)", len(f.own), sum, len(f.phys), held, f.slot)
	if held >= sum {
		t.Fatalf("the laid-out buffers hold %d bytes of %d: nothing was saved", held, sum)
	}
	for i, o := range f.own {
		if f.phys[f.slot[i]].n < o.n {
			t.Fatalf("buffer %d of %d bytes laid on a device buffer of %d", i, o.n, f.phys[f.slot[i]].n)
		}
	}

	// A paged form's selection and table, fresh and short-lived, are the
	// host's between the halves: never shared.
	p := &partForm{
		codes: make([]partCode, 3),
		own:   []partBuf{{n: 64}, {n: 64}, {n: 64}},
		args:  [][]partArg{{w(0)}, {r(0), w(1)}, {w(2)}},
		sel:   0, table: 2,
	}
	p.share(true)
	if p.slot[2] == p.slot[0] || p.slot[2] == p.slot[1] || p.slot[0] == p.slot[1] {
		t.Fatalf("a paged form's selection or table shares a device buffer: slots %v", p.slot)
	}

	// A code that lists a buffer and neither reads nor writes it does not
	// begin its life: written by code 1, read by code 2, it passes to the
	// scratch's arena as any buffer live within one call.
	u := func(i int) partArg { return partArg{kind: partOwn, i: i} }
	q := &partForm{
		codes: make([]partCode, 3),
		own:   []partBuf{{n: 64}, {n: 1 << 20}},
		args:  [][]partArg{{w(0), u(1)}, {r(0), w(1)}, {r(1)}},
	}
	q.share(false)
	if q.arenaAt[q.slot[1]] < 0 {
		t.Fatalf("a buffer bound unused before its first write is held as carried state (arena %v, slots %v)", q.arenaAt, q.slot)
	}
}
