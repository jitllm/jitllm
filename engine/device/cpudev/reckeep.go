package cpudev

// recKeep is a session's copies of its recurrent state on the CPU device, as
// a card keeps its own (nn.RecRewinder, nn.RecRowKeeper): each block's state
// as it was at the mark, copied as the block is first stepped after it, and
// with rows asked each block's state after each of a call's first rows.
type recKeep struct {
	marked      bool
	dirty       []bool
	conv, state [][]float32
	rows        int
	kept        [][]bool
	rconv       [][][]float32
	rstate      [][][]float32
}

// stepping is the lanes about to step block li's state in view v: its state
// copied first where a mark is set and the block has not stepped since.
func (k *recKeep) stepping(v *View, li int) {
	if !k.marked || v == nil || li < 0 || li >= v.Blocks() {
		return
	}
	if len(k.dirty) < v.Blocks() {
		k.dirty = append(k.dirty, make([]bool, v.Blocks()-len(k.dirty))...)
		k.conv = append(k.conv, make([][]float32, v.Blocks()-len(k.conv))...)
		k.state = append(k.state, make([][]float32, v.Blocks()-len(k.state))...)
	}
	if k.dirty[li] {
		return
	}
	k.dirty[li] = true
	k.conv[li] = append(k.conv[li][:0], v.Conv(li)...)
	k.state[li] = append(k.state[li][:0], v.State(li)...)
}

// stepped is the lanes having stepped block li's state in view v by row
// row of the call: copied where the call keeps that row.
func (k *recKeep) stepped(v *View, li, row int) {
	if row < 0 || row >= k.rows || v == nil || li < 0 || li >= v.Blocks() {
		return
	}
	for len(k.kept) <= row {
		k.kept, k.rconv, k.rstate = append(k.kept, nil), append(k.rconv, nil), append(k.rstate, nil)
	}
	if len(k.kept[row]) < v.Blocks() {
		k.kept[row] = append(k.kept[row], make([]bool, v.Blocks()-len(k.kept[row]))...)
		k.rconv[row] = append(k.rconv[row], make([][]float32, v.Blocks()-len(k.rconv[row]))...)
		k.rstate[row] = append(k.rstate[row], make([][]float32, v.Blocks()-len(k.rstate[row]))...)
	}
	k.kept[row][li] = true
	k.rconv[row][li] = append(k.rconv[row][li][:0], v.Conv(li)...)
	k.rstate[row][li] = append(k.rstate[row][li][:0], v.State(li)...)
}

// whole says the lanes may run a linear block's rows as one form: not while
// a mark or a call's rows are kept, which a form over every row at once
// would step past.
func (k *recKeep) whole() bool { return k == nil || !k.marked && k.rows == 0 }

// RecMark records where the session's recurrent state on the CPU device is
// (nn.RecRewinder): each block's is copied as it is first stepped after.
func (s *SessionLanes) RecMark() {
	k := &s.keep
	k.marked = true
	clear(k.dirty)
}

// RecRewind returns every block the CPU device stepped since the mark to its
// state there (nn.RecRewinder).
func (s *SessionLanes) RecRewind() bool {
	k := &s.keep
	if !k.marked {
		return false
	}
	v := s.of.History().View()
	for li, d := range k.dirty {
		if d {
			copy(v.Conv(li), k.conv[li])
			copy(v.State(li), k.state[li])
		}
	}
	clear(k.dirty)
	return true
}

// RecKeepRows has the session's next calls keep each block's state after
// each of their first rows rows (nn.RecRowKeeper); zero keeps none.
func (s *SessionLanes) RecKeepRows(rows int) {
	k := &s.keep
	k.rows = rows
	for r := range k.kept {
		clear(k.kept[r])
	}
}

// RecRestoreRow returns every block kept after row of the last call to its
// state there (nn.RecRowKeeper).
func (s *SessionLanes) RecRestoreRow(row int) bool {
	k := &s.keep
	if row < 0 || row >= k.rows || row >= len(k.kept) {
		return false
	}
	v := s.of.History().View()
	for li, ok := range k.kept[row] {
		if ok {
			copy(v.Conv(li), k.rconv[row][li])
			copy(v.State(li), k.rstate[row][li])
		}
	}
	return true
}
