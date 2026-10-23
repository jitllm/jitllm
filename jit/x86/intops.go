package x86

func (a *Buf) XORQ(dst, src Reg) {
	a.rex(true, src, noIndex, dst)
	a.u8(0x31)
	a.modrmReg(src, dst)
}

func (a *Buf) DIVQ(src Reg) {
	a.rex(true, Y0, noIndex, src)
	a.u8(0xF7)
	a.modrmReg(Reg(6), src)
}

func (a *Buf) MOVLLoad(dst Reg, m Mem) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, dst, idx, m.Base)
	a.u8(0x8B)
	a.modrmMem(dst, m)
}

func (a *Buf) MOVLStore(m Mem, src Reg) {
	idx := noIndex
	if m.hasIndex() {
		idx = m.Index
	}
	a.rex(false, src, idx, m.Base)
	a.u8(0x89)
	a.modrmMem(src, m)
}
