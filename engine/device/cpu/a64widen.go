package cpu

// The int8 dots widened to int16 products and pairwise sums: what the
// composer forms a dot of with no FEAT_DotProd (lowera64_node.go).

// SMULL8b is smull vd.8h, vn.8b, vm.8b: the low eight signed byte products.
func (a *A64) SMULL8b(vd, vn, vm VReg) {
	a.w(0x0E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMULL16b is smull2 vd.8h, vn.16b, vm.16b: the high eight.
func (a *A64) SMULL16b(vd, vn, vm VReg) {
	a.w(0x4E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SADDLP8h is saddlp vd.4s, vn.8h: adjacent int16 pairs summed into int32.
func (a *A64) SADDLP8h(vd, vn VReg) {
	a.w(0x4E602800 | uint32(vn)<<5 | uint32(vd))
}

// SADALP8h is sadalp vd.4s, vn.8h: adjacent int16 pairs summed and added into
// vd's int32 lanes.
func (a *A64) SADALP8h(vd, vn VReg) {
	a.w(0x4E606800 | uint32(vn)<<5 | uint32(vd))
}

// ADDP8h is addp vd.8h, vn.8h, vm.8h: int16 pairwise add, vn's pairs in the
// low four lanes and vm's in the high four.
func (a *A64) ADDP8h(vd, vn, vm VReg) {
	a.w(0x4E60BC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMLAL8b is smlal vd.8h, vn.8b, vm.8b: the low eight products accumulated.
func (a *A64) SMLAL8b(vd, vn, vm VReg) {
	a.w(0x0E208000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMLAL16b is smlal2 vd.8h, vn.16b, vm.16b: the high eight accumulated.
func (a *A64) SMLAL16b(vd, vn, vm VReg) {
	a.w(0x4E208000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// ADDP4s is addp vd.4s, vn.4s, vm.4s: integer pairwise add, vn's pairs in the
// low two lanes and vm's in the high two.
func (a *A64) ADDP4s(vd, vn, vm VReg) {
	a.w(0x4EA0BC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}
