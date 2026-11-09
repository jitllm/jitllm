package cuda

import (
	"fmt"

	"github.com/jitllm/jitllm/backend/ffi"
)

// errPeerAlreadyEnabled is CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: a second
// enable of the same pair, which is success here.
const errPeerAlreadyEnabled CUresult = 704

func (drv *Driver) loadPeer(lib *ffi.Lib) {
	for _, n := range []string{"cuMemcpyPeer", "cuDeviceCanAccessPeer", "cuCtxEnablePeerAccess"} {
		if !lib.Has(n) {
			return
		}
	}
	drv.cuMemcpyPeer = ffi.Fn5[CUresult, CUdevptr, CUctx, CUdevptr, CUctx, uint64](lib, "cuMemcpyPeer")
	drv.cuDeviceCanAccessPeer = ffi.Fn3[CUresult, *int32, CUdevice, CUdevice](lib, "cuDeviceCanAccessPeer")
	drv.cuCtxEnablePeerAccess = ffi.Fn2[CUresult, CUctx, uint32](lib, "cuCtxEnablePeerAccess")
	if lib.Has("cuCtxDisablePeerAccess") {
		drv.cuCtxDisablePeerAccess = ffi.Fn1[CUresult, CUctx](lib, "cuCtxDisablePeerAccess")
	}
}

// PeerCopyAvailable reports whether this driver copies between contexts.
func (drv *Driver) PeerCopyAvailable() bool { return drv.cuMemcpyPeer != nil }

// CanAccessPeer reports whether d can address peer's memory directly (NVLink
// or PCIe peer-to-peer), as the driver says.
func (d *Device) CanAccessPeer(peer *Device) bool {
	if d.drv.cuDeviceCanAccessPeer == nil || d == peer {
		return false
	}
	var can int32
	if d.drv.cuDeviceCanAccessPeer(&can, d.dev, peer.dev) != 0 {
		return false
	}
	return can != 0
}

// EnablePeerAccess lets the context current on the calling thread -- d's,
// on its owner thread -- address peer's allocations. Run it on d's thread.
func (d *Device) EnablePeerAccess(peer *Device) error {
	if d.drv.cuCtxEnablePeerAccess == nil {
		return fmt.Errorf("cuda: this driver has no cuCtxEnablePeerAccess")
	}
	if r := d.drv.cuCtxEnablePeerAccess(peer.ctx, 0); r != 0 && r != errPeerAlreadyEnabled {
		return d.drv.call(r, "cuCtxEnablePeerAccess")
	}
	return nil
}

// DisablePeerAccess undoes EnablePeerAccess for the context current on the
// calling thread -- d's, on its owner thread.
func (d *Device) DisablePeerAccess(peer *Device) error {
	if d.drv.cuCtxDisablePeerAccess == nil {
		return fmt.Errorf("cuda: this driver has no cuCtxDisablePeerAccess")
	}
	return d.drv.call(d.drv.cuCtxDisablePeerAccess(peer.ctx), "cuCtxDisablePeerAccess")
}

// CopyPeer copies n bytes of src (an allocation of srcDev) at srcOff to dst
// (dstDev's) at dstOff. It returns once the copy is done with respect to the
// host (cuMemcpyPeer's synchronous behaviour). Ranges are the caller's to have
// checked.
func (drv *Driver) CopyPeer(dst *Buffer, dstDev *Device, dstOff int, src *Buffer, srcDev *Device, srcOff, n int) error {
	if drv.cuMemcpyPeer == nil {
		return fmt.Errorf("cuda: this driver has no cuMemcpyPeer")
	}
	return drv.call(drv.cuMemcpyPeer(dst.p+CUdevptr(dstOff), dstDev.ctx, src.p+CUdevptr(srcOff), srcDev.ctx, uint64(n)), "cuMemcpyPeer")
}
