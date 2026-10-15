package model

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// d1-omni's audio tower (jlm.Audio, the reference's audio.py): a clip of 16
// kHz samples in, one prefix row of the text model's width per 80 ms out.
// Every product is the generated matmul, every elementwise step a generated
// kernel, on the Embedder's JIT and pool; what is Go is the geometry (the
// frame and convolution strides) and the front end's constant tables, built
// once at load: the windowed DFT basis and the Slaney filterbank.
//
//	front end   preemphasis (an axpy), the centred 512-sample frames every
//	            160 (one strided copy), |DFT|^2 as two matmuls against the
//	            Hann-windowed cos and sin bases, the filterbank's matmul, ln
//	            of it plus 2^-24, each feature normalised over the clip's
//	            frames (mean, and std with n-1, plus 1e-5)
//	subsampling a 3x3 stride-2 convolution of one channel (im2col by nine
//	            strided copies, then a matmul), ReLU; twice a depthwise 3x3
//	            stride-2 convolution, a pointwise one and ReLU; each layer's
//	            input cut at the frames still valid; then the linear from
//	            16 frequencies x 256 channels (channel-last, as the converter
//	            reordered it) into 512
//	conformer   per block: x += FF(LN x)/2; relative-position attention
//	            (scores (q+u).k + (q+v).P[t-1-i+j], so row i's relative term
//	            is the score kernel over the projected position table read
//	            from row t-1-i); the convolution module (LN, the GLU, a
//	            depthwise convolution of ConvKernel taps, the folded
//	            BatchNorm, SiLU, pointwise); x += FF(LN x)/2; LN
//	adapter     LN, linear, erf GELU, linear; then x += up(GELU(down(LN x)))
type audioTower struct {
	cfg  jlm.Audio
	base int // the tower block of audio block 0, past the vision blocks

	// The front end's tables, rows of F32: the windowed DFT bases (NFFT/2+1
	// rows of NFFT) and the filterbank (MelBins rows of NFFT/2+1).
	dftCos, dftSin, mel tensor

	// The subsampling: conv0 a matrix (channels x 9 taps), the depthwise
	// layers' taps tap-major, the pointwise layers matrices.
	conv0, pw1, pw2 tensor
	dw1, dw2        []float32
	bias            [5][]float32 // per layer 0, 2, 3, 5, 6
	subOut          tensor
	subOutB         []float32

	blocks []audBlock

	adNorm, adNormB, ad1B, ad2B []float32
	ad1, ad2                    tensor
	resNorm, resNormB, resDB    []float32
	resUB                       []float32
	resDown, resUp              tensor

	// pages is each block's matrices and their spans (encPage).
	pages []encPage
	// pe is the relative positions' sinusoids a device's blocks read
	// (audioPE), built on the first offer.
	pe []float32
}

type audBlock struct {
	ff1N, ff1NB, ff1UB, ff1DB []float32
	ff1Up, ff1Down            tensor
	attN, attNB               []float32
	q, k, v, o, pos           tensor
	qB, kB, vB, oB, u, vbias  []float32
	convN, convNB             []float32
	gluV, gluG, pw2           tensor
	gluVB, gluGB, pw2B        []float32
	// bnB is the BatchNorm's shift with the depthwise bias folded in at
	// conversion (jlm.RoleABNShift): the module's one affine.
	dw, bnS, bnB              []float32
	ff2N, ff2NB, ff2UB, ff2DB []float32
	ff2Up, ff2Down            tensor
	outN, outNB               []float32
}

func (b *audBlock) weights() []*tensor {
	return []*tensor{&b.ff1Up, &b.ff1Down, &b.q, &b.k, &b.v, &b.o, &b.pos, &b.gluV, &b.gluG, &b.pw2,
		&b.ff2Up, &b.ff2Down}
}

// f32Tensor is vals as an F32 matrix of rows x k, a table the engine built.
func f32Tensor(vals []float32, rows, k int) tensor {
	b := unsafe.Slice((*byte)(unsafe.Pointer(&vals[0])), 4*len(vals))
	return tensor{typ: quant.F32, data: b, rows: rows, k: k}
}

// containerMatrix is a 2-D weight of the container by its identity, checked
// against the shape a graph reads it at (rows 0: any).
func containerMatrix(c *jlm.File, role jlm.Role, block, index int32, rows, k int) (tensor, error) {
	e, ok := c.Find(role, block, index)
	if !ok {
		return tensor{}, fmt.Errorf("model: missing %v (block %d)", role, block)
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok || !quant.Dequantable(typ) || e.NDim != 2 {
		return tensor{}, fmt.Errorf("model: %v (block %d) is %v in %d dimensions", role, block, e.Type, e.NDim)
	}
	qs, _, _ := c.Span(e)
	t := tensor{e: e, typ: typ, data: qs, rows: int(e.Dims[1]), k: int(e.Dims[0])}
	if jlm.Packed(e.Type) {
		t.packed = &nn.Packed{}
	}
	if (rows > 0 && t.rows != rows) || t.k != k {
		return tensor{}, fmt.Errorf("model: %v (block %d) is %dx%d, want %dx%d", role, block, t.rows, t.k, rows, k)
	}
	return t, nil
}

// containerVec is a vector of the container by its identity, dequantized,
// of want values.
func containerVec(c *jlm.File, role jlm.Role, block, index int32, want int) ([]float32, error) {
	e, ok := c.Find(role, block, index)
	if !ok {
		return nil, fmt.Errorf("model: missing %v (block %d)", role, block)
	}
	typ, ok := jlm.SourceType(e.Type)
	if !ok {
		return nil, fmt.Errorf("model: %v is %v, which has no dequantizer", role, e.Type)
	}
	n := 1
	for i := 0; i < int(e.NDim); i++ {
		n *= int(e.Dims[i])
	}
	if n != want {
		return nil, fmt.Errorf("model: %v (block %d) has %d values, want %d", role, block, n, want)
	}
	qs, _, _ := c.Span(e)
	out := make([]float32, n)
	if err := quant.Dequant32(typ, qs, out); err != nil {
		return nil, fmt.Errorf("model: %v: %w", role, err)
	}
	return out, nil
}

// buildAudio loads the container's audio tower, nil where it has none.
func buildAudio(c *jlm.File, m *Model) (*audioTower, error) {
	v := c.Vision()
	if v == nil || v.Audio == nil {
		return nil, nil
	}
	a := &audioTower{cfg: *v.Audio, base: int(v.NLayer)}
	cf := &a.cfg
	if cf.Projector != jlm.ProjD1OmniA || cf.NFFT != 512 || cf.SampleRate != 16000 || cf.Hop == 0 ||
		cf.Window == 0 || cf.Window > cf.NFFT || cf.SubChannels == 0 || cf.ConvKernel%2 != 1 {
		return nil, fmt.Errorf("model: audio tower %v at %d Hz, a %d-point transform of %d-sample windows "+
			"every %d, %d channels, %d taps: not d1-omni's", cf.Projector, cf.SampleRate, cf.NFFT, cf.Window,
			cf.Hop, cf.SubChannels, cf.ConvKernel)
	}
	if int(cf.ProjDim) != m.Cfg.NEmbd {
		return nil, fmt.Errorf("model: the audio tower projects to %d and this model wants %d", cf.ProjDim, m.Cfg.NEmbd)
	}
	d, ch, ff := int(cf.NEmbd), int(cf.SubChannels), int(cf.NFFN)
	var err error
	mat := func(dst *tensor, role jlm.Role, block, index int32, rows, k int) {
		if err == nil {
			*dst, err = containerMatrix(c, role, block, index, rows, k)
		}
	}
	vec := func(dst *[]float32, role jlm.Role, block, index int32, n int) {
		if err == nil {
			*dst, err = containerVec(c, role, block, index, n)
		}
	}
	dense := int32(jlm.DenseBlock)
	// The subsampling's convolutions: F32 tables read once (Role.Expanded).
	var w0, w3, w6 []float32
	vec(&w0, jlm.RoleAConv, dense, 0, ch*9)
	vec(&a.dw1, jlm.RoleAConv, dense, 2, ch*9)
	vec(&w3, jlm.RoleAConv, dense, 3, ch*ch)
	vec(&a.dw2, jlm.RoleAConv, dense, 5, ch*9)
	vec(&w6, jlm.RoleAConv, dense, 6, ch*ch)
	for i, l := range [...]int{0, 2, 3, 5, 6} {
		vec(&a.bias[i], jlm.RoleAConvBias, dense, int32(l), ch)
	}
	if err != nil {
		return nil, err
	}
	a.conv0, a.pw1, a.pw2 = f32Tensor(w0, ch, 9), f32Tensor(w3, ch, ch), f32Tensor(w6, ch, ch)
	fr := int(cf.MelBins)
	for range 3 {
		fr = (fr-1)/2 + 1
	}
	mat(&a.subOut, jlm.RoleASubOut, dense, -1, d, fr*ch)
	vec(&a.subOutB, jlm.RoleASubOutB, dense, -1, d)
	out := int(cf.ProjDim)
	vec(&a.adNorm, jlm.RoleAAdNorm, dense, -1, d)
	vec(&a.adNormB, jlm.RoleAAdNormB, dense, -1, d)
	mat(&a.ad1, jlm.RoleAAd1, dense, -1, out, d)
	vec(&a.ad1B, jlm.RoleAAd1B, dense, -1, out)
	mat(&a.ad2, jlm.RoleAAd2, dense, -1, out, out)
	vec(&a.ad2B, jlm.RoleAAd2B, dense, -1, out)
	rw := int(cf.ResWidth)
	vec(&a.resNorm, jlm.RoleAResNorm, dense, -1, out)
	vec(&a.resNormB, jlm.RoleAResNrmB, dense, -1, out)
	mat(&a.resDown, jlm.RoleAResDown, dense, -1, rw, out)
	vec(&a.resDB, jlm.RoleAResDnB, dense, -1, rw)
	mat(&a.resUp, jlm.RoleAResUp, dense, -1, out, rw)
	vec(&a.resUB, jlm.RoleAResUpB, dense, -1, out)
	a.blocks = make([]audBlock, cf.NLayer)
	kt := int(cf.ConvKernel)
	for j := range a.blocks {
		b, bi := &a.blocks[j], int32(a.base+j)
		vec(&b.ff1N, jlm.RoleAFF1Norm, bi, -1, d)
		vec(&b.ff1NB, jlm.RoleAFF1NormB, bi, -1, d)
		mat(&b.ff1Up, jlm.RoleAFF1Up, bi, -1, ff, d)
		vec(&b.ff1UB, jlm.RoleAFF1UpB, bi, -1, ff)
		mat(&b.ff1Down, jlm.RoleAFF1Down, bi, -1, d, ff)
		vec(&b.ff1DB, jlm.RoleAFF1DownB, bi, -1, d)
		vec(&b.attN, jlm.RoleAAttnNorm, bi, -1, d)
		vec(&b.attNB, jlm.RoleAAttnNormB, bi, -1, d)
		for _, p := range []struct {
			w     *tensor
			b     *[]float32
			r, rb jlm.Role
		}{{&b.q, &b.qB, jlm.RoleAQ, jlm.RoleAQB}, {&b.k, &b.kB, jlm.RoleAK, jlm.RoleAKB},
			{&b.v, &b.vB, jlm.RoleAV, jlm.RoleAVB}, {&b.o, &b.oB, jlm.RoleAOut, jlm.RoleAOutB}} {
			mat(p.w, p.r, bi, -1, d, d)
			vec(p.b, p.rb, bi, -1, d)
		}
		mat(&b.pos, jlm.RoleAPos, bi, -1, d, d)
		vec(&b.u, jlm.RoleAPosU, bi, -1, d)
		vec(&b.vbias, jlm.RoleAPosV, bi, -1, d)
		vec(&b.convN, jlm.RoleAConvNorm, bi, -1, d)
		vec(&b.convNB, jlm.RoleAConvNormB, bi, -1, d)
		mat(&b.gluV, jlm.RoleAGLUVal, bi, -1, d, d)
		vec(&b.gluVB, jlm.RoleAGLUValB, bi, -1, d)
		mat(&b.gluG, jlm.RoleAGLUGate, bi, -1, d, d)
		vec(&b.gluGB, jlm.RoleAGLUGateB, bi, -1, d)
		vec(&b.dw, jlm.RoleADW, bi, -1, kt*d)
		vec(&b.bnS, jlm.RoleABNScale, bi, -1, d)
		vec(&b.bnB, jlm.RoleABNShift, bi, -1, d)
		mat(&b.pw2, jlm.RoleAPW2, bi, -1, d, d)
		vec(&b.pw2B, jlm.RoleAPW2B, bi, -1, d)
		vec(&b.ff2N, jlm.RoleAFF2Norm, bi, -1, d)
		vec(&b.ff2NB, jlm.RoleAFF2NormB, bi, -1, d)
		mat(&b.ff2Up, jlm.RoleAFF2Up, bi, -1, ff, d)
		vec(&b.ff2UB, jlm.RoleAFF2UpB, bi, -1, ff)
		mat(&b.ff2Down, jlm.RoleAFF2Down, bi, -1, d, ff)
		vec(&b.ff2DB, jlm.RoleAFF2DownB, bi, -1, d)
		vec(&b.outN, jlm.RoleAOutNorm, bi, -1, d)
		vec(&b.outNB, jlm.RoleAOutNormB, bi, -1, d)
	}
	if err != nil {
		return nil, err
	}
	a.tables()
	for _, w := range []*tensor{&a.subOut, &a.ad1, &a.ad2, &a.resDown, &a.resUp} {
		bindOne(c, w)
	}
	a.pages = make([]encPage, len(a.blocks))
	for j := range a.blocks {
		ws := a.blocks[j].weights()
		var rs []jlm.Range
		for _, w := range ws {
			rs = appendSpans(rs, c, w.e)
		}
		a.pages[j] = encPage{ws: ws, rs: rs}
	}
	return a, nil
}

// tables builds the front end's constant tables: the DFT bases with the
// window folded in -- torch.hann_window(Window, periodic=False), centred in
// the NFFT frame as torch.stft pads it -- and librosa's Slaney filterbank
// (audio.py's slaney_filterbank: computed in f64, stored f32).
func (a *audioTower) tables() {
	n, win, bins, nm := int(a.cfg.NFFT), int(a.cfg.Window), int(a.cfg.NFFT)/2+1, int(a.cfg.MelBins)
	w := make([]float64, n)
	left := (n - win) / 2
	for i := 0; i < win; i++ {
		w[left+i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(win-1))
	}
	cs, sn := make([]float32, bins*n), make([]float32, bins*n)
	for k := 0; k < bins; k++ {
		for t := 0; t < n; t++ {
			ang := 2 * math.Pi * float64((k*t)%n) / float64(n)
			cs[k*n+t] = float32(w[t] * math.Cos(ang))
			sn[k*n+t] = float32(w[t] * math.Sin(ang))
		}
	}
	a.dftCos, a.dftSin = f32Tensor(cs, bins, n), f32Tensor(sn, bins, n)
	a.mel = f32Tensor(slaneyFilterbank(int(a.cfg.SampleRate), n, nm), nm, bins)
}

// slaneyFilterbank is librosa.filters.mel(sr, n_fft, n_mels, norm="slaney")
// as audio.py computes it: n_mels rows of n_fft/2+1.
func slaneyFilterbank(sr, nfft, nmels int) []float32 {
	const fsp, minLogHz = 200.0 / 3, 1000.0
	logstep := math.Log(6.4) / 27
	minLogMel := minLogHz / fsp
	hzToMel := func(f float64) float64 {
		if f >= minLogHz {
			return minLogMel + math.Log(f/minLogHz)/logstep
		}
		return f / fsp
	}
	melToHz := func(m float64) float64 {
		if m >= minLogMel {
			return minLogHz * math.Exp(logstep*(m-minLogMel))
		}
		return fsp * m
	}
	bins := nfft/2 + 1
	lo, hi := hzToMel(0), hzToMel(float64(sr)/2)
	melF := make([]float64, nmels+2)
	for i := range melF {
		// np.linspace's formula: lo + i*step, the last point exact.
		melF[i] = melToHz(lo + float64(i)*(hi-lo)/float64(nmels+1))
	}
	melF[nmels+1] = melToHz(hi)
	freq := make([]float64, bins)
	for i := range freq {
		freq[i] = float64(i) * float64(sr) / float64(nfft)
	}
	out := make([]float32, nmels*bins)
	for i := 0; i < nmels; i++ {
		fd0, fd1 := melF[i+1]-melF[i], melF[i+2]-melF[i+1]
		enorm := 2 / (melF[i+2] - melF[i])
		for j, f := range freq {
			lower := -(melF[i] - f) / fd0
			upper := (melF[i+2] - f) / fd1
			v := math.Max(0, math.Min(lower, upper))
			out[i*bins+j] = float32(v) * float32(enorm)
		}
	}
	return out
}

// audioPageIn makes audio block j's matrices readable and bound.
func (m *Model) audioPageIn(j int) error {
	c := m.container
	a := m.audio
	if c == nil || j >= len(a.pages) {
		return nil
	}
	li := int(c.H.NBlocks) + a.base + j
	pg := &a.pages[j]
	if err := c.EnsureRanges(li, pg.rs); err != nil {
		return err
	}
	m.bind.Lock()
	defer m.bind.Unlock()
	for _, w := range pg.ws {
		bindOne(c, w)
	}
	return nil
}

// audioRun is an Embedder's audio scratch, grown to the longest clip seen
// and kept, so a warm clip of a size seen before allocates nothing.
type audioRun struct {
	cap                                    int
	sig, frames, re, im, mel, melT, ones   []float32
	pre, neg                               []float32
	dot                                    [1]float32
	pad0, col, x0, pad1, x1, x2, pad2, x3  []float32
	x4, x5, x6                             []float32
	x, h, q, k, v, qu, qv, p, t, ff, g, cv []float32
	att, bd, acc                           []float32
	tapRows                                []float32
	devOut                                 []float32
	posTab                                 []float32
	attn                                   *nn.AttnSet
	attStride                              int
	hd                                     int
	scale                                  float32
	rows                                   int // the attention's rows this call
}

func growA(b *[]float32, n int) []float32 {
	if cap(*b) < n {
		*b = make([]float32, n)
	}
	*b = (*b)[:n]
	return *b
}

// clipFrames is the front end's frame count for n samples (torch.stft,
// centred) and how many of them are the clip's (NeMo's frames).
func (a *audioTower) clipFrames(n int) (all, valid int) {
	hop := int(a.cfg.Hop)
	return n/hop + 1, n / hop
}

// audioPrefixRows is how many prefix rows a clip of n samples encodes to.
func (a *audioTower) audioPrefixRows(n int) int {
	_, l := a.clipFrames(n)
	for range 3 {
		l = (l+2-3)/2 + 1
	}
	return l
}

// encodeAudio runs the audio tower over samples (16 kHz mono, cut and padded
// as audio.waveform does) and returns its prefix rows, rows of ProjDim, in
// the run's scratch: good until the next clip.
func (e *Embedder) encodeAudio(samples []float32) ([]float32, error) {
	a := e.m.audio
	if a == nil {
		return nil, fmt.Errorf("model: this container carries no audio encoder; convert the model with its mmproj")
	}
	r := &e.aud
	cf := &a.cfg
	n := len(samples)
	nfft, hop, nm := int(cf.NFFT), int(cf.Hop), int(cf.MelBins)
	bins := nfft/2 + 1
	all, valid := a.clipFrames(n)

	// ---- the front end ----
	// Preemphasis into the centred, zero-padded signal.
	sig := growA(&r.sig, n+nfft)
	clear(sig)
	y := sig[nfft/2 : nfft/2+n]
	copy(y, samples)
	// y[i] -= 0.97 x[i-1]: a scaled copy added (an axpy's alpha other than
	// one would be a heap escape a clip).
	pre := growA(&r.pre, n)
	copy(pre, samples)
	nn.Scale32JIT(pre, -0.97)
	nn.Axpy32JIT(y[1:], pre[:n-1], 1)
	frames := growA(&r.frames, all*nfft)
	nn.Copy32JIT(frames, nfft, 1, sig, hop, 1, all, nfft)
	re, im := growA(&r.re, all*bins), growA(&r.im, all*bins)
	if err := e.mm(re, a.dftCos, frames, all); err != nil {
		return nil, err
	}
	if err := e.mm(im, a.dftSin, frames, all); err != nil {
		return nil, err
	}
	e.actMul(re, re, nn.ActIdentity)
	e.actMul(im, im, nn.ActIdentity)
	e.axpyRows(re, im)
	mel := growA(&r.mel, all*nm)
	if err := e.mm(mel, a.mel, re, all); err != nil {
		return nil, err
	}
	floor := growA(&r.ones, max(nm, valid))
	for i := range floor[:nm] {
		floor[i] = 1.0 / (1 << 24)
	}
	e.addBiasRows(mel, floor[:nm], all)
	nn.Ln32JIT(mel)
	// Each feature over the clip's frames: transposed, normalised, back.
	melT := growA(&r.melT, nm*valid)
	nn.Copy32JIT(melT, valid, 1, mel, 1, nm, nm, valid)
	ones := floor[:valid]
	for i := range ones {
		ones[i] = 1
	}
	for f := 0; f < nm; f++ {
		row := melT[f*valid : (f+1)*valid]
		s, err := e.dotF32(row, ones)
		if err != nil {
			return nil, err
		}
		neg := growA(&r.neg, valid)
		copy(neg, ones)
		nn.Scale32JIT(neg, -s/float32(valid))
		nn.Axpy32JIT(row, neg, 1)
		ss, err := e.dotF32(row, row)
		if err != nil {
			return nil, err
		}
		std := math.Sqrt(float64(ss) / float64(valid-1))
		if math.IsNaN(std) {
			std = 0
		}
		nn.Scale32JIT(row, float32(1/(std+1e-5)))
	}
	clear(mel)
	nn.Copy32JIT(mel, 1, nm, melT, valid, 1, nm, valid)
	if e.audioTap != nil {
		e.audioTap("mel", mel[:valid*nm])
	}

	// ---- the subsampling ----
	ch := int(cf.SubChannels)
	l1 := (valid+2-3)/2 + 1
	t1, f1 := (all+2-3)/2+1, (nm+2-3)/2+1
	wp := nm + 2
	pad0 := growA(&r.pad0, (all+2)*wp)
	clear(pad0)
	nn.Copy32JIT(pad0[wp+1:], wp, 1, mel, nm, 1, all, nm)
	col := growA(&r.col, t1*f1*9)
	for dy := 0; dy < 3; dy++ {
		for dx := 0; dx < 3; dx++ {
			nn.Copy32JIT(col[dy*3+dx:], f1*9, 9, pad0[dy*wp+dx:], 2*wp, 2, t1, f1)
		}
	}
	x0 := growA(&r.x0, t1*f1*ch)
	if err := e.mm(x0, a.conv0, col, t1*f1); err != nil {
		return nil, err
	}
	e.addBiasRows(x0, a.bias[0], t1*f1)
	e.actRows(x0, nn.ActReLU)
	clear(x0[l1*f1*ch:])
	x1, t2, f2, err := e.audioDW(&r.pad1, &r.x1, x0, t1, f1, ch, a.dw1, a.bias[1])
	if err != nil {
		return nil, err
	}
	l2 := (l1+2-3)/2 + 1
	clear(x1[l2*f2*ch:])
	x2 := growA(&r.x2, t2*f2*ch)
	if err := e.mm(x2, a.pw1, x1, t2*f2); err != nil {
		return nil, err
	}
	e.addBiasRows(x2, a.bias[2], t2*f2)
	e.actRows(x2, nn.ActReLU)
	clear(x2[l2*f2*ch:])
	x3, _, f3, err := e.audioDW(&r.pad2, &r.x3, x2, t2, f2, ch, a.dw2, a.bias[3])
	if err != nil {
		return nil, err
	}
	l3 := (l2+2-3)/2 + 1
	// Only the valid rows go on: the conformer's mask is exactly their cut.
	x4 := growA(&r.x4, l3*f3*ch)
	if err := e.mm(x4, a.pw2, x3[:l3*f3*ch], l3*f3); err != nil {
		return nil, err
	}
	e.addBiasRows(x4, a.bias[4], l3*f3)
	e.actRows(x4, nn.ActReLU)
	d := int(cf.NEmbd)
	x := growA(&r.x, l3*d)
	if err := e.mm(x, a.subOut, x4, l3); err != nil {
		return nil, err
	}
	e.addBiasRows(x, a.subOutB, l3)
	if e.audioTap != nil {
		e.audioTap("subsampled", x)
	}

	// ---- the conformer ----
	if err := e.audioBlocks(x, l3); err != nil {
		return nil, err
	}

	// ---- the adapter and the residual correction ----
	out := int(cf.ProjDim)
	h := growA(&r.h, l3*max(d, out))
	e.rowsLNW(h[:l3*d], x, a.adNorm, a.adNormB, l3, d, float64(cf.Eps))
	t := growA(&r.t, l3*out)
	if err := e.mm(t, a.ad1, h[:l3*d], l3); err != nil {
		return nil, err
	}
	e.addBiasRows(t, a.ad1B, l3)
	e.actRows(t, nn.ActGELUErf)
	x5 := growA(&r.x5, l3*out)
	if err := e.mm(x5, a.ad2, t, l3); err != nil {
		return nil, err
	}
	e.addBiasRows(x5, a.ad2B, l3)
	e.rowsLNW(h[:l3*out], x5, a.resNorm, a.resNormB, l3, out, float64(cf.Eps))
	rw := int(cf.ResWidth)
	g := growA(&r.g, l3*rw)
	if err := e.mm(g, a.resDown, h[:l3*out], l3); err != nil {
		return nil, err
	}
	e.addBiasRows(g, a.resDB, l3)
	e.actRows(g, nn.ActGELUErf)
	if err := e.mm(t, a.resUp, g, l3); err != nil {
		return nil, err
	}
	e.addBiasRows(t, a.resUB, l3)
	e.axpyRows(x5, t)
	return x5, nil
}

// audioDW is a depthwise 3x3 stride-2 convolution over x (t rows of f
// frequencies of ch channels, channel-last), padded by one all round, its
// output rows in *out with the layer's bias added.
func (e *Embedder) audioDW(pad, out *[]float32, x []float32, t, f, ch int, w, b []float32) ([]float32, int, int, error) {
	wp := f + 2
	p := growA(pad, (t+2)*wp*ch)
	clear(p)
	nn.Copy32JIT(p[(wp+1)*ch:], wp*ch, 1, x, f*ch, 1, t, f*ch)
	to, fo := (t+2-3)/2+1, (f+2-3)/2+1
	o := growA(out, to*fo*ch)
	s := cpu.DWShape{K: 3, Stride: 2, WP: wp, Chans: ch, WOut: fo}
	for yy := 0; yy < to; yy++ {
		e.jit.DWConvRow(o[yy*fo*ch:(yy+1)*fo*ch], p[2*yy*wp*ch:], w, s)
	}
	e.addBiasRows(o, b, to*fo)
	return o, to, fo, nil
}

// dotF32 is x . w on the generated f32 matvec.
func (e *Embedder) dotF32(x, w []float32) (float32, error) {
	out := e.aud.dot[:]
	b := unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), 4*len(w))
	if !e.jit.MatVecHost(out, quant.F32, b, x, 1, len(x)) {
		return 0, fmt.Errorf("model: no generated f32 matvec of width %d on this host", len(x))
	}
	return out[0], nil
}

// rowsLNW is y[r] = LayerNorm(x[r]; w, b) over n rows of width wd at eps.
func (e *Embedder) rowsLNW(y, x, w, b []float32, n, wd int, eps float64) {
	j := &e.rg.rows
	j.op, j.y, j.x, j.w, j.b, j.d, j.eps = encRowLN, y, x, w, b, wd, eps
	e.rowRun(n)
}

// audioBlocks runs the conformer blocks over x's t rows.
func (e *Embedder) audioBlocks(x []float32, t int) error {
	m, a, r := e.m, e.m.audio, &e.aud
	cf := &a.cfg
	d, nh, ff, kt := int(cf.NEmbd), int(cf.NHead), int(cf.NFFN), int(cf.ConvKernel)
	hd := d / nh
	eps := float64(cf.Eps)
	h, tt := growA(&r.h, t*max(d, int(cf.ProjDim))), growA(&r.t, t*max(d, int(cf.ProjDim)))
	h, tt = h[:t*d], tt[:t*d]
	q, k, v := growA(&r.q, t*d), growA(&r.k, t*d), growA(&r.v, t*d)
	qu, qv := growA(&r.qu, t*d), growA(&r.qv, t*d)
	np := 2*t - 1
	p := growA(&r.p, np*d)
	fbuf := growA(&r.ff, t*ff)
	// The relative positions' sinusoids, t-1 down to -(t-1): a table of the
	// sizes, as the rotary's is.
	pt := growA(&r.posTab, np*d)
	for i := 0; i < np; i++ {
		pos := float64(t - 1 - i)
		for c := 0; c < d/2; c++ {
			div := math.Exp(float64(2*c) * -(math.Log(10000.0) / float64(d)))
			pt[i*d+2*c] = float32(math.Sin(pos * div))
			pt[i*d+2*c+1] = float32(math.Cos(pos * div))
		}
	}
	if r.attn == nil {
		r.attn = e.jit.AttnSetFor(hd, hd, d, cpu.KVF32)
	}
	r.hd, r.scale, r.rows = hd, float32(1/math.Sqrt(float64(hd))), t
	r.attStride = nn.SoftmaxPad(t)
	growA(&r.att, nh*r.attStride)
	growA(&r.bd, nh*r.attStride)
	growA(&r.acc, nh*hd)
	for j := 0; j < len(a.blocks); j++ {
		// A run of placed blocks is one device call over the clip's rows.
		if e.audioOnDevice(j) {
			hi, err := e.audioDevRun(j, x, t)
			if err != nil {
				return err
			}
			j = hi - 1
			continue
		}
		if err := m.audioPageIn(j); err != nil {
			return err
		}
		b := &a.blocks[j]
		if err := e.audioFF(x, h, tt, fbuf[:t*ff], t, d, eps, b.ff1N, b.ff1NB, &b.ff1Up, &b.ff1Down, b.ff1UB, b.ff1DB); err != nil {
			return err
		}
		// Relative-position attention.
		e.rowsLNW(h, x, b.attN, b.attNB, t, d, eps)
		for _, pr := range []struct {
			w   tensor
			b   []float32
			dst []float32
		}{{b.q, b.qB, q}, {b.k, b.kB, k}, {b.v, b.vB, v}} {
			if err := e.mm(pr.dst, pr.w, h, t); err != nil {
				return err
			}
			e.addBiasRows(pr.dst, pr.b, t)
		}
		if err := e.mm(p, b.pos, pt, np); err != nil {
			return err
		}
		copy(qu, q)
		e.addBiasRows(qu, b.u, t)
		copy(qv, q)
		e.addBiasRows(qv, b.vbias, t)
		e.audioAttend(h)
		if err := e.mm(tt, b.o, h, t); err != nil {
			return err
		}
		e.addBiasRows(tt, b.oB, t)
		e.axpyRows(x, tt)
		// The convolution module.
		e.rowsLNW(h, x, b.convN, b.convNB, t, d, eps)
		gv, gg := q, k // free until the next block's projections
		if err := e.mm(gv, b.gluV, h, t); err != nil {
			return err
		}
		e.addBiasRows(gv, b.gluVB, t)
		if err := e.mm(gg, b.gluG, h, t); err != nil {
			return err
		}
		e.addBiasRows(gg, b.gluGB, t)
		nn.SigmoidMul32JIT(gg, gv) // gg = sigma(gate) * value
		e.audioDepthwise(tt, gg, b.dw, t, d, kt)
		e.actMul(tt, e.audioRowsOf(b.bnS, t, d), nn.ActIdentity)
		e.addBiasRows(tt, b.bnB, t)
		e.actRows(tt, nn.ActSiLU)
		if err := e.mm(h, b.pw2, tt, t); err != nil {
			return err
		}
		e.addBiasRows(h, b.pw2B, t)
		if e.violation != "no-conv-module" {
			e.axpyRows(x, h)
		}
		if err := e.audioFF(x, h, tt, fbuf[:t*ff], t, d, eps, b.ff2N, b.ff2NB, &b.ff2Up, &b.ff2Down, b.ff2UB, b.ff2DB); err != nil {
			return err
		}
		e.rowsLNW(x, x, b.outN, b.outNB, t, d, eps)
		if e.audioTap != nil {
			e.audioTap(fmt.Sprintf("block%d", j), x)
		}
	}
	return nil
}

// audioFF is x += FF(LN x)/2, a conformer block's half feed-forward: LN,
// up and its bias, SiLU, down and its bias, halved.
func (e *Embedder) audioFF(x, h, tt, fb []float32, t, d int, eps float64, norm, normB []float32, up, down *tensor,
	upB, downB []float32) error {
	e.rowsLNW(h, x, norm, normB, t, d, eps)
	if err := e.mm(fb, *up, h, t); err != nil {
		return err
	}
	e.addBiasRows(fb, upB, t)
	e.actRows(fb, nn.ActSiLU)
	if err := e.mm(tt, *down, fb, t); err != nil {
		return err
	}
	e.addBiasRows(tt, downB, t)
	nn.Scale32JIT(tt, 0.5)
	e.axpyRows(x, tt)
	return nil
}

// audioRowsOf is a vector of width d repeated over t rows, in the run's
// scratch (the taps' broadcast).
func (e *Embedder) audioRowsOf(w []float32, t, d int) []float32 {
	rows := growA(&e.aud.tapRows, t*d)
	nn.Copy32JIT(rows, d, 1, w, 0, 1, t, d)
	return rows
}

// audioDepthwise is out = the depthwise convolution of x (t rows of d) by
// taps (k of them, tap-major, centred, zero past either end).
func (e *Embedder) audioDepthwise(out, x, taps []float32, t, d, k int) {
	clear(out[:t*d])
	c := growA(&e.aud.cv, t*d)
	half := k / 2
	for u := 0; u < k; u++ {
		s := u - half // out[r] += w[u] * x[r+s]
		lo, hi := max(0, -s), min(t, t-s)
		if lo >= hi {
			continue
		}
		copy(c[:t*d], x[:t*d])
		e.actMul(c[:t*d], e.audioRowsOf(taps[u*d:(u+1)*d], t, d), nn.ActIdentity)
		e.axpyRows(out[lo*d:hi*d], c[(lo+s)*d:(hi+s)*d])
	}
}

// audioAttend writes the relative-position attention of the run's rows into
// out: per head, row i's scores (q+u).k_j + (q+v).P[t-1-i+j], scaled,
// softmaxed over every row, and the values' sum.
func (e *Embedder) audioAttend(out []float32) {
	j := &e.rg.audAttn
	if j.fn == nil {
		j.fn = e.audioAttnHeads
	}
	j.out = out
	e.jit.Parallel(int(e.m.audio.cfg.NHead), 1, j.fn)
	j.out = nil
}

// audioAttnHeads is one chunk of audioAttend: heads h0..h1-1.
func (e *Embedder) audioAttnHeads(h0, h1 int) {
	r := &e.aud
	d := int(e.m.audio.cfg.NEmbd)
	t, hd := r.rows, r.hd
	out := e.rg.audAttn.out
	for h := h0; h < h1; h++ {
		off := h * hd
		att := r.att[h*r.attStride:]
		bd := r.bd[h*r.attStride : h*r.attStride+t]
		acc := r.acc[h*hd : (h+1)*hd]
		for i := 0; i < t; i++ {
			r.attn.AttnScores(att[:t], r.k[off:], r.qu[i*d+off:], t)
			r.attn.AttnScores(bd, r.p[(t-1-i)*d+off:], r.qv[i*d+off:], t)
			if e.violation != "no-relpos" {
				nn.Axpy32JIT(att[:t], bd, 1)
			}
			nn.Scale32JIT(att[:t], r.scale)
			nn.Softmax32JIT(att, t)
			r.attn.AttnAcc(acc, r.v[off:], att[:t], t)
			copy(out[i*d+off:i*d+off+hd], acc)
		}
	}
}

// audAttnRegion is audioAttend's pool region.
type audAttnRegion struct {
	out []float32
	fn  func(lo, hi int)
}
