package convert

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// audioOf reads f's audio tower, its blocks after nVis vision blocks, or nil
// for an mmproj with none or of an audio projector the library does not
// state. The statement (projectors.json's audio entries) gives its
// geometry, its tensors' names and layouts, and its tower.
func audioOf(f *meta.File, nVis uint32, taken map[string]bool) (*jlm.Audio, []jlm.Tensor, error) {
	if v, ok := f.KV["clip.has_audio_encoder"]; !ok || v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] == 0 {
		return nil, nil, nil
	}
	// Another family's audio half (Gemma 3n's) is not built: set aside
	// (withoutAudio), as it was before this one.
	proj, _ := f.KV["clip.audio.projector_type"].String()
	if !slices.Contains(AudioProjectors(), proj) {
		return nil, nil, nil
	}
	st, err := libraryAudio(proj)
	if err != nil {
		return nil, nil, err
	}
	a, err := st.geometry(proj, ggufKeys(f))
	if err != nil {
		return nil, nil, err
	}
	names, err := st.roles(proj)
	if err != nil {
		return nil, nil, err
	}
	lay, err := st.Layout.resolve(proj)
	if err != nil {
		return nil, nil, err
	}
	var out []jlm.Tensor
	tensor := func(t *meta.Tensor, role jlm.Role, block int32, index int32) (jlm.Tensor, error) {
		e := jlm.Tensor{Role: role, Block: block, Index: index, NDim: uint8(len(t.Dims)), Data: f.Bytes(t), Name: t.Name}
		copy(e.Dims[:], t.Dims)
		if w, ok := lay.widths[role]; ok {
			to, n := audioFields(a)[w.field], uint32(e.Dims[w.dim])
			if *to != 0 && *to != n {
				return e, fmt.Errorf("states %s %d where another states %d", w.field, n, *to)
			}
			*to = n
		}
		var err error
		e.Type, err = typeOf(t.Type)
		return e, err
	}
	for i := range f.Tensors {
		t := &f.Tensors[i]
		name := t.Name
		switch {
		case strings.HasPrefix(name, st.Names.BlockPrefix):
			rest := name[len(st.Names.BlockPrefix):]
			dot := strings.IndexByte(rest, '.')
			// A block index the file states, parsed to 31 bits so every
			// conversion below holds it (RULE 9).
			n, err := strconv.ParseInt(rest[:max(dot, 0)], 10, 31)
			if dot <= 0 || err != nil || n < 0 || uint64(n) >= uint64(a.NLayer) {
				return nil, nil, fmt.Errorf("convert: %q: an audio block index outside %d", name, a.NLayer)
			}
			role, ok := names.Block[rest[dot+1:]]
			if !ok {
				return nil, nil, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
					"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
			}
			e, err := tensor(t, role, int32(nVis)+int32(n), -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			es, err := lay.block(e, a)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, es...)
		case strings.HasPrefix(name, st.Names.IndexedPrefix):
			rest := strings.TrimPrefix(name, st.Names.IndexedPrefix)
			dot := strings.IndexByte(rest, '.')
			n64, err := strconv.ParseInt(rest[:max(dot, 0)], 10, 31)
			if err == nil && n64 < 0 {
				err = fmt.Errorf("a negative index")
			}
			if err == nil && (dot <= 0 || rest[dot+1:] != "weight" && rest[dot+1:] != "bias") {
				err = fmt.Errorf("not a weight or a bias")
			}
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %q: %v", name, err)
			}
			role, ok := names.Indexed[rest]
			if !ok {
				return nil, nil, fmt.Errorf("convert: %s: the subsampling has no layer %d", name, n64)
			}
			e, err := tensor(t, role, jlm.DenseBlock, -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			if err := lay.subsampling(&e); err != nil {
				return nil, nil, err
			}
			out = append(out, e)
		case slices.ContainsFunc(st.Names.DensePrefixes, func(p string) bool { return strings.HasPrefix(name, p) }):
			role, ok := names.Dense[name]
			if !ok {
				return nil, nil, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
					"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
			}
			e, err := tensor(t, role, jlm.DenseBlock, -1)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			out = append(out, e)
		}
	}
	if err := st.holds(proj, ggufKeys(f), a, st.RequireRead); err != nil {
		return nil, nil, err
	}
	// A channel-last matrix, once the subsampling's channel count is known.
	for i := range out {
		if lay.channelLast[out[i].Role] {
			if err := channelLast(&out[i], a); err != nil {
				return nil, nil, err
			}
		}
	}
	if lay.foldBias != 0 {
		if out, err = foldBiasShift(out, lay.foldBias, lay.foldScale, lay.foldShift); err != nil {
			return nil, nil, err
		}
	}
	out = append(out, lay.frontTables(a)...)
	if err := quantizeTower(out); err != nil {
		return nil, nil, err
	}
	if a.Tower, err = st.Tower.stateOf("convert/projectors.json", &jlm.Source{Tensors: out}, a.Projector, audioGeom(a),
		towerTook(taken, proj)); err != nil {
		return nil, nil, fmt.Errorf("convert: the audio tower: %w", err)
	}
	if err := a.Tower.Valid(); err != nil {
		return nil, nil, fmt.Errorf("convert: the audio tower: %w", err)
	}
	return a, out, nil
}

// audioRows is the rows an audio block is formed for: the centred frames of
// the longest clip (one more than its hops), halved three times as the
// subsampling's stride-two convolutions padded by one halve them.
func audioRows(a *jlm.Audio) int {
	t := int(a.MaxSamples/a.Hop) + 1
	for range 3 {
		t = (t + 1) / 2
	}
	return t
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
	out := make([]float32, nmels*bins)
	for i := 0; i < nmels; i++ {
		fd0, fd1 := melF[i+1]-melF[i], melF[i+2]-melF[i+1]
		enorm := 2 / (melF[i+2] - melF[i])
		for j := 0; j < bins; j++ {
			f := float64(j) * float64(sr) / float64(nfft)
			v := math.Max(0, math.Min(-(melF[i]-f)/fd0, (melF[i+2]-f)/fd1))
			out[i*bins+j] = float32(v) * float32(enorm)
		}
	}
	return out
}

// audioLayout is how an audio statement lays its tensors out, by role name:
// a tensor cut into halves, the second taking its own role (a GLU's value
// and gate); a depthwise filter of taps by channels; vectors of the tower's
// width; a relative-position projection, which becomes its product with the
// sinusoids the attention reads; a subsampling's convolutions -- full ones
// as matrices, each of a window (kx, ky) over cin input channels (0 for
// any), depthwise 3x3 ones tap-major -- and its flat vectors; matrices whose
// inputs are reordered channel-last; the bias folded into a shift through a
// scale; the tensors whose dimension states a geometry field; and the front
// end's tables.
type audioLayout struct {
	Halves      map[string]string     `json:"halves"`
	Taps        []string              `json:"taps"`
	Vectors     []string              `json:"vectors"`
	Relative    []string              `json:"relative"`
	Conv        map[string][3]int     `json:"conv"`
	Depthwise   []string              `json:"depthwise"`
	Flat        []string              `json:"flat"`
	ChannelLast []string              `json:"channel_last"`
	Fold        audioFold             `json:"fold"`
	Widths      map[string]audioWidth `json:"widths"`
	Tables      audioTables           `json:"tables"`
}

// audioFold is the bias folded into a shift through a scale.
type audioFold struct {
	Bias  string `json:"bias"`
	Scale string `json:"scale"`
	Shift string `json:"shift"`
}

// audioTables is the roles of a front end's tables: the DFT's cos and sin
// bases and the mel filterbank.
type audioTables struct {
	Cos string `json:"cos"`
	Sin string `json:"sin"`
	Mel string `json:"mel"`
}

// audioWidth is the tensor whose dimension Dim, as the file lays it out,
// states a geometry field; every tensor of that role states the same.
type audioWidth struct {
	Of  string `json:"of"`
	Dim int    `json:"dim"`
}

// audioWidthOf is a width with its role resolved.
type audioWidthOf struct {
	field string
	dim   int
}

// audioLay is an audio layout with its roles resolved.
type audioLay struct {
	halves                         map[jlm.Role]jlm.Role
	taps, vectors, relative, flat  map[jlm.Role]bool
	depthwise, channelLast         map[jlm.Role]bool
	conv                           map[jlm.Role][3]int
	widths                         map[jlm.Role]audioWidthOf
	foldBias, foldScale, foldShift jlm.Role
	cos, sin, mel                  jlm.Role
	tables                         audioTables
}

// resolve is l with its roles resolved, refusing a role the container does
// not name.
func (l *audioLayout) resolve(name string) (*audioLay, error) {
	named := rolesByName()
	var err error
	role := func(n string) jlm.Role {
		if n == "" {
			return 0
		}
		r, ok := named[n]
		if !ok && err == nil {
			err = fmt.Errorf("convert: projectors.json: audio %s: %q is no role the container names", name, n)
		}
		return r
	}
	set := func(ns []string) map[jlm.Role]bool {
		out := map[jlm.Role]bool{}
		for _, n := range ns {
			out[role(n)] = true
		}
		return out
	}
	out := &audioLay{halves: map[jlm.Role]jlm.Role{}, taps: set(l.Taps), vectors: set(l.Vectors), relative: set(l.Relative),
		flat: set(l.Flat), depthwise: set(l.Depthwise), channelLast: set(l.ChannelLast), conv: map[jlm.Role][3]int{},
		widths: map[jlm.Role]audioWidthOf{}, tables: l.Tables}
	for a, b := range l.Halves {
		out.halves[role(a)] = role(b)
	}
	for n, w := range l.Conv {
		out.conv[role(n)] = w
	}
	for field, w := range l.Widths {
		if audioFields(&jlm.Audio{})[field] == nil || w.Of == "" || w.Dim < 0 || w.Dim >= len(jlm.Tensor{}.Dims) {
			return nil, fmt.Errorf("convert: projectors.json: audio %s: width %s is no geometry field, or no tensor's dimension", name, field)
		}
		out.widths[role(w.Of)] = audioWidthOf{field, w.Dim}
	}
	out.foldBias, out.foldScale, out.foldShift = role(l.Fold.Bias), role(l.Fold.Scale), role(l.Fold.Shift)
	out.cos, out.sin, out.mel = role(l.Tables.Cos), role(l.Tables.Sin), role(l.Tables.Mel)
	return out, err
}

// block is one block tensor in the engine's layout: halves cut apart, a
// depthwise filter F32 as the file lays it out (channel-major), a vector of
// the width one vector, and a relative-position projection the table of the
// projected positions.
func (l *audioLay) block(e jlm.Tensor, a *jlm.Audio) ([]jlm.Tensor, error) {
	d := int(a.NEmbd)
	switch {
	case l.halves[e.Role] != 0:
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		if len(vals)%2 != 0 {
			return nil, fmt.Errorf("convert: %s has %d values, not two halves", e.Name, len(vals))
		}
		half := len(vals) / 2
		g := e
		g.Name = e.Name + "/gate"
		g.Role = l.halves[e.Role]
		if e.NDim >= 2 {
			setF32Dims(&e, vals[:half], uint64(d), uint64(half/d))
			setF32Dims(&g, vals[half:], uint64(d), uint64(half/d))
		} else {
			setF32Dims(&e, vals[:half], uint64(half))
			setF32Dims(&g, vals[half:], uint64(half))
		}
		return []jlm.Tensor{e, g}, nil
	case l.taps[e.Role]:
		// [chan][tap] in the file ({k, d}), as graph.OpRowConv reads it.
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		k := int(e.Dims[0])
		if e.NDim != 2 || int(e.Dims[1]) != d || k < 1 || k%2 != 1 {
			return nil, fmt.Errorf("convert: %s is %v, want an odd number of taps over %d channels",
				e.Name, e.Dims[:e.NDim], d)
		}
		setF32Dims(&e, vals, uint64(k), uint64(d))
	case l.vectors[e.Role]:
		vals, err := normF32(&e)
		if err != nil {
			return nil, err
		}
		if len(vals) != d {
			return nil, fmt.Errorf("convert: %s has %d values over %d channels", e.Name, len(vals), d)
		}
		setF32Dims(&e, vals, uint64(d))
	case l.relative[e.Role]:
		if err := relativePositions(&e, d, audioRows(a)); err != nil {
			return nil, err
		}
	}
	return []jlm.Tensor{e}, nil
}

// relativePositions is a d x d projection times the sinusoids of every
// position rows reach, rows-1 down to 1-rows (sin at the even elements, cos
// at the odd, NeMo's RelPositionalEncoding): the product a relative
// attention reads, computed here once rather than every clip.
func relativePositions(e *jlm.Tensor, d, rows int) error {
	w, err := normF32(e)
	if err != nil {
		return err
	}
	if e.NDim != 2 || int(e.Dims[0]) != d || int(e.Dims[1]) != d || d%2 != 0 {
		return fmt.Errorf("convert: %s is %v, want %dx%d", e.Name, e.Dims[:e.NDim], d, d)
	}
	n := 2*rows - 1
	pe := make([]float64, d)
	p := make([]float32, n*d)
	for k := 0; k < n; k++ {
		pos := float64(rows - 1 - k)
		for c := 0; c < d/2; c++ {
			div := math.Exp(float64(2*c) * -(math.Log(10000.0) / float64(d)))
			pe[2*c], pe[2*c+1] = math.Sin(pos*div), math.Cos(pos*div)
		}
		for r := 0; r < d; r++ {
			s := 0.0
			for c := 0; c < d; c++ {
				s += float64(w[r*d+c]) * pe[c]
			}
			p[k*d+r] = float32(s)
		}
	}
	setF32Dims(e, p, uint64(d), uint64(n))
	return nil
}

// subsampling is a subsampling layer's tensor in the engine's layout: a full
// convolution a matrix of out x (in*ky*kx), the file's own order; a
// depthwise 3x3 one tap-major (ky, kx, c); a flat one a vector.
func (l *audioLay) subsampling(e *jlm.Tensor) error {
	vals, err := normF32(e)
	if err != nil {
		return err
	}
	if l.flat[e.Role] {
		setF32Dims(e, vals, uint64(len(vals)))
		return nil
	}
	if e.NDim != 4 {
		return fmt.Errorf("convert: %s is %d-dimensional, want a 4-D convolution", e.Name, e.NDim)
	}
	kx, ky, cin, cout := int(e.Dims[0]), int(e.Dims[1]), int(e.Dims[2]), int(e.Dims[3])
	if w, ok := l.conv[e.Role]; ok {
		if kx != w[0] || ky != w[1] || w[2] != 0 && cin != w[2] {
			return fmt.Errorf("convert: %s is %dx%dx%dx%d, not a %dx%d convolution over %d channels",
				e.Name, kx, ky, cin, cout, w[0], w[1], w[2])
		}
		setF32Dims(e, vals, uint64(cin*ky*kx), uint64(cout))
		return nil
	}
	if !l.depthwise[e.Role] {
		return fmt.Errorf("convert: %s: the subsampling lays out no %v", e.Name, e.Role)
	}
	if cin != 1 || kx != 3 || ky != 3 {
		return fmt.Errorf("convert: %s is %dx%dx%dx%d, not a 3x3 depthwise convolution", e.Name, kx, ky, cin, cout)
	}
	tm := make([]float32, len(vals))
	for c := 0; c < cout; c++ {
		for u := 0; u < 9; u++ {
			tm[u*cout+c] = vals[c*9+u]
		}
	}
	setF32Dims(e, tm, uint64(cout), 9)
	return nil
}

// channelLast reorders a matrix's inputs from a channel-major flattening
// (c*F + f) of the subsampling's channels to channel-last (f*C + c).
func channelLast(e *jlm.Tensor, a *jlm.Audio) error {
	vals, err := normF32(e)
	if err != nil {
		return err
	}
	k, rows := int(e.Dims[0]), int(e.Dims[1])
	c := int(a.SubChannels)
	if c == 0 || k%c != 0 {
		return fmt.Errorf("convert: %s reads %d inputs, not a whole number of %d channels", e.Name, k, c)
	}
	fr := k / c
	out := make([]float32, len(vals))
	for r := 0; r < rows; r++ {
		for ch := 0; ch < c; ch++ {
			for f := 0; f < fr; f++ {
				out[r*k+f*c+ch] = vals[r*k+ch*fr+f]
			}
		}
	}
	setF32Dims(e, out, uint64(k), uint64(rows))
	return nil
}

// foldBiasShift folds each block's bias into its shift through its scale,
// bias*scale + shift, and drops the bias: what the bias feeds is the affine
// alone, so the module is one affine and no load or token folds it again.
func foldBiasShift(out []jlm.Tensor, biasRole, scaleRole, shiftRole jlm.Role) ([]jlm.Tensor, error) {
	type trio struct{ bias, scale, shift int }
	blocks := map[int32]*trio{}
	at := func(b int32) *trio {
		if blocks[b] == nil {
			blocks[b] = &trio{-1, -1, -1}
		}
		return blocks[b]
	}
	for i := range out {
		switch out[i].Role {
		case biasRole:
			at(out[i].Block).bias = i
		case scaleRole:
			at(out[i].Block).scale = i
		case shiftRole:
			at(out[i].Block).shift = i
		}
	}
	for b, t := range blocks {
		if t.bias < 0 || t.scale < 0 || t.shift < 0 {
			return nil, fmt.Errorf("convert: audio block %d carries the bias, the scale and the shift only in part", b)
		}
		bias, err := normF32(&out[t.bias])
		if err != nil {
			return nil, err
		}
		scale, err := normF32(&out[t.scale])
		if err != nil {
			return nil, err
		}
		shift, err := normF32(&out[t.shift])
		if err != nil {
			return nil, err
		}
		if len(bias) != len(scale) || len(shift) != len(scale) {
			return nil, fmt.Errorf("convert: audio block %d: bias %d, scale %d and shift %d wide", b, len(bias), len(scale), len(shift))
		}
		folded := make([]float32, len(shift))
		for c := range folded {
			folded[c] = bias[c]*scale[c] + shift[c]
		}
		setF32Dims(&out[t.shift], folded, uint64(len(folded)))
	}
	kept := out[:0]
	for _, e := range out {
		if e.Role != biasRole {
			kept = append(kept, e)
		}
	}
	return kept, nil
}

// frontTables is the front end's tables l states, computed in float64 and
// stored F32: the DFT bases with a symmetric Hann window of Window samples
// folded in (torch.hann_window(Window, periodic=False)), centred in the
// NFFT frame as torch.stft pads it, and librosa's Slaney filterbank.
func (l *audioLay) frontTables(a *jlm.Audio) []jlm.Tensor {
	n, win, nm := int(a.NFFT), int(a.Window), int(a.MelBins)
	bins := n/2 + 1
	var out []jlm.Tensor
	add := func(role jlm.Role, name string, vals []float32, k, rows int) {
		e := jlm.Tensor{Role: role, Block: jlm.DenseBlock, Index: -1, Name: name}
		setF32Dims(&e, vals, uint64(k), uint64(rows))
		out = append(out, e)
	}
	if l.cos != 0 || l.sin != 0 {
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
		if l.cos != 0 {
			add(l.cos, l.tables.Cos, cs, n, bins)
		}
		if l.sin != 0 {
			add(l.sin, l.tables.Sin, sn, n, bins)
		}
	}
	if l.mel != 0 {
		add(l.mel, l.tables.Mel, slaneyFilterbank(int(a.SampleRate), n, nm), bins, nm)
	}
	return out
}

// withoutAudio is an mmproj with its audio encoder's tensors (a.*, mm.a.*)
// set aside, for the vision half's reading; audioOf reads them. A file with
// no audio encoder comes back as it is.
func withoutAudio(f *meta.File) *meta.File {
	if v, ok := f.KV["clip.has_audio_encoder"]; !ok || v.Type != meta.Bool || len(v.Raw) != 1 || v.Raw[0] == 0 {
		return f
	}
	var ts []meta.Tensor
	for _, t := range f.Tensors {
		if !strings.HasPrefix(t.Name, "a.") && !strings.HasPrefix(t.Name, "mm.a.") {
			ts = append(ts, t)
		}
	}
	g := *f
	g.Tensors = ts
	g.Reindex()
	return &g
}
