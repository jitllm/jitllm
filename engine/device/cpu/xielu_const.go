package cpu

// The constant block and the contract every xIELU kernel reads -- Apertus's
// ungated activation. One copy for all three tiers, in an untagged file,
// because the layout is the ABI between the emitters and nn.
//
// The activation, with each block's four numbers already in their effective
// form (the converter applies the softplus once, jlm.RoleXIELU):
//
//	x > 0:   alpha_p*x*x + beta*x
//	x <= 0:  (expm1(min(x, eps)) - x)*alpha_n + beta*x
//
// transformers' XIELUActivation in its Python form, and ggml's op_xielu. With
// m = min(x, eps), expm1(m) - x is computed as g(m) + (m - x), where
// g(m) = expm1(m) - m: near zero g is m^2/2 + m^3/6 + ..., which exp(m) - 1 - m
// in f32 loses entirely (every digit of exp(m) - 1 cancels against m), so g is
// a degree-8 Taylor polynomial for m >= -1/2 -- the next term is m^9/9!, a
// relative 5e-8 there -- and exp(m) - 1 - m below it, where nothing cancels.
//
//	Out     x, transformed in place          (f32, n)
//	AScale  the block's four numbers: alpha_p, alpha_n, beta, eps   (f32, 4)
//	Scr     XIELUConsts()
//	K/Rows  n/ElemLanes whole units and n%ElemLanes tail elements

// XIELUConsts is the block: exp's (expConstsShared, offsets 0..40) and then
//
//	44  -0.5, the polynomial's lower bound
//	48  1/8!, 52 1/7!, 56 1/6!, 60 1/5!, 64 1/4!, 68 1/3!, 72 1/2: g's Horner
//	    coefficients, highest first
func XIELUConsts() []float32 {
	return append(expConstsShared(),
		-0.5,
		float32(1.0/40320), float32(1.0/5040), float32(1.0/720), float32(1.0/120),
		float32(1.0/24), float32(1.0/6), 0.5)
}

const (
	xieluLoOff   = 44
	xieluC8Off   = 48 // the highest coefficient; the rest follow at +4
	xieluNCoef   = 7
	xieluParamAP = 0
	xieluParamAN = 4
	xieluParamB  = 8
	xieluParamE  = 12
)
