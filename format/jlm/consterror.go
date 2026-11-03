package jlm

type constError string

func (e constError) Error() string { return string(e) }
