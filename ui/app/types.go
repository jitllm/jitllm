package app

import (
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/widgets"
)

// DialogReq is re-exported so a screen can raise a dialog with one import.
type DialogReq = widgets.DialogReq

// The dialog shapes(), re-exported.
const (
	DialogAlert   = widgets.DialogAlert
	DialogConfirm = widgets.DialogConfirm
)

// The session types, re-exported so a screen reads them from one package.
type (
	Role          = session.Role
	Turn          = session.Turn
	MachineReport = session.MachineReport
	GPUInfo       = session.GPUInfo
	PagerStat     = session.PagerStat
	DeviceUse     = session.DeviceUse
	Loading       = session.Loading
	Allocation    = session.Allocation
	LoadedModel   = session.LoadedModel
	Problem       = session.Problem
	Sampling      = session.Sampling
)

// The turn roles, re-exported.
const (
	RoleUser      = session.RoleUser
	RoleAssistant = session.RoleAssistant
	RoleSystem    = session.RoleSystem
	RoleError     = session.RoleError
)
