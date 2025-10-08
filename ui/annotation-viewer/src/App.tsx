import { useCallback, useEffect, useMemo, useState, type ChangeEvent, type CSSProperties } from 'react'
import { useDisagreementsQuery, useRunDetailQuery, useRunsQuery } from './lib/api/queries'
import type { StepResponse } from './lib/api/schemas'
import type { InsightRow, TokenCategory } from './lib/insightRow'
import { RunsSidebar } from './components/RunsSidebar'
import { MoveInsights } from './components/MoveInsights'
import { TokenizationControls } from './components/TokenizationControls'
import { useViewerPreferences } from './hooks/useViewerPreferences'
import './App.css'

const MOVE_LABELS = ['Up', 'Down', 'Left', 'Right']
const MOVE_ICONS = ['↑', '↓', '←', '→']
const RUNS_PAGE_SIZE = 25
const WINDOW_RADIUS = 128
const WINDOW_SIZE = WINDOW_RADIUS * 2 + 1
const SAFETY_BUFFER = 8
const SEGMENT_LENGTH = 16
const FOCUS_RADIUS = 8

const TILE_COLORS: Record<number, { bg: string; fg: string }> = {
  0: { bg: '#d7cec3', fg: '#6e645c' },
  1: { bg: '#f3ede4', fg: '#6e645c' },
  2: { bg: '#eee4da', fg: '#6e645c' },
  3: { bg: '#f2b179', fg: '#f9f6f2' },
  4: { bg: '#f59563', fg: '#f9f6f2' },
  5: { bg: '#f67c5f', fg: '#f9f6f2' },
  6: { bg: '#f65e3b', fg: '#f9f6f2' },
  7: { bg: '#edcf72', fg: '#f9f6f2' },
  8: { bg: '#edcc61', fg: '#f9f6f2' },
  9: { bg: '#edc850', fg: '#f9f6f2' },
  10: { bg: '#edc53f', fg: '#f9f6f2' },
  11: { bg: '#edc22e', fg: '#f9f6f2' },
  12: { bg: '#c29d48', fg: '#fdfcf7' },
  13: { bg: '#a47c3b', fg: '#fdfcf7' },
  14: { bg: '#7d5b83', fg: '#f9f6f2' },
  15: { bg: '#5a458b', fg: '#f9f6f2' },
  16: { bg: '#3f3d7a', fg: '#f9f6f2' },
  17: { bg: '#2f2d5f', fg: '#f3f2fa' },
}

const formatTile = (exponent: number) => {
  if (exponent <= 0) {
    return ''
  }
  return (2 ** exponent).toLocaleString()
}

const decodeLegalMask = (mask: number) =>
  MOVE_LABELS.filter((_, idx) => (mask & (1 << idx)) !== 0)

const tileStyle = (exponent: number): CSSProperties => {
  const palette = TILE_COLORS[exponent] ?? { bg: '#3c3a32', fg: '#f9f6f2' }
  const gradient = `linear-gradient(135deg, ${palette.bg}, ${palette.bg}d9)`
  return {
    backgroundImage: gradient,
    color: palette.fg,
  }
}

const normalize = (values: Array<number | null>) => {
  const finiteValues = values.filter((val): val is number => typeof val === 'number' && Number.isFinite(val))
  if (finiteValues.length === 0) {
    return values.map(() => null)
  }
  const max = Math.max(...finiteValues)
  const min = Math.min(...finiteValues)
  if (max === min) {
    return values.map((val) => (val === null ? null : 1))
  }
  return values.map((val) => (val === null ? null : (val - min) / (max - min)))
}

const formatPercent = (value: number | null | undefined) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }
  if (value === 0 || Math.abs(value) < 1e-3) {
    return '0%'
  }
  const percent = value * 100
  if (percent >= 1) {
    return `${percent.toFixed(3)}%`
  }
  if (percent >= 0.01) {
    return `${percent.toFixed(4)}%`
  }
  return `${percent.toFixed(5)}%`
}

const formatTeacherEvDisplay = (value: number | null, mode: 'relative' | 'raw'): string => {
  if (value === null || Number.isNaN(value)) return '—'
  if (mode === 'relative') {
    if (value === 0) return '+0'
    return `${value > 0 ? '+' : ''}${value}`
  }
  return value.toFixed(4)
}

const formatAdvantageDisplay = (value: number | null, mode: 'relative' | 'raw'): string => {
  if (value === null || Number.isNaN(value)) return '—'
  if (mode === 'relative') {
    if (value === 0) return '+0'
    return `${value > 0 ? '+' : ''}${value}`
  }
  const magnitude = Math.abs(value)
  const rounded = magnitude < 0.00005 ? 0 : magnitude
  const sign = rounded === 0 ? '+' : value >= 0 ? '+' : '-'
  return `${sign}${rounded.toFixed(4)}`
}

const formatAdvantageIdDisplay = (value: number | null, mode: 'relative' | 'raw'): string => {
  if (value === null || Number.isNaN(value)) return '—'
  if (mode === 'relative') {
    if (value === 0) return '+0'
    return `${value > 0 ? '+' : ''}${value}`
  }
  const scaled = value / 1000
  if (Math.abs(scaled) < 0.0005) {
    return '+0.000'
  }
  const magnitude = Math.abs(scaled)
  const sign = scaled >= 0 ? '+' : '-'
  return `${sign}${magnitude.toFixed(3)}`
}

const branchRows = (
  step: StepResponse | undefined,
  options: { vocabOrder?: string[] | null } = {},
): InsightRow[] => {
  if (!step) return []
  const { vocabOrder = null } = options

  const valuationMode: 'relative' | 'raw' =
    step.valuation_type.toLowerCase() === 'search' ? 'relative' : 'raw'

  const rawEvs = step.branch_evs.map((value) => (value === null ? null : value))
  const relativeEvs = step.relative_branch_evs.map((value) => (value === null ? null : value))
  const teacherValues = valuationMode === 'relative' ? relativeEvs : rawEvs
  const normalizedTeacher = normalize(teacherValues)

  const finiteTeacher = teacherValues.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  )
  const bestTeacher = finiteTeacher.length ? Math.max(...finiteTeacher) : null
  const advantageValues = teacherValues.map((value) => {
    if (value === null || bestTeacher === null) return null
    return value - bestTeacher
  })
  const normalizedAdvantage = normalize(advantageValues)

  const student = step.annotation
  const studentProb = student ? student.policy_p1 : null
  const studentLogp = student ? student.policy_logp : null
  const studentProbNormalized: Array<number | null> = student
    ? normalize(student.policy_p1.map((value) => value))
    : [null, null, null, null]
  const annotationMask = student?.policy_kind_mask ?? 0
  const tokens = step.tokens ?? null
  const vocabLength = vocabOrder ? vocabOrder.length : 0

  return MOVE_LABELS.map((label, idx) => {
    const legal = (step.legal_mask & (1 << idx)) !== 0
    const teacher_ev = teacherValues[idx]
    const advantage = advantageValues[idx]
    const probability = studentProb ? studentProb[idx] : null
    const probExp = studentLogp ? Math.exp(studentLogp[idx]) : null
    const rawToken = tokens && idx < tokens.length ? tokens[idx] : null

    let tokenId: number | null = rawToken ?? null
    let tokenCategory: TokenCategory | null = null
    let tokenLabel: string | null = null
    let tokenBinDisplay: string | null = null

    if (tokenId !== null && Number.isFinite(tokenId)) {
      if (tokenId === 0) {
        tokenCategory = 'illegal'
        tokenLabel = 'Illegal'
      } else if (tokenId === 1) {
        tokenCategory = 'failure'
        tokenLabel = 'Failure'
      } else if (vocabLength > 0 && tokenId === vocabLength - 1) {
        tokenCategory = 'winner'
        tokenLabel = 'Winner'
      } else if (tokenId >= 0) {
        tokenCategory = 'bin'
        const binIndex = tokenId - 2
        if (binIndex >= 0) {
          tokenLabel = `Bin ${String(binIndex + 1).padStart(2, '0')}`
        } else {
          tokenLabel = `Bin ${tokenId}`
        }
      }

      const vocabLabel =
        vocabOrder && tokenId >= 0 && tokenId < vocabOrder.length
          ? vocabOrder[tokenId]
          : null
      if (vocabLabel) {
        const normalized = vocabLabel.replace(/_/g, ' ')
        tokenBinDisplay = normalized
      } else if (tokenCategory === 'bin' && tokenId !== null) {
        tokenBinDisplay = `Class ${tokenId}`
      }
    }

    const advantageId = step.advantage_branch[idx] ?? null
    const advantageIdDisplay =
      advantageId !== null ? formatAdvantageIdDisplay(advantageId, valuationMode) : null

    return {
      label,
      icon: MOVE_ICONS[idx],
      teacher_ev,
      teacher_ev_normalized: normalizedTeacher[idx],
      teacher_ev_mode: valuationMode,
      teacher_ev_display: formatTeacherEvDisplay(teacher_ev, valuationMode),
      advantage,
      advantage_display: formatAdvantageDisplay(advantage, valuationMode),
      advantage_normalized: normalizedAdvantage[idx],
      probability,
      probabilityNormalized: studentProbNormalized[idx],
      probFromLogp: probExp,
      legal,
      isTeacher: step.teacher_move === idx,
      isStudent: student?.argmax_head === idx,
      hasP1: Boolean(annotationMask & 1),
      hasLogp: Boolean(annotationMask & 2),
      tokenId,
      tokenLabel,
      tokenCategory,
      tokenBinDisplay,
      advantageId,
      advantageIdDisplay,
    }
  })
}

function App() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [selectedStep, setSelectedStep] = useState(0)
  const [windowOffset, setWindowOffset] = useState(0)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const tokenizationMode = useViewerPreferences((state) => state.tokenizationMode)
  const tokenizerInfo = useViewerPreferences((state) => state.tokenizerInfo)
  const tokenizationEnabled = tokenizationMode === 'preview' && tokenizerInfo !== null

  const runsQuery = useRunsQuery({ pageSize: RUNS_PAGE_SIZE })

  useEffect(() => {
    if (!runsQuery.isSuccess) return
    const firstRun = runsQuery.data.runs[0]
    if (firstRun && selectedRunId === null) {
      setSelectedRunId(firstRun.run_id)
    }
  }, [runsQuery.isSuccess, runsQuery.data, selectedRunId])

  useEffect(() => {
    setSelectedStep(0)
    setWindowOffset(0)
  }, [selectedRunId])

  const runDetailQuery = useRunDetailQuery(
    selectedRunId,
    { offset: windowOffset, limit: WINDOW_SIZE, tokenize: tokenizationEnabled },
    {
      placeholderData: (previous) => previous,
    },
  )

  const disagreementsQuery = useDisagreementsQuery(selectedRunId)
  const disagreements = disagreementsQuery.data?.disagreements ?? []

  const steps = runDetailQuery.data?.steps ?? []
  const pagination = runDetailQuery.data?.pagination
  const totalSteps = pagination?.total ?? 0

  const disagreementSegments = useMemo(() => {
    if (!totalSteps || disagreements.length === 0) return null
    const segmentCount = Math.ceil(totalSteps / SEGMENT_LENGTH)
    const segments: Array<{ pct: number; start: number; end: number }> = []
    for (let i = 0; i < segmentCount; i++) {
      const globalStart = i * SEGMENT_LENGTH
      const globalEnd = Math.min(globalStart + SEGMENT_LENGTH, totalSteps)
      const count = disagreements.filter(idx => idx >= globalStart && idx < globalEnd).length
      const span = Math.max(globalEnd - globalStart, 1)
      segments.push({
        pct: count / span,
        start: globalStart,
        end: globalEnd,
      })
    }
    return segments
  }, [disagreements, totalSteps])

  const disagreementOverviewGradient = useMemo(() => {
    if (!disagreementSegments) return null
    const stops: string[] = []
    const totalSpan = Math.max(totalSteps - 1, 1)
    disagreementSegments.forEach(({ pct, start, end }) => {
      const intensity = Math.pow(pct, 0.7)
      const alpha = 0.1 + 0.9 * intensity
      const r = 200
      const g = 64
      const b = 46
      const startPct = (start / totalSpan) * 100
      const endPct = (Math.max(end - 1, start) / totalSpan) * 100
      const color = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`
      stops.push(`${color} ${startPct.toFixed(2)}%`, `${color} ${endPct.toFixed(2)}%`)
    })
    return `linear-gradient(90deg, ${stops.join(', ')})`
  }, [disagreementSegments, totalSteps])

  const focusBandGradient = useMemo(() => {
    if (!totalSteps || totalSteps <= 1) return null
    const totalSpan = totalSteps - 1
    const start = Math.max(selectedStep - FOCUS_RADIUS, 0)
    const end = Math.min(selectedStep + FOCUS_RADIUS, totalSteps - 1)
    const startPct = (start / totalSpan) * 100
    const endPct = (end / totalSpan) * 100
    const softenedStart = Math.max(startPct - 0.6, 0)
    const softenedEnd = Math.min(endPct + 0.6, 100)
    return `linear-gradient(90deg,
      transparent 0%,
      transparent ${softenedStart.toFixed(2)}%,
      rgba(255, 214, 181, 0.06) ${softenedStart.toFixed(2)}%,
      rgba(255, 184, 130, 0.16) ${startPct.toFixed(2)}%,
      rgba(255, 184, 130, 0.16) ${endPct.toFixed(2)}%,
      rgba(255, 214, 181, 0.06) ${softenedEnd.toFixed(2)}%,
      transparent ${softenedEnd.toFixed(2)}%,
      transparent 100%)`
  }, [selectedStep, totalSteps])

  useEffect(() => {
    if (!totalSteps) return
    if (selectedStep > totalSteps - 1) {
      setSelectedStep(totalSteps - 1)
    }
  }, [totalSteps, selectedStep])

  useEffect(() => {
    if (!totalSteps) return

    const maxOffset = Math.max(totalSteps - WINDOW_SIZE, 0)
    const currentStart = steps.length > 0 ? steps[0].step_index : windowOffset
    const effectiveEnd = steps.length > 0 ? steps[steps.length - 1].step_index : windowOffset + WINDOW_SIZE - 1
    const currentEnd = Math.min(effectiveEnd, totalSteps - 1)

    const computeOffset = (stepIndex: number) =>
      Math.min(Math.max(stepIndex - WINDOW_RADIUS, 0), maxOffset)

    if (selectedStep < currentStart || selectedStep > currentEnd) {
      const nextOffset = computeOffset(selectedStep)
      if (nextOffset !== windowOffset) {
        setWindowOffset(nextOffset)
      }
      return
    }

    const distanceFromStart = selectedStep - currentStart
    const distanceFromEnd = currentEnd - selectedStep

    if (distanceFromStart <= SAFETY_BUFFER && windowOffset > 0) {
      const nextOffset = computeOffset(selectedStep)
      if (nextOffset !== windowOffset) {
        setWindowOffset(nextOffset)
      }
      return
    }

    if (distanceFromEnd <= SAFETY_BUFFER && currentEnd < totalSteps - 1) {
      const nextOffset = computeOffset(selectedStep)
      if (nextOffset !== windowOffset) {
        setWindowOffset(nextOffset)
      }
    }
  }, [selectedStep, totalSteps, windowOffset, steps])

  useEffect(() => {
    if (!steps.length) return
    const found = steps.find((step) => step.step_index === selectedStep)
    if (found) return

    const requestedEndBase = windowOffset + WINDOW_SIZE - 1
    const requestedEnd = totalSteps
      ? Math.min(requestedEndBase, totalSteps - 1)
      : requestedEndBase
    const withinRequestedWindow =
      selectedStep >= windowOffset && selectedStep <= requestedEnd

    if (withinRequestedWindow) {
      return
    }

    const clampedIndex = Math.max(selectedStep - windowOffset, 0)
    const fallback = steps[Math.min(steps.length - 1, clampedIndex)] ?? steps[0]
    if (fallback && fallback.step_index !== selectedStep) {
      setSelectedStep(fallback.step_index)
    }
  }, [steps, selectedStep, windowOffset, totalSteps])

  const selectedIndex = useMemo(
    () => steps.findIndex((step) => step.step_index === selectedStep),
    [steps, selectedStep],
  )

  const selectedStepData = selectedIndex >= 0 ? steps[selectedIndex] : steps[0]
  const legalMoves = decodeLegalMask(selectedStepData?.legal_mask ?? 0)
  const legalMoveIcons = useMemo(() => {
    if (!selectedStepData) return []
    return MOVE_ICONS.filter((_, idx) => (selectedStepData.legal_mask & (1 << idx)) !== 0)
  }, [selectedStepData])
  const vocabOrder = tokenizerInfo?.vocabOrder ?? null
  const rows = useMemo<InsightRow[]>(() => branchRows(selectedStepData, { vocabOrder }), [selectedStepData, vocabOrder])

  const teacherMove = selectedStepData?.teacher_move ?? null
  const studentMove = selectedStepData?.annotation?.argmax_head ?? null
  const hasTeacherMove = teacherMove !== null && teacherMove !== 255
  const hasStudentMove = studentMove !== null && studentMove !== undefined
  const movesDisagree = hasTeacherMove && hasStudentMove && teacherMove !== studentMove
  const showTokenization = tokenizationEnabled

  const disagreementTargets = useMemo(() => {
    let prev: number | null = null
    let next: number | null = null
    let current: number | null = null

    for (const index of disagreements) {
      if (index < selectedStep) {
        prev = index
        continue
      }
      if (index === selectedStep) {
        current = index
        continue
      }
      next = index
      break
    }

    return { prev, next, current }
  }, [disagreements, selectedStep])

  const prevDisagreement = disagreementTargets.prev
  const nextDisagreement = disagreementTargets.next

  const disagreementsLoading = disagreementsQuery.isLoading || disagreementsQuery.isFetching

  const handleStepDelta = useCallback((delta: number) => {
    if (!totalSteps) return
    setSelectedStep((current) => {
      const next = Math.min(Math.max(current + delta, 0), Math.max(totalSteps - 1, 0))
      return next
    })
  }, [totalSteps])

  const jumpToPrevDisagreement = useCallback(() => {
    if (prevDisagreement !== null) {
      setSelectedStep(prevDisagreement)
    }
  }, [prevDisagreement, setSelectedStep])

  const jumpToNextDisagreement = useCallback(() => {
    if (nextDisagreement !== null) {
      setSelectedStep(nextDisagreement)
    }
  }, [nextDisagreement, setSelectedStep])

  const prevDisagreementDisabled = disagreementsLoading || prevDisagreement === null
  const nextDisagreementDisabled = disagreementsLoading || nextDisagreement === null

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement) {
        const tagName = event.target.tagName
        if (tagName === 'INPUT' || tagName === 'TEXTAREA' || event.target.isContentEditable) {
          return
        }
      }

      if (event.key === 'h' || event.key === 'ArrowLeft') {
        handleStepDelta(-1)
      } else if (event.key === 'l' || event.key === 'ArrowRight') {
        handleStepDelta(1)
      } else if (event.key === '[') {
        event.preventDefault()
        jumpToPrevDisagreement()
      } else if (event.key === ']') {
        event.preventDefault()
        jumpToNextDisagreement()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [handleStepDelta, jumpToPrevDisagreement, jumpToNextDisagreement])

  const canStepBackward = selectedStep > 0
  const canStepForward = totalSteps > 0 && selectedStep < totalSteps - 1

  const sliderMax = Math.max(totalSteps - 1, 0)
  const sliderValue = Math.min(selectedStep, sliderMax)

  const handleSliderChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedStep(Number(event.target.value))
  }

  const handleRunSelect = (runId: number) => {
    if (runId === selectedRunId) return
    setSelectedRunId(runId)
  }

  return (
    <div className={`app-shell${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
      <RunsSidebar
        runsQuery={runsQuery}
        selectedRunId={selectedRunId}
        onRunSelect={handleRunSelect}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
      />
      <main className="main-content">
        {selectedRunId === null ? (
          <div className="status-card">Select a run to begin.</div>
        ) : runDetailQuery.isLoading && steps.length === 0 ? (
          <div className="status-card">Loading run {selectedRunId}…</div>
        ) : runDetailQuery.isError ? (
          <div className="status-card status-error">
            Unable to load run details. Try selecting another run.
          </div>
        ) : runDetailQuery.data ? (
          <div className="run-content">
            <div className="run-statusline">
              <div className="status-item">
                <span className="status-label">Run</span>
                <span className="status-value">#{runDetailQuery.data.run.run_id}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Seed</span>
                <span className="status-value status-mono">
                  {runDetailQuery.data.run.seed.toString()}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Max score</span>
                <span className="status-value">
                  {runDetailQuery.data.run.max_score.toLocaleString()}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Highest tile</span>
                <span className="status-value">{runDetailQuery.data.run.highest_tile}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Steps</span>
                <span className="status-value">
                  {runDetailQuery.data.run.steps.toLocaleString()}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Disagreements</span>
                <span className="status-value">
                  {(runDetailQuery.data.run.disagreement_percentage * 100).toFixed(1)}%
                </span>
              </div>
              <TokenizationControls />
             </div>

             <section className="disagreement-overview-section">
               <div className="disagreement-overview-track">
                 {disagreementOverviewGradient ? (
                   <div
                     className="disagreement-overview-density"
                     style={{ background: disagreementOverviewGradient }}
                     aria-hidden="true"
                   />
                 ) : null}
               </div>
             </section>

             <section className="scrubber-section">
                <div className="scrubber-meta">
                  <div className="scrubber-info">
                    <span className="scrubber-step">
                      Step{' '}
                      <span className="scrubber-step-value">
                        {selectedStepData ? selectedStepData.step_index + 1 : '—'}
                      </span>
                      <span className="scrubber-step-total">
                        {' '}/ {totalSteps.toLocaleString() || '—'}
                      </span>
                    </span>
                    <span className="scrubber-legals">
                      Legal:{' '}
                      {legalMoves.length ? (
                        <span className="scrubber-legal-list">
                          {legalMoveIcons.map((icon, idx) => (
                            <span
                              key={idx}
                              className="scrubber-legal-icon"
                              aria-hidden="true"
                            >
                              {icon}
                            </span>
                          ))}
                        </span>
                      ) : (
                        'None'
                      )}
                    </span>
                  </div>
                  <div className="scrubber-action-group">
                    <div className="scrubber-disagreement-controls">
                      <button
                        type="button"
                        className="scrubber-button scrubber-disagree-button"
                        onClick={jumpToPrevDisagreement}
                        disabled={prevDisagreementDisabled}
                        aria-label="Jump to previous disagreement"
                        aria-keyshortcuts="["
                        title="Prev disagreement ["
                      >
                        ← Δ
                      </button>
                      <button
                        type="button"
                        className="scrubber-button scrubber-disagree-button"
                        onClick={jumpToNextDisagreement}
                        disabled={nextDisagreementDisabled}
                        aria-label="Jump to next disagreement"
                        aria-keyshortcuts="]"
                        title="Next disagreement ]"
                      >
                        Δ →
                      </button>
                    </div>
                    <div className="scrubber-controls">
                      <button
                        type="button"
                        className="scrubber-button"
                        onClick={() => handleStepDelta(-1)}
                        disabled={!canStepBackward}
                        aria-label="Go to previous step"
                      >
                        ←
                      </button>
                      <button
                        type="button"
                        className="scrubber-button"
                        onClick={() => handleStepDelta(1)}
                        disabled={!canStepForward}
                        aria-label="Go to next step"
                      >
                        →
                      </button>
                    </div>
                  </div>
                </div>
                 <div className="scrubber-track">
                   {focusBandGradient ? (
                     <div
                       className="scrubber-focus"
                       style={{ background: focusBandGradient }}
                       aria-hidden="true"
                     />
                   ) : null}
                   <input
                     className="scrubber-slider"
                     type="range"
                     min={0}
                     max={sliderMax}
                     value={sliderValue}
                     onChange={handleSliderChange}
                     disabled={totalSteps <= 1}
                     style={{
                       background: 'transparent'
                     }}
                   />
                 </div>
              </section>


            <section className="board-insights">
              <div className="board-column">
                 <div className="board-frame">
                   <div className="board-grid">
                     {selectedStepData ? (
                       selectedStepData.board.map((value, idx) => (
                         <div
                           key={idx}
                           className={`board-cell${value === 0 ? ' empty' : ''}`}
                           style={tileStyle(value)}
                         >
                           {formatTile(value)}
                         </div>
                       ))
                     ) : (
                       <div className="board-placeholder">Select a step</div>
                     )}
                   </div>
                 </div>
                {selectedStepData && (
                  <div className="board-value-bar" role="status" aria-live="polite">
                    <span className="board-value-label">Board value</span>
                    <span className="board-value-number">
                      {selectedStepData.board_value.toFixed(0)}
                    </span>
                  </div>
                )}
                <div className={`board-meta${movesDisagree ? ' has-disagreement' : ''}`}>
                  <div className="board-meta-entry">
                    <span className="meta-label">Teacher</span>
                    <span className="meta-value" role="text">
                      {hasTeacherMove ? (
                        <span className="meta-arrow" title={MOVE_LABELS[teacherMove!]}>
                          {MOVE_ICONS[teacherMove!]}
                        </span>
                      ) : (
                        '—'
                      )}
                    </span>
                  </div>
                  <div
                    className={`board-meta-entry student${movesDisagree ? ' disagree' : ''}`}
                  >
                    <span className="meta-label">Student</span>
                    <span className="meta-value" role="text">
                      {hasStudentMove ? (
                        <span className="meta-arrow" title={MOVE_LABELS[studentMove!]}>
                          {MOVE_ICONS[studentMove!]}
                        </span>
                      ) : (
                        '—'
                      )}
                    </span>
                  </div>
                </div>
              </div>

              <section className="insights-section">
                <h2 className="section-title">Teacher vs student</h2>
                <div className="insights-grid">
                  <MoveInsights
                    rows={rows}
                    formatPercent={formatPercent}
                    showTokenization={showTokenization}
                  />
                </div>
              </section>
            </section>
          </div>
        ) : (
          <div className="status-card">No data available.</div>
        )}
      </main>
    </div>
  )
}

export default App
