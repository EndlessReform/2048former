import { useEffect, useMemo, useState, type ChangeEvent, type CSSProperties } from 'react'
import { useRunDetailQuery, useRunsQuery } from './lib/api/queries'
import type { StepResponse } from './lib/api/schemas'
import './App.css'

const MOVE_LABELS = ['Up', 'Down', 'Left', 'Right']
const MOVE_ICONS = ['↑', '↓', '←', '→']
const RUNS_PAGE_SIZE = 25
const WINDOW_RADIUS = 128
const WINDOW_SIZE = WINDOW_RADIUS * 2 + 1

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

const branchRows = (step: StepResponse | undefined) => {
  if (!step) return []
  const teacherEvs = step.branch_evs.map((value) => (value === null ? null : value))
  const normalizedTeacher = normalize(teacherEvs)
  const validTeacherEvs = teacherEvs.filter((value): value is number => value !== null)
  const teacherBest = validTeacherEvs.length ? Math.max(...validTeacherEvs) : null

  const student = step.annotation
  const studentProb = student ? student.policy_p1 : null
  const studentLogp = student ? student.policy_logp : null
  const studentProbNormalized: Array<number | null> = student
    ? normalize(student.policy_p1.map((value) => value))
    : [null, null, null, null]
  const annotationMask = student?.policy_kind_mask ?? 0

  return MOVE_LABELS.map((label, idx) => {
    const legal = (step.legal_mask & (1 << idx)) !== 0
    const ev = teacherEvs[idx]
    const evDelta = ev === null || teacherBest === null ? null : ev - teacherBest
    const probability = studentProb ? studentProb[idx] : null
    const probExp = studentLogp ? Math.exp(studentLogp[idx]) : null
    return {
      label,
      icon: MOVE_ICONS[idx],
      ev,
      evNormalized: normalizedTeacher[idx],
      evDelta,
      probability,
      probabilityNormalized: studentProbNormalized[idx],
      probFromLogp: probExp,
      legal,
      isTeacher: step.teacher_move === idx,
      isStudent: student?.argmax_head === idx,
      hasP1: Boolean(annotationMask & 1),
      hasLogp: Boolean(annotationMask & 2),
    }
  })
}

function App() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [selectedStep, setSelectedStep] = useState(0)
  const [windowOffset, setWindowOffset] = useState(0)

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

  const runDetailQuery = useRunDetailQuery(selectedRunId, { offset: windowOffset, limit: WINDOW_SIZE }, {
    placeholderData: (previous) => previous,
  })

  const steps = runDetailQuery.data?.steps ?? []
  const pagination = runDetailQuery.data?.pagination
  const totalSteps = pagination?.total ?? 0

  useEffect(() => {
    if (!totalSteps) return
    if (selectedStep > totalSteps - 1) {
      setSelectedStep(totalSteps - 1)
    }
  }, [totalSteps, selectedStep])

  useEffect(() => {
    if (!totalSteps) return
    const maxOffset = Math.max(totalSteps - WINDOW_SIZE, 0)
    const targetOffset = Math.min(Math.max(selectedStep - WINDOW_RADIUS, 0), maxOffset)
    if (targetOffset !== windowOffset) {
      setWindowOffset(targetOffset)
    }
  }, [selectedStep, totalSteps, windowOffset])

  useEffect(() => {
    if (!steps.length) return
    const found = steps.find((step) => step.step_index === selectedStep)
    if (!found) {
      const fallback = steps[Math.min(steps.length - 1, selectedStep - windowOffset)] ?? steps[0]
      if (fallback && fallback.step_index !== selectedStep) {
        setSelectedStep(fallback.step_index)
      }
    }
  }, [steps, selectedStep, windowOffset])

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
  const upcoming = selectedIndex >= 0 ? steps.slice(selectedIndex + 1, selectedIndex + 4) : []
  const rows = useMemo(() => branchRows(selectedStepData), [selectedStepData])

  const teacherMove = selectedStepData?.teacher_move ?? null
  const studentMove = selectedStepData?.annotation?.argmax_head ?? null
  const hasTeacherMove = teacherMove !== null && teacherMove !== 255
  const hasStudentMove = studentMove !== null && studentMove !== undefined
  const movesDisagree = hasTeacherMove && hasStudentMove && teacherMove !== studentMove

  const handleStepDelta = (delta: number) => {
    if (!totalSteps) return
    setSelectedStep((current) => {
      const next = Math.min(Math.max(current + delta, 0), Math.max(totalSteps - 1, 0))
      return next
    })
  }

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

  const handleSelectStep = (stepIndex: number) => {
    setSelectedStep(stepIndex)
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1 className="brand-title">Board Annotation</h1>
          <p className="brand-subtitle">Compare teacher trajectories with model policy.</p>
        </div>
        <div className="sidebar-header">
          <h2 className="sidebar-title">Runs</h2>
          <span className="sidebar-count">
            {runsQuery.data?.total ? `${runsQuery.data.total} loaded` : ''}
          </span>
        </div>
        {runsQuery.isLoading ? (
          <div className="status-card">Loading runs…</div>
        ) : runsQuery.isError ? (
          <div className="status-card status-error">
            Unable to load runs. Verify the annotation server is reachable.
          </div>
        ) : runsQuery.data?.runs.length ? (
          <div className="run-list">
            {runsQuery.data.runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                className={`run-button${run.run_id === selectedRunId ? ' active' : ''}`}
                onClick={() => handleRunSelect(run.run_id)}
              >
                <span className="run-button-title">Run {run.run_id}</span>
                <span className="run-button-meta">
                  <span className="meta-label">Score</span>
                  <span>{run.max_score.toLocaleString()}</span>
                </span>
                <span className="run-button-meta">
                  <span className="meta-label">Steps</span>
                  <span>{run.steps.toLocaleString()}</span>
                </span>
                <span className="run-button-meta">
                  <span className="meta-label">Highest Tile</span>
                  <span>{run.highest_tile}</span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <div className="status-card">No runs available.</div>
        )}
      </aside>
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
            </div>

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
              <input
                className="scrubber-slider"
                type="range"
                min={0}
                max={sliderMax}
                value={sliderValue}
                onChange={handleSliderChange}
                disabled={totalSteps <= 1}
              />
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
                {upcoming.length > 0 && (
                  <div className="board-preview-rail">
                    <h3 className="preview-title">Upcoming Boards</h3>
                    <div className="preview-strip">
                      {upcoming.map((step) => (
                        <button
                          key={step.step_index}
                          type="button"
                          className="preview-cell"
                          onClick={() => handleSelectStep(step.step_index)}
                        >
                          <span className="preview-meta">#{step.step_index}</span>
                          <div className="preview-board">
                            {step.board.map((value, idx) => (
                              <span
                                key={idx}
                                className={`preview-tile${value === 0 ? ' empty' : ''}`}
                                style={tileStyle(value)}
                              />
                            ))}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <section className="insights-section">
                <h2 className="section-title">Teacher vs student</h2>
                <div className="insights-grid">
                  <div className="insight-table">
                    <div className="insight-header">
                      <span className="insight-header-label">Move</span>
                      <span className="insight-header-label mono">Teacher EV</span>
                      <span className="insight-header-label mono">Advantage (Teacher)</span>
                      <span className="insight-header-label mono">Student π₁</span>
                      <span className="insight-header-label mono">Student prob</span>
                    </div>
                    {rows.map((row) => (
                      <div
                        key={row.label}
                        className={[
                          'insight-row',
                          row.legal ? '' : 'disabled',
                          row.isTeacher ? 'teacher' : '',
                          row.isStudent ? 'student' : '',
                          row.isStudent && !row.isTeacher ? 'student-disagree' : '',
                        ]
                          .filter(Boolean)
                          .join(' ')}
                      >
                        <span className="insight-move">
                          <span className="move-icon">{row.icon}</span>
                          {row.label}
                        </span>
                        <span className="insight-value mono">
                          {row.ev === null ? '—' : row.ev.toFixed(3)}
                        </span>
                        <span className="insight-bar mono">
                          <span
                            className="adv-bar"
                            style={{ '--fill': row.evNormalized ?? 0 } as CSSProperties}
                          />
                          <span className="adv-delta">
                            {row.evDelta === null
                              ? '—'
                              : row.evDelta === 0
                              ? '+0.000'
                              : row.evDelta.toFixed(3)}
                          </span>
                        </span>
                        <span className="insight-bar mono">
                          <span
                            className="prob-bar"
                            style={{ '--fill': row.probabilityNormalized ?? 0 } as CSSProperties}
                          />
                          <span className="prob-value">{formatPercent(row.probability)}</span>
                        </span>
                        <span className="insight-value mono">
                          {row.probFromLogp === null ? '—' : formatPercent(row.probFromLogp)}
                        </span>
                      </div>
                    ))}
                  </div>
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
