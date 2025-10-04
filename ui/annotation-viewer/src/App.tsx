import { useEffect, useMemo, useState } from 'react'
import {
  useRunDetailQuery,
  useRunsQuery,
} from './lib/api/queries'
import type { StepResponse } from './lib/api/schemas'
import './App.css'

const MOVE_NAMES = ['Up', 'Down', 'Left', 'Right']
const RUNS_PAGE_SIZE = 25
const STEP_SAMPLE_LIMIT = 128

const formatMove = (move: number) => {
  if (move < 0 || move >= MOVE_NAMES.length) {
    return '—'
  }
  return MOVE_NAMES[move]
}

const decodeLegalMask = (mask: number) =>
  MOVE_NAMES.filter((_, idx) => (mask & (1 << idx)) !== 0)

const formatTile = (exponent: number) => {
  if (exponent === 0) {
    return ''
  }
  return (2 ** exponent).toLocaleString()
}

const findRepresentativeStep = (steps: StepResponse[] | undefined) => {
  if (!steps || steps.length === 0) {
    return undefined
  }
  return steps[0]
}

const formatProbability = (prob?: number) => {
  if (prob === undefined) return '—'
  return `${(prob * 100).toFixed(1)}%`
}

function App() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const runsQuery = useRunsQuery({ pageSize: RUNS_PAGE_SIZE })

  useEffect(() => {
    if (!runsQuery.isSuccess) {
      return
    }
    if (selectedRunId === null) {
      const first = runsQuery.data.runs[0]
      if (first) {
        setSelectedRunId(first.run_id)
      }
    }
  }, [runsQuery.isSuccess, runsQuery.data, selectedRunId])

  const runDetailQuery = useRunDetailQuery(selectedRunId, {
    limit: STEP_SAMPLE_LIMIT,
  })

  const representativeStep = useMemo(
    () => findRepresentativeStep(runDetailQuery.data?.steps),
    [runDetailQuery.data?.steps],
  )

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="app-title">Board Annotation Viewer</h1>
      </header>
      <div className="app-body">
        <aside className="sidebar">
          <h2 className="sidebar-title">Runs</h2>
          {runsQuery.isLoading ? (
            <div className="status-card">Loading runs…</div>
          ) : runsQuery.isError ? (
            <div className="status-card status-error">
              Unable to load runs. Please verify the annotation server is running.
            </div>
          ) : runsQuery.data?.runs.length ? (
            <div className="run-list">
              {runsQuery.data.runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  className={`run-button${
                    run.run_id === selectedRunId ? ' active' : ''
                  }`}
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <span className="run-button-title">Run {run.run_id}</span>
                  <span className="run-button-meta">
                    <span>Score {run.max_score.toLocaleString()}</span>
                    <span>Steps {run.steps.toLocaleString()}</span>
                  </span>
                  <span className="run-button-meta">
                    <span>Highest Tile {run.highest_tile}</span>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="status-card">No runs available.</div>
          )}
        </aside>
        <main className="content">
          {selectedRunId === null ? (
            <div className="status-card">Select a run to begin.</div>
          ) : runDetailQuery.isLoading ? (
            <div className="status-card">Loading run {selectedRunId}…</div>
          ) : runDetailQuery.isError ? (
            <div className="status-card status-error">
              Unable to load run details. Try selecting another run.
            </div>
          ) : runDetailQuery.data ? (
            <>
              <section className="section">
                <h2 className="section-title">Run Summary</h2>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Run ID</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.run_id}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Seed</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.seed.toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Steps</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.steps.toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Max Score</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.max_score.toLocaleString()}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Highest Tile</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.highest_tile}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Policy Mask</span>
                    <span className="summary-value">
                      {runDetailQuery.data.run.policy_kind_mask}
                    </span>
                  </div>
                </div>
              </section>

              <section className="section board-preview">
                <h2 className="section-title">Step Preview</h2>
                {representativeStep ? (
                  <>
                    <div className="summary-grid">
                      <div className="summary-item">
                        <span className="summary-label">Step</span>
                        <span className="summary-value">
                          {representativeStep.step_index.toLocaleString()}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Legal Moves</span>
                        <span className="summary-value">
                          {decodeLegalMask(representativeStep.legal_mask).join(', ') ||
                            'None'}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Teacher Move</span>
                        <span className="summary-value">
                          {representativeStep.teacher_move === 255
                            ? 'Unknown'
                            : formatMove(representativeStep.teacher_move)}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Argmax Move</span>
                        <span className="summary-value">
                          {representativeStep.annotation
                            ? formatMove(representativeStep.annotation.argmax_head)
                            : '—'}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Argmax Prob</span>
                        <span className="summary-value">
                          {representativeStep.annotation
                            ? formatProbability(
                                representativeStep.annotation.argmax_prob,
                              )
                            : '—'}
                        </span>
                      </div>
                    </div>
                    <div className="board-grid">
                      {representativeStep.board.map((value, idx) => (
                        <div
                          key={idx}
                          className={`board-cell${value === 0 ? ' empty' : ''}`}
                        >
                          {formatTile(value)}
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="status-card">No steps loaded for this run.</div>
                )}
              </section>
            </>
          ) : (
            <div className="status-card">No data available.</div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
