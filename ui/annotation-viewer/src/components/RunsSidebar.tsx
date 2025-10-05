import type { UseQueryResult } from '@tanstack/react-query'
import type { RunsResponse } from '../lib/api/schemas'
import { ApiError } from '../lib/api/errors'

interface RunsSidebarProps {
  runsQuery: UseQueryResult<RunsResponse, ApiError>
  selectedRunId: number | null
  onRunSelect: (runId: number) => void
}

const formatDisagreementPercent = (value: number | undefined) => {
  if (value === undefined || Number.isNaN(value)) {
    return '—'
  }
  return `${(value * 100).toFixed(1)}%`
}

export function RunsSidebar({ runsQuery, selectedRunId, onRunSelect }: RunsSidebarProps) {
  return (
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
              onClick={() => onRunSelect(run.run_id)}
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
              <span className="run-button-meta">
                <span className="meta-label">Disagreements</span>
                <span>{formatDisagreementPercent(run.disagreement_percentage)}</span>
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div className="status-card">No runs available.</div>
      )}
    </aside>
  )
}
