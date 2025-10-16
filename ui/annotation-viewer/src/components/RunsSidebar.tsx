import type { ChangeEvent } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import type { RunsResponse } from '../lib/api/schemas'
import { ApiError } from '../lib/api/errors'

interface RunsSidebarProps {
  runsQuery: UseQueryResult<RunsResponse, ApiError>
  selectedRunId: number | null
  onRunSelect: (runId: number) => void
  collapsed: boolean
  onToggle: () => void
  page: number
  pageSize: number
  totalRuns: number
  pageSizeOptions: number[]
  onPageChange: (page: number) => void
  onPageSizeChange: (pageSize: number) => void
  isFetchingPage: boolean
}

const formatDisagreementPercent = (value: number | undefined) => {
  if (value === undefined || Number.isNaN(value)) {
    return '—'
  }
  return `${(value * 100).toFixed(1)}%`
}

export function RunsSidebar({
  runsQuery,
  selectedRunId,
  onRunSelect,
  collapsed,
  onToggle,
  page,
  pageSize,
  totalRuns,
  pageSizeOptions,
  onPageChange,
  onPageSizeChange,
  isFetchingPage,
}: RunsSidebarProps) {
  const runs = runsQuery.data?.runs ?? []
  const hasRuns = runs.length > 0
  const safePage = Math.max(page, 1)
  const safePageSize = Math.max(pageSize, 1)
  const effectiveTotal = totalRuns > 0 ? totalRuns : runs.length
  const showingFrom = hasRuns
    ? Math.min((safePage - 1) * safePageSize + 1, effectiveTotal || runs.length)
    : 0
  const showingTo = hasRuns
    ? Math.min(showingFrom + runs.length - 1, effectiveTotal || runs.length)
    : 0
  const totalPages =
    effectiveTotal === 0 ? 1 : Math.max(1, Math.ceil(effectiveTotal / safePageSize))
  const canGoPrev = safePage > 1
  const canGoNext = safePage < totalPages
  const disablePrev = !canGoPrev || runsQuery.isLoading || runsQuery.isError || isFetchingPage
  const disableNext = !canGoNext || runsQuery.isLoading || runsQuery.isError || isFetchingPage
  const rangeLabel =
    effectiveTotal === 0
      ? '0 runs'
      : hasRuns
        ? `${showingFrom.toLocaleString()}–${showingTo.toLocaleString()} of ${effectiveTotal.toLocaleString()}`
        : `0 of ${effectiveTotal.toLocaleString()}`
  const pageLabel =
    effectiveTotal === 0
      ? 'Page 1 of 1'
      : `Page ${safePage.toLocaleString()} of ${totalPages.toLocaleString()}`

  const handlePageSizeChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextSize = Number.parseInt(event.target.value, 10)
    if (Number.isFinite(nextSize)) {
      onPageSizeChange(nextSize)
    }
  }

  return (
    <aside className={`sidebar${collapsed ? ' collapsed' : ''}`}>
      <div className="sidebar-header">
        {!collapsed && <h2 className="sidebar-title">Runs</h2>}
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="sidebar-brand">
            <h1 className="brand-title">Board Annotation</h1>
            <p className="brand-subtitle">Compare teacher trajectories with model policy.</p>
          </div>
          <div className="runs-section">
            <div className="sidebar-header">
              <h2 className="sidebar-title">Runs</h2>
              <span className="sidebar-count">{rangeLabel}</span>
            </div>
            {runsQuery.isLoading ? (
              <div className="status-card">Loading runs…</div>
            ) : runsQuery.isError ? (
              <div className="status-card status-error">
                Unable to load runs. Verify the annotation server is reachable.
              </div>
            ) : hasRuns ? (
              <div className="run-list-wrapper">
                <div className="run-list" aria-busy={isFetchingPage}>
                  {runs.map((run) => (
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
              <div className="run-pagination">
                <div className="run-pagination-controls">
                  <label className="run-pagination-size">
                    <span className="run-pagination-label">Per page</span>
                    <select
                      className="run-pagination-select"
                      value={safePageSize}
                      onChange={handlePageSizeChange}
                    >
                      {pageSizeOptions.map((size) => (
                        <option key={size} value={size}>
                          {size}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="run-pagination-nav" role="group" aria-label="Runs pagination">
                    <button
                      type="button"
                      className="run-pagination-button"
                      onClick={() => onPageChange(safePage - 1)}
                      disabled={disablePrev}
                      aria-label="Go to previous page"
                    >
                      ‹
                    </button>
                    <span className="run-pagination-page">{pageLabel}</span>
                    <button
                      type="button"
                      className="run-pagination-button"
                      onClick={() => onPageChange(safePage + 1)}
                      disabled={disableNext}
                      aria-label="Go to next page"
                    >
                      ›
                    </button>
                  </div>
                </div>
                {isFetchingPage && (
                  <span className="run-pagination-status" role="status">
                    Updating…
                  </span>
                )}
              </div>
              </div>
            ) : (
              <div className="status-card">No runs available.</div>
            )}
          </div>
        </>
      )}
    </aside>
  )
}
