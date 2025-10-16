import { memo, type CSSProperties } from 'react'
import type { InsightRow, TokenCategory } from '../lib/insightRow'
import styles from './MoveInsights.module.css'

export interface MoveInsightsProps {
  rows: InsightRow[]
  formatPercent: (value: number | null | undefined) => string
  showTokenization?: boolean
}

type Severity = 'soft' | 'medium' | 'severe'

const severityRowClass: Record<Severity, string> = {
  soft: styles.studentDisagreeSoft,
  medium: styles.studentDisagreeMedium,
  severe: styles.studentDisagreeSevere,
}

const severityBadgeClass: Record<Severity, string> = {
  soft: styles.badgeSoft,
  medium: styles.badgeMedium,
  severe: styles.badgeSevere,
}

const badgeSymbol: Record<Severity, string> = {
  soft: '≈',
  medium: '!',
  severe: '‼',
}

const badgeLabel: Record<Severity, string> = {
  soft: 'Marginal toss-up miss',
  medium: 'Overconfident miss',
  severe: 'Severe overconfidence mismatch',
}

const cx = (...classNames: Array<string | false | null | undefined>) =>
  classNames.filter(Boolean).join(' ')

const tokenChipClass: Record<TokenCategory, string> = {
  illegal: styles.tokenChipIllegal,
  failure: styles.tokenChipFailure,
  winner: styles.tokenChipWinner,
  bin: styles.tokenChipBin,
}

interface StudentBinsSparklineProps {
  probs: number[]
  labels: string[] | null
  teacherBinIndex: number | null
  studentBinTopIndex: number | null
  summary: string | null
  formatPercent: (value: number | null | undefined) => string
}

const StudentBinsSparkline = memo(function StudentBinsSparkline({
  probs,
  labels,
  teacherBinIndex,
  studentBinTopIndex,
  summary,
  formatPercent,
}: StudentBinsSparklineProps) {
  // TODO: tighten validation once we revisit bin visualizations; for now assume `probs` is sane.
  if (!probs.length) {
    return <span className={styles.binPlaceholder}>—</span>
  }

  const total = probs.reduce((acc, value) => acc + Math.max(value, 0), 0)
  const max = probs.reduce((acc, value) => Math.max(acc, value), 0)

  return (
    <div className={styles.binColumn}>
      <div className={styles.binSpark} role="img" aria-label="Student probability per bin">
        {probs.map((prob, idx) => {
          const normalized = total > 0 ? prob / total : 0
          const intensity = max > 0 ? prob / max : 0
          const minHeight = 0.06
          const height = `${Math.max(normalized, minHeight) * 100}%`
          const backgroundAlpha = 0.18 + intensity * 0.62
          const defaultLabel = `Bin ${String(idx + 1).padStart(2, '0')}`
          const label = labels && idx < labels.length ? labels[idx] ?? defaultLabel : defaultLabel
          const labelLower = label.toLowerCase()
          let background = `rgba(233, 181, 92, ${backgroundAlpha.toFixed(3)})`
          if (labelLower === 'illegal') {
            background = 'rgba(63, 56, 47, 0.5)'
          } else if (labelLower === 'failure') {
            background = 'rgba(200, 64, 46, 0.55)'
          }
          const isTeacher = teacherBinIndex !== null && idx === teacherBinIndex
          const isArgmax = studentBinTopIndex !== null && idx === studentBinTopIndex
          const style: CSSProperties = { height }
          style.background = isTeacher ? 'rgba(125, 31, 43, 0.82)' : background
          const className = cx(
            styles.binSegment,
            isTeacher ? styles.binSegmentTeacher : null,
            isArgmax ? styles.binSegmentArgmax : null,
            labelLower === 'illegal' ? styles.binSegmentIllegal : null,
            labelLower === 'failure' ? styles.binSegmentFailure : null,
          )
          const tooltip = `${label}: ${formatPercent(total > 0 ? normalized : null)}`
          return <span key={idx} className={className} style={style} title={tooltip} />
        })}
      </div>
      {summary ? <span className={styles.binSummary}>{summary}</span> : null}
    </div>
  )
})

export const MoveInsights = memo(function MoveInsights({
  rows,
  formatPercent,
  showTokenization = false,
}: MoveInsightsProps) {
  const hasStudentBins = rows.some(
    (row) => Array.isArray(row.studentBins) && row.studentBins.length > 0,
  )
  const showSecondary = showTokenization || hasStudentBins;

  return (
    <div className={styles.table}>
      <div className={styles.header}>
        <span>Move</span>
        <span className={styles.mono}>Teacher EV</span>
        <span className={styles.mono}>Advantage (Teacher)</span>
        <span className={styles.mono}>Student probs</span>
      </div>
      {rows.map((row) => {
        const isStudentDisagree = row.isStudent && !row.isTeacher

        let severity: Severity | null = null
        if (isStudentDisagree) {
          const rawEvLoss = row.advantage !== null ? Math.abs(row.advantage) : 0
          const scaledEvLoss =
            row.teacher_ev_mode === 'relative' ? rawEvLoss : rawEvLoss * 1000
          const studentProb = row.probability ?? 0

          if (
            studentProb >= 0.9 ||
            (studentProb >= 0.8 && scaledEvLoss >= 200) ||
            scaledEvLoss >= 800
          ) {
            severity = 'severe'
          } else if (scaledEvLoss <= 60 && studentProb <= 0.45) {
            severity = 'soft'
          } else {
            severity = 'medium'
          }
        }

        const rowClassName = cx(
          styles.row,
          row.legal ? null : styles.disabled,
          row.isTeacher ? styles.teacher : null,
          row.isStudent ? styles.student : null,
          isStudentDisagree ? styles.studentDisagree : null,
          severity ? severityRowClass[severity] : null,
        )

        const badgeClassName = severity ? cx(styles.badge, severityBadgeClass[severity]) : null

        return (
          <div key={row.label} className={rowClassName}>
            <div className={styles.rowPrimary}>
              <span className={styles.move}>
                <span className={styles.moveIcon}>{row.icon}</span>
                {severity ? (
                  <span
                    className={badgeClassName ?? undefined}
                    title={badgeLabel[severity]}
                  aria-label={badgeLabel[severity]}
                  role="img"
                >
                  {badgeSymbol[severity]}
                </span>
              ) : null}
              {row.label}
            </span>
              <span className={cx(styles.value, styles.mono)}>{row.teacher_ev_display}</span>
              <span className={cx(styles.bar, styles.mono)}>
                <span className={styles.advBar} style={{ '--fill': row.advantage_normalized ?? 0 } as CSSProperties} />
                <span className={styles.delta}>
                  {row.advantage_display}
                </span>
              </span>
              <span className={styles.probSection}>
                <div className={styles.probRow}>
                  <span className={styles.probLabel}>π₁</span>
                  <span className={cx(styles.bar, styles.mono, styles.probBarWrapper)}>
                    <span
                      className={styles.probBar}
                      style={{ '--fill': row.probabilityNormalized ?? 0 } as CSSProperties}
                    />
                  </span>
                  <span className={styles.probability}>{formatPercent(row.probability)}</span>
                </div>
                <div className={styles.probRow}>
                  <span className={styles.probLabel}>log</span>
                  <span className={styles.logValue}>
                    {row.probFromLogp === null ? '—' : formatPercent(row.probFromLogp)}
                  </span>
                </div>
              </span>
            </div>
            {showSecondary ? (
              <div className={styles.rowSecondary}>
                {showTokenization ? (
                  <div className={styles.secondarySection}>
                    <span className={styles.secondaryLabel}>Tokenizer</span>
                    <div className={styles.tokenCell}>
                      {row.tokenLabel ? (
                        <>
                          <span
                            className={cx(
                              styles.tokenChip,
                              row.tokenCategory ? tokenChipClass[row.tokenCategory] : null,
                            )}
                          >
                            {row.tokenLabel}
                          </span>
                          {row.tokenBinDisplay || row.advantageIdDisplay ? (
                            <span className={styles.tokenMeta}>
                              {row.tokenBinDisplay ? (
                                <span className={styles.tokenMetaRow}>{row.tokenBinDisplay}</span>
                              ) : null}
                              {row.advantageIdDisplay ? (
                                <span className={styles.tokenMetaRow}>
                                  Δᵢ {row.advantageIdDisplay}
                                </span>
                              ) : null}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className={styles.tokenMeta}>—</span>
                      )}
                    </div>
                  </div>
                ) : null}
                {hasStudentBins ? (
                  <div className={styles.secondarySection}>
                    <span className={styles.secondaryLabel}>Student bins</span>
                    <div className={styles.binCell}>
                      {row.studentBins ? (
                        <StudentBinsSparkline
                          probs={row.studentBins}
                          labels={row.studentBinLabels}
                          teacherBinIndex={row.teacherBinIndex}
                          studentBinTopIndex={row.studentBinTopIndex}
                          summary={row.studentBinSummary}
                          formatPercent={formatPercent}
                        />
                      ) : (
                        <span className={styles.binPlaceholder}>—</span>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
})
