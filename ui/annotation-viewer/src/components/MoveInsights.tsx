import { memo, type CSSProperties } from 'react'
import type { InsightRow } from '../lib/insightRow'
import styles from './MoveInsights.module.css'

export interface MoveInsightsProps {
  rows: InsightRow[]
  formatPercent: (value: number | null | undefined) => string
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

export const MoveInsights = memo(function MoveInsights({ rows, formatPercent }: MoveInsightsProps) {
  return (
    <div className={styles.table}>
      <div className={styles.header}>
        <span>Move</span>
        <span className={styles.mono}>Teacher EV</span>
        <span className={styles.mono}>Advantage (Teacher)</span>
        <span className={styles.mono}>Student π₁</span>
        <span className={styles.mono}>Student prob</span>
      </div>
      {rows.map((row) => {
        const isStudentDisagree = row.isStudent && !row.isTeacher

        let severity: Severity | null = null
        if (isStudentDisagree) {
          const evLoss = row.evDelta !== null ? Math.abs(row.evDelta) : 0
          const studentProb = row.probability ?? 0

          if (studentProb >= 0.9 || (studentProb >= 0.8 && evLoss >= 0.02) || evLoss >= 0.08) {
            severity = 'severe'
          } else if (evLoss <= 0.012 && studentProb <= 0.45) {
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
            <span className={cx(styles.value, styles.mono)}>
              {row.ev === null ? '—' : row.ev.toFixed(3)}
            </span>
            <span className={cx(styles.bar, styles.mono)}>
              <span className={styles.advBar} style={{ '--fill': row.evNormalized ?? 0 } as CSSProperties} />
              <span className={styles.delta}>
                {row.evDelta === null
                  ? '—'
                  : row.evDelta === 0
                  ? '+0.000'
                  : row.evDelta.toFixed(3)}
              </span>
            </span>
            <span className={cx(styles.bar, styles.mono)}>
              <span className={styles.probBar} style={{ '--fill': row.probabilityNormalized ?? 0 } as CSSProperties} />
              <span className={styles.probability}>{formatPercent(row.probability)}</span>
            </span>
            <span className={cx(styles.value, styles.mono)}>
              {row.probFromLogp === null ? '—' : formatPercent(row.probFromLogp)}
            </span>
          </div>
        )
      })}
    </div>
  )
})
