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

export const MoveInsights = memo(function MoveInsights({
  rows,
  formatPercent,
  showTokenization = false,
}: MoveInsightsProps) {
  return (
    <div className={styles.table}>
      <div className={cx(styles.header, showTokenization ? styles.headerTokenized : null)}>
        <span>Move</span>
        <span className={styles.mono}>Teacher EV</span>
        <span className={styles.mono}>Advantage (Teacher)</span>
        {showTokenization ? <span className={styles.mono}>Tokenizer</span> : null}
        <span className={styles.mono}>Student π₁</span>
        <span className={styles.mono}>Student prob</span>
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
          showTokenization ? styles.rowTokenized : null,
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
            <span className={cx(styles.value, styles.mono)}>{row.teacher_ev_display}</span>
            <span className={cx(styles.bar, styles.mono)}>
              <span className={styles.advBar} style={{ '--fill': row.advantage_normalized ?? 0 } as CSSProperties} />
              <span className={styles.delta}>
                {row.advantage_display}
              </span>
            </span>
            {showTokenization ? (
              <span className={styles.tokenCell}>
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
              </span>
            ) : null}
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
