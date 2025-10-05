export interface InsightRow {
  label: string
  icon: string
  ev: number | null
  evNormalized: number | null
  evDelta: number | null
  probability: number | null
  probabilityNormalized: number | null
  probFromLogp: number | null
  legal: boolean
  isTeacher: boolean
  isStudent: boolean
  hasP1: boolean
  hasLogp: boolean
}
