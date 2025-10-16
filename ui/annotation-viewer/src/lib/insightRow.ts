export type TeacherEvMode = 'relative' | 'raw'
export type TokenCategory = 'illegal' | 'failure' | 'winner' | 'bin'

export interface InsightRow {
  label: string
  icon: string
  teacher_ev: number | null
  teacher_ev_normalized: number | null
  teacher_ev_mode: TeacherEvMode
  teacher_ev_display: string
  advantage: number | null
  advantage_display: string
  advantage_normalized: number | null
  probability: number | null
  probabilityNormalized: number | null
  probFromLogp: number | null
  legal: boolean
  isTeacher: boolean
  isStudent: boolean
  hasP1: boolean
  hasLogp: boolean
  tokenId: number | null
  tokenLabel: string | null
  tokenCategory: TokenCategory | null
  tokenBinDisplay: string | null
  advantageId: number | null
  advantageIdDisplay: string | null
  studentBins: number[] | null
  studentBinTopIndex: number | null
  teacherBinIndex: number | null
  studentBinSummary: string | null
  studentBinLabels: string[] | null
}
