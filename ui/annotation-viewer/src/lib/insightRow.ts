export type TeacherEvMode = 'relative' | 'raw'

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
}
