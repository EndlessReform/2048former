import { z } from 'zod'

export const policyKindLegendSchema = z.object({
  policy_p1: z.number().int(),
  policy_logprobs: z.number().int(),
  policy_hard: z.number().int(),
  policy_student_bins: z.number().int(),
})

export type PolicyKindLegend = z.infer<typeof policyKindLegendSchema>

export const runSummarySchema = z.object({
  run_id: z.number().int(),
  seed: z.number().int().nonnegative(),
  steps: z.number().int().nonnegative(),
  max_score: z.number().int().nonnegative(),
  highest_tile: z.number().int().nonnegative(),
  policy_kind_mask: z.number().int().nonnegative(),
  disagreement_count: z.number().int().nonnegative(),
  disagreement_percentage: z.number(),
})

export type RunSummary = z.infer<typeof runSummarySchema>

export const runsResponseSchema = z.object({
  total: z.number().int().nonnegative(),
  page: z.number().int().min(1),
  page_size: z.number().int().min(1),
  runs: z.array(runSummarySchema),
  policy_kind_legend: policyKindLegendSchema,
})

export type RunsResponse = z.infer<typeof runsResponseSchema>

export const annotationPayloadSchema = z.object({
  policy_kind_mask: z.number().int().nonnegative(),
  argmax_head: z.number().int().nonnegative(),
  argmax_prob: z.number(),
  policy_p1: z.array(z.number()).length(4),
  policy_logp: z.array(z.number()).length(4),
  policy_hard: z.array(z.number()).length(4),
})

export type AnnotationPayload = z.infer<typeof annotationPayloadSchema>

export const stepResponseSchema = z.object({
  step_index: z.number().int().nonnegative(),
  board: z.array(z.number().int().nonnegative()).length(16),
  board_value: z.number(),
  branch_evs: z.array(z.number().nullable()),
  relative_branch_evs: z.array(z.number().int().nullable()),
  advantage_branch: z.array(z.number().int().nullable()),
  legal_mask: z.number().int().nonnegative(),
  teacher_move: z.number().int().nonnegative(),
  is_disagreement: z.boolean(),
  valuation_type: z.string(),
  annotation: annotationPayloadSchema.optional(),
  tokens: z.array(z.number().int()).optional(),
})

export type StepResponse = z.infer<typeof stepResponseSchema>

export const paginationSchema = z.object({
  offset: z.number().int().nonnegative(),
  limit: z.number().int().nonnegative(),
  total: z.number().int().nonnegative(),
})

export type Pagination = z.infer<typeof paginationSchema>

export const runDetailResponseSchema = z.object({
  run: runSummarySchema,
  pagination: paginationSchema,
  steps: z.array(stepResponseSchema),
  policy_kind_legend: policyKindLegendSchema,
})

export type RunDetailResponse = z.infer<typeof runDetailResponseSchema>

export const disagreementsResponseSchema = z.object({
  disagreements: z.array(z.number().int().nonnegative()),
  total: z.number().int().nonnegative(),
})

export type DisagreementsResponse = z.infer<typeof disagreementsResponseSchema>

export const healthTokenizerInfoSchema = z.object({
  tokenizer_type: z.string(),
  num_bins: z.number().int().nonnegative(),
  vocab_order: z.array(z.string()),
  valuation_types: z.array(z.string()),
})

export type HealthTokenizerInfo = z.infer<typeof healthTokenizerInfoSchema>

export const healthResponseSchema = z.object({
  status: z.string(),
  tokenizer: healthTokenizerInfoSchema.nullable().optional(),
})

export type HealthResponse = z.infer<typeof healthResponseSchema>
