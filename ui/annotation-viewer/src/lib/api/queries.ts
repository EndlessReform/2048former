import { useQuery } from '@tanstack/react-query'
import { fetchJson } from './client'
import { runDetailResponseSchema, runsResponseSchema } from './schemas'
import type { RunDetailResponse, RunsResponse } from './schemas'
import type { UseQueryOptions } from '@tanstack/react-query'
import { ApiError } from './errors'

export type RunsSortOption =
  | 'score_asc'
  | 'score_desc'
  | 'tile_desc'
  | 'tile_asc'
  | 'steps_desc'
  | 'steps_asc'

export type RunsQueryParams = {
  page?: number
  pageSize?: number
  minScore?: number
  maxScore?: number
  minHighestTile?: number
  maxHighestTile?: number
  minSteps?: number
  maxSteps?: number
  sort?: RunsSortOption
}

export const buildRunsSearchParams = (params: RunsQueryParams): URLSearchParams => {
  const search = new URLSearchParams()
  const {
    page,
    pageSize,
    minScore,
    maxScore,
    minHighestTile,
    maxHighestTile,
    minSteps,
    maxSteps,
    sort,
  } = params

  if (page !== undefined) search.set('page', String(page))
  if (pageSize !== undefined) search.set('page_size', String(pageSize))
  if (minScore !== undefined) search.set('min_score', String(minScore))
  if (maxScore !== undefined) search.set('max_score', String(maxScore))
  if (minHighestTile !== undefined) search.set('min_highest_tile', String(minHighestTile))
  if (maxHighestTile !== undefined) search.set('max_highest_tile', String(maxHighestTile))
  if (minSteps !== undefined) search.set('min_steps', String(minSteps))
  if (maxSteps !== undefined) search.set('max_steps', String(maxSteps))
  if (sort) search.set('sort', sort)

  return search
}

const fetchRuns = async (params: RunsQueryParams): Promise<RunsResponse> => {
  const searchParams = buildRunsSearchParams(params)
  return fetchJson<RunsResponse>({
    path: '/runs',
    schema: runsResponseSchema,
    searchParams,
  })
}

type RunsQueryOptions = Omit<
  UseQueryOptions<RunsResponse, ApiError, RunsResponse, [string, string]>,
  'queryKey' | 'queryFn'
>

export const useRunsQuery = (
  params: RunsQueryParams = {},
  options?: RunsQueryOptions,
) => {
  const searchParams = buildRunsSearchParams(params)
  const queryKey: [string, string] = ['runs', searchParams.toString()]

  return useQuery({
    queryKey,
    queryFn: () => fetchRuns(params),
    staleTime: 60_000,
    ...options,
  })
}

export type RunDetailQueryParams = {
  offset?: number
  limit?: number
}

export const buildRunDetailSearchParams = (
  params: RunDetailQueryParams,
): URLSearchParams => {
  const search = new URLSearchParams()
  const { offset, limit } = params
  if (offset !== undefined) search.set('offset', String(offset))
  if (limit !== undefined) search.set('limit', String(limit))
  return search
}

const fetchRunDetail = async (
  runId: number,
  params: RunDetailQueryParams,
): Promise<RunDetailResponse> => {
  const searchParams = buildRunDetailSearchParams(params)
  return fetchJson<RunDetailResponse>({
    path: `/runs/${runId}`,
    schema: runDetailResponseSchema,
    searchParams,
  })
}

type RunDetailQueryOptions = Omit<
  UseQueryOptions<RunDetailResponse, ApiError, RunDetailResponse, [string, number, string]>,
  'queryKey' | 'queryFn'
>

export const useRunDetailQuery = (
  runId: number | null,
  params: RunDetailQueryParams = {},
  options?: RunDetailQueryOptions,
) => {
  const searchParams = buildRunDetailSearchParams(params)
  const queryKey: [string, number, string] = [
    'runDetail',
    runId ?? -1,
    searchParams.toString(),
  ]

  const { enabled, ...rest } = options ?? {}
  const isEnabled = runId !== null && (enabled ?? true)

  return useQuery({
    queryKey,
    queryFn: () => {
      if (runId === null) {
        throw new ApiError('runId is required to fetch run details', {
          status: 400,
          url: '/runs/:run_id',
        })
      }
      return fetchRunDetail(runId, params)
    },
    enabled: isEnabled,
    staleTime: 30_000,
    ...rest,
  })
}
