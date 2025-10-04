import type { ZodType } from 'zod'
import { ApiError } from './errors'

const RAW_BASE = import.meta.env.VITE_ANNOTATION_API_BASE ?? ''
const API_BASE = typeof RAW_BASE === 'string' ? RAW_BASE.replace(/\/+$/, '') : ''

const HTTP_SCHEME_REGEX = /^https?:\/\//i

const hasSearchParams = (search?: URLSearchParams) =>
  Boolean(search && Array.from(search.entries()).length > 0)

const buildUrl = (path: string, search?: URLSearchParams) => {
  if (HTTP_SCHEME_REGEX.test(path)) {
    const url = new URL(path)
    if (hasSearchParams(search)) {
      url.search = search!.toString()
    }
    return url.toString()
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  if (HTTP_SCHEME_REGEX.test(API_BASE)) {
    const baseUrl = API_BASE.endsWith('/') ? API_BASE : `${API_BASE}/`
    const url = new URL(normalizedPath.replace(/^\//, ''), baseUrl)
    if (hasSearchParams(search)) {
      url.search = search!.toString()
    }
    return url.toString()
  }

  const basePrefix = API_BASE ? API_BASE : ''
  const url = `${basePrefix}${normalizedPath}`
  if (hasSearchParams(search)) {
    return `${url}?${search!.toString()}`
  }
  return url
}

export const fetchJson = async <T>(args: {
  path: string
  schema: ZodType<T>
  init?: RequestInit
  searchParams?: URLSearchParams
}): Promise<T> => {
  const { path, schema, init, searchParams } = args
  const url = buildUrl(path, searchParams)

  const response = await fetch(url, {
    headers: {
      Accept: 'application/json',
      ...init?.headers,
    },
    ...init,
  })

  if (!response.ok) {
    let detail: unknown
    try {
      detail = await response.json()
    } catch {
      try {
        detail = await response.text()
      } catch {
        detail = null
      }
    }
    throw new ApiError(`Request failed with status ${response.status}`, {
      status: response.status,
      detail,
      url,
    })
  }

  const data: unknown = await response.json()
  return schema.parse(data)
}
