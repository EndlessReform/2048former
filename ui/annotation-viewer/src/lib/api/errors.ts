type ApiErrorInit = {
  status: number
  detail?: unknown
  url: string
}

export class ApiError extends Error {
  readonly status: number
  readonly detail?: unknown
  readonly url: string

  constructor(message: string, init: ApiErrorInit) {
    super(message)
    this.name = 'ApiError'
    this.status = init.status
    this.detail = init.detail
    this.url = init.url
  }
}
