import { useEffect, type ChangeEvent } from 'react'
import { useHealthQuery } from '../lib/api/queries'
import { useViewerPreferences, type TokenizationMode } from '../hooks/useViewerPreferences'

export function TokenizationControls() {
  const tokenizationMode = useViewerPreferences((state) => state.tokenizationMode)
  const setTokenizationMode = useViewerPreferences((state) => state.setTokenizationMode)
  const tokenizerInfo = useViewerPreferences((state) => state.tokenizerInfo)
  const setTokenizerInfo = useViewerPreferences((state) => state.setTokenizerInfo)

  const { data, isLoading, isFetching, isError } = useHealthQuery()

  const tokenizerAvailable = Boolean(data?.tokenizer)
  const fetching = isLoading || isFetching

  useEffect(() => {
    if (data?.tokenizer) {
      const info = data.tokenizer
      setTokenizerInfo({
        tokenizerType: info.tokenizer_type,
        numBins: info.num_bins,
        vocabOrder: info.vocab_order,
        valuationTypes: info.valuation_types,
      })
    } else if (!fetching && tokenizerInfo !== null) {
      setTokenizerInfo(null)
    }
  }, [data, fetching, setTokenizerInfo])

  useEffect(() => {
    if (!fetching && !tokenizerAvailable && tokenizationMode === 'preview') {
      setTokenizationMode('off')
    }
  }, [fetching, tokenizerAvailable, setTokenizationMode, tokenizationMode])

  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextMode = event.target.value as TokenizationMode
    setTokenizationMode(nextMode)
  }

  let metaText: string | null = null
  let metaClass = 'tokenization-meta'

  if (fetching) {
    metaText = 'Checking…'
  } else if (tokenizerAvailable && data?.tokenizer) {
    metaText = `${data.tokenizer.num_bins} bins · ${data.tokenizer.tokenizer_type}`
    if (tokenizationMode === 'preview') {
      metaClass += ' tokenization-meta-active'
    }
  } else if (isError) {
    metaText = 'Tokenizer health unavailable'
    metaClass += ' tokenization-meta-warning'
  } else {
    metaText = 'Tokenizer not loaded'
    if (tokenizationMode === 'preview') {
      metaClass += ' tokenization-meta-warning'
    }
  }

  return (
    <div className="status-item tokenization-control">
      <label className="status-label" htmlFor="tokenization-mode">
        Tokenization
      </label>
      <select
        id="tokenization-mode"
        className="tokenization-select"
        value={tokenizationMode}
        onChange={handleChange}
        disabled={fetching}
      >
        <option value="off">Annotations only</option>
        <option value="preview" disabled={!tokenizerAvailable}>
          Tokenizer bins
        </option>
      </select>
      {metaText ? <span className={metaClass}>{metaText}</span> : null}
    </div>
  )
}
