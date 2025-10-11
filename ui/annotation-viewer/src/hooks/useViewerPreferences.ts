import { create } from 'zustand'

export type TokenizationMode = 'off' | 'preview'

export interface TokenizerInfo {
  tokenizerType: string
  numBins: number
  vocabOrder: string[]
  valuationTypes: string[]
}

interface ViewerPreferencesState {
  tokenizationMode: TokenizationMode
  setTokenizationMode: (mode: TokenizationMode) => void
  tokenizerInfo: TokenizerInfo | null
  setTokenizerInfo: (info: TokenizerInfo | null) => void
}

export const useViewerPreferences = create<ViewerPreferencesState>((set) => ({
  tokenizationMode: 'off',
  setTokenizationMode: (mode) => set({ tokenizationMode: mode }),
  tokenizerInfo: null,
  setTokenizerInfo: (info) =>
    set((state) => ({
      tokenizerInfo: info,
      tokenizationMode:
        info && state.tokenizerInfo === null && state.tokenizationMode === 'off'
          ? 'preview'
          : state.tokenizationMode,
    })),
}))
