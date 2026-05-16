import { create } from 'zustand'

const useAppStore = create((set, get) => ({
  currentRun: null,
  generatedOutput: '',
  isGenerating: false,
  generationError: null,
  prompt: '',

  setCurrentRun: (run) => set({ currentRun: run }),

  setPrompt: (prompt) => set({ prompt }),

  startGeneration: () => set({
    isGenerating: true,
    generatedOutput: '',
    generationError: null
  }),

  setGeneratedOutput: (output) => set({ generatedOutput: output }),

  completeGeneration: (output) => set({
    isGenerating: false,
    generatedOutput: output
  }),

  setGenerationError: (error) => set({
    isGenerating: false,
    generationError: error
  }),

  clearOutput: () => set({
    generatedOutput: '',
    generationError: null,
    prompt: ''
  }),

  reset: () => set({
    currentRun: null,
    generatedOutput: '',
    isGenerating: false,
    generationError: null,
    prompt: ''
  })
}))

export default useAppStore