import client from './client'

export async function generateText({
  runId,
  prompt,
  maxNewTokens = 200,
  temperature = 0.8,
  topK = 50,
  topP = 0.9,
  greedy = false,
  repetitionPenalty = 1.0
}) {
  if (!runId) throw new Error('runId is required for generation')
  if (!prompt || !prompt.trim()) throw new Error('Prompt cannot be empty')

  try {
    const response = await client.post('/generate/', {
      run_id: runId,
      prompt: prompt.trim(),
      max_new_tokens: maxNewTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      greedy,
      repetition_penalty: repetitionPenalty
    })

    return response.data

  } catch (error) {
    console.error('[generateText] Failed:', error.message)
    throw error
  }
}