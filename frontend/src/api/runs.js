import client from './client'

export async function fetchRuns() {
  try {
    const response = await client.get('/runs/')
    return response.data

  } catch (error) {
    console.error('[fetchRuns] Failed:', error.message)
    throw error
  }
}

export async function fetchRun(runId) {
  if (!runId) throw new Error('runId is required')

  try {
    const response = await client.get(`/runs/${runId}`)
    return response.data

  } catch (error) {
    console.error(`[fetchRun] Failed for runId=${runId}:`, error.message)
    throw error
  }
}