import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

client.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

client.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    if (error.response) {
      const { status, data } = error.response

      if (status === 404) {
        console.error('[API 404]', data?.detail || 'Resource not found')
      } else if (status === 400) {
        console.error('[API 400]', data?.detail || 'Bad request')
      } else if (status === 500) {
        console.error('[API 500]', data?.detail || 'Internal server error')
      } else {
        console.error(`[API ${status}]`, data?.detail || 'Unknown error')
      }

      return Promise.reject({
        status,
        message: data?.detail || 'Something went wrong',
        raw: error
      })
    }

    if (error.request) {
      console.error('[API No Response] Server may be down or unreachable')
      return Promise.reject({
        status: null,
        message: 'No response from server — check if the API is running',
        raw: error
      })
    }

    console.error('[API Setup Error]', error.message)
    return Promise.reject({
      status: null,
      message: error.message,
      raw: error
    })
  }
)

export default client