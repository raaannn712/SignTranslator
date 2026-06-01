import 'dotenv/config'
import express from 'express'
import cors from 'cors'

const app = express()
app.use(express.json())
app.use(cors())
const PORT = 3000

const SUPABASE_URL = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || ''
const SUPABASE_KEY = process.env.VITE_SUPABASE_PUBLISHABLE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || ''

function requireSupabaseConfig(res) {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
        res.status(500).json({
            error: 'Supabase environment variables are missing. Set VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY in Backend/.env.'
        })
        return false
    }
    return true
}

async function supabaseRequest(path, options = {}) {
    const response = await fetch(`${SUPABASE_URL}/rest/v1${path}`, {
        ...options,
        headers: {
            apikey: SUPABASE_KEY,
            Authorization: `Bearer ${SUPABASE_KEY}`,
            'Content-Type': 'application/json',
            Prefer: 'return=representation',
            ...(options.headers || {})
        }
    })

    const text = await response.text()
    const data = text ? JSON.parse(text) : null

    if (!response.ok) {
        const message = data?.message || data?.error || text || `Supabase request failed with ${response.status}`
        const error = new Error(message)
        error.status = response.status
        error.details = data
        throw error
    }

    return data
}

async function ensureGesture(label, numHands = 1) {
    const lookup = await supabaseRequest(`/gestures?name=eq.${encodeURIComponent(label)}&select=id,name,num_hands&limit=1`, {
        method: 'GET'
    })

    if (Array.isArray(lookup) && lookup.length > 0) {
        return lookup[0]
    }

    const created = await supabaseRequest('/gestures', {
        method: 'POST',
        body: JSON.stringify({
            name: label,
            owner_tag: 'backend',
            is_public: true,
            num_hands: numHands
        })
    })

    return Array.isArray(created) ? created[0] : created
}

async function listApprovedSamples(gestureName = null) {
    if (!gestureName) {
        return supabaseRequest(
            '/samples?approved=eq.true&select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc',
            { method: 'GET' }
        )
    }

    const gestures = await supabaseRequest(
        `/gestures?name=eq.${encodeURIComponent(String(gestureName).trim().toUpperCase())}&select=id,name&limit=1`,
        { method: 'GET' }
    )

    if (!Array.isArray(gestures) || gestures.length === 0) {
        return []
    }

    return supabaseRequest(
        `/samples?approved=eq.true&gesture_id=eq.${gestures[0].id}&select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc`,
        { method: 'GET' }
    )
}

app.get('/health', (req, res) => {
    res.json({ ok: true })
})

app.post('/api/training/sample', async (req, res) => {
    try {
        if (!requireSupabaseConfig(res)) return

        const {
            label,
            landmarks,
            features,
            numHands = 1,
            handedness = 'Right',
            imagePath = null,
            source = 'webcam',
            quality = 0,
            approved = false,
            uploaderTag = 'anonymous',
            sessionId = null
        } = req.body || {}

        const normalizedLabel = String(label || '').trim().toUpperCase()
        if (!normalizedLabel) {
            return res.status(400).json({ error: 'label is required' })
        }

        if (!Array.isArray(landmarks) || landmarks.length === 0) {
            return res.status(400).json({ error: 'landmarks array is required' })
        }

        if (!Array.isArray(features) || features.length === 0) {
            return res.status(400).json({ error: 'features array is required' })
        }

        const gesture = await ensureGesture(normalizedLabel, Number(numHands) || 1)

        const insertedSamples = await supabaseRequest('/samples', {
            method: 'POST',
            body: JSON.stringify({
                gesture_id: gesture.id,
                uploader_tag: String(uploaderTag || 'anonymous'),
                session_id: sessionId,
                num_hands: Number(numHands) || 1,
                handedness,
                landmarks,
                features,
                image_path: imagePath,
                source,
                quality: Number.isFinite(Number(quality)) ? Number(quality) : 0,
                approved: Boolean(approved)
            })
        })

        res.status(201).json({
            ok: true,
            gesture,
            sample: Array.isArray(insertedSamples) ? insertedSamples[0] : insertedSamples
        })
    } catch (error) {
        console.error('Failed to save training sample:', error)
        res.status(error.status || 500).json({
            error: 'Failed to save training sample',
            message: error.message
        })
    }
})

async function handleExportApprovedSamples(req, res) {
    try {
        if (!requireSupabaseConfig(res)) return

        const gesture = req.query.gesture ? String(req.query.gesture) : null
        const samples = await listApprovedSamples(gesture)

        const normalized = Array.isArray(samples)
            ? samples.map((sample) => ({
                id: sample.id,
                gesture_id: sample.gesture_id,
                gesture_name: sample.gestures?.name || null,
                num_hands: sample.num_hands,
                handedness: sample.handedness,
                landmarks: sample.landmarks,
                features: sample.features,
                image_path: sample.image_path,
                source: sample.source,
                quality: sample.quality,
                created_at: sample.created_at
            }))
            : []

        res.json({
            ok: true,
            count: normalized.length,
            gesture: gesture,
            samples: normalized
        })
    } catch (error) {
        console.error('Failed to export approved samples:', error)
        res.status(error.status || 500).json({
            error: 'Failed to export approved samples',
            message: error.message
        })
    }
}

app.get('/api/training/export', handleExportApprovedSamples)

app.get('/api/training/export/approved', handleExportApprovedSamples)

app.get("/", (req, res) => {
    res.send("Hello from the backend")
})

app.listen(PORT, () => {
    console.log(`Serve is running on port: http://localhost:${PORT}`)
})

