// NVRTC‑freundliche Version ohne eigene float2-Definition (float2 kommt aus den eingebauten NVRTC Headers)

// Eigene Konstanten
#ifndef PHASE_PVOC_PI_F
#define PHASE_PVOC_PI_F 3.14159265358979323846f
#endif
#ifndef PHASE_PVOC_TWOPI_F
#define PHASE_PVOC_TWOPI_F (2.0f * PHASE_PVOC_PI_F)
#endif

// llround Ersatz (Header-frei)
__device__ __forceinline__ long long llround_device(double v)
{
    return (long long)(v >= 0.0 ? v + 0.5 : v - 0.5);
}

// Globale Zustände (per cudaMemcpyToSymbol zu setzen)
__device__ float* g_prevPhase = nullptr;
__device__ float* g_phaseAccum = nullptr;

// Phase Wrap in (-PI, PI]
__device__ __forceinline__ float wrapPhase(float x)
{
    x = fmodf(x + PHASE_PVOC_PI_F, PHASE_PVOC_TWOPI_F);
    if (x < 0.0f) x += PHASE_PVOC_TWOPI_F;
    return x - PHASE_PVOC_PI_F;
}

extern "C" __global__ void timestretch_complexes01(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    const int chunkSize,
    const int overlapSize,
    const int samplerate,
    const int channels,
    const double factor)
{
    if (!g_prevPhase || !g_phaseAccum) return;

    int totalBins = channels * chunkSize;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= totalBins) return;

    int channel = gid / chunkSize; // reserviert für Erweiterungen
    int k = gid - channel * chunkSize;

    int Ha = chunkSize - overlapSize;
    if (Ha <= 0) return;

    int Hs = (int)llround_device(factor * (double)Ha);
    if (Hs <= 0) Hs = 1;

    float2 in = input[gid];
    // eigener Betrag statt hypotf (vermeidet zusätzliche Abhängigkeit)
    float mag = sqrtf(in.x * in.x + in.y * in.y);
    float phase = atan2f(in.y, in.x);

    float prevPhase = g_prevPhase[gid];
    float phaseAccum = g_phaseAccum[gid];

    float omega_k = (PHASE_PVOC_TWOPI_F * (float)k / (float)chunkSize) * (float)Ha;

    float delta = (phase - prevPhase) - omega_k;
    delta = wrapPhase(delta);

    float trueFreq = omega_k + delta;
    float scale = (float)Hs / (float)Ha;
    phaseAccum += trueFreq * scale;

    float outPhase = phaseAccum;
    float s = sinf(outPhase);
    float c = cosf(outPhase);

    float2 out;
    out.x = mag * c;
    out.y = mag * s;
    output[gid] = out;

    g_prevPhase[gid] = phase;
    g_phaseAccum[gid] = phaseAccum;
}

extern "C" __global__ void initPhaseState(
    const float2* __restrict__ firstFrame,
    const int chunkSize,
    const int channels)
{
    if (!g_prevPhase || !g_phaseAccum) return;

    int totalBins = channels * chunkSize;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= totalBins) return;

    float2 v = firstFrame[gid];
    float p = atan2f(v.y, v.x);
    g_prevPhase[gid] = p;
    g_phaseAccum[gid] = p;
}

// Alternative Kernel ohne globale Symbole (falls gewünscht):
extern "C" __global__ void timestretch01_stateptr(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    float* __restrict__ prevPhase,
    float* __restrict__ phaseAccumBuf,
    const int chunkSize,
    const int overlapSize,
    const int samplerate,
    const int channels,
    const double factor)
{
    int totalBins = channels * chunkSize;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= totalBins) return;

    int channel = gid / chunkSize;
    int k = gid - channel * chunkSize;

    int Ha = chunkSize - overlapSize;
    if (Ha <= 0) return;

    int Hs = (int)llround_device(factor * (double)Ha);
    if (Hs <= 0) Hs = 1;

    float2 in = input[gid];
    float mag = sqrtf(in.x * in.x + in.y * in.y);
    float phase = atan2f(in.y, in.x);

    float prev = prevPhase[gid];
    float accum = phaseAccumBuf[gid];

    float omega_k = (PHASE_PVOC_TWOPI_F * (float)k / (float)chunkSize) * (float)Ha;

    float delta = (phase - prev) - omega_k;
    delta = wrapPhase(delta);

    float trueFreq = omega_k + delta;
    float scale = (float)Hs / (float)Ha;
    accum += trueFreq * scale;

    float outPhase = accum;
    float s = sinf(outPhase);
    float c = cosf(outPhase);

    float2 out;
    out.x = mag * c;
    out.y = mag * s;
    output[gid] = out;

    prevPhase[gid] = phase;
    phaseAccumBuf[gid] = accum;
}