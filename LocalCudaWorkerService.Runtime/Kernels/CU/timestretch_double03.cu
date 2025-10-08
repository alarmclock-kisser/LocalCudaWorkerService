// CUDA version of timestretch_double03 using float2
// Variant A: single launch over all chunks (2D grid: bins x chunks) with contiguous input/output.
// This kernel now expects a contiguous buffer of Nchunks * chunkSize complex samples.
// chunkSize = number of complex bins per chunk; totalBins = chunkSize; total complex frames = chunks * chunkSize.
// prev chunk access uses previous chunk's same bin. No out-of-bounds checks beyond standard guards.

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" __global__ void timestretch_double03(
    const float2* __restrict__ inputAll, // size: chunkCount * chunkSize
    float2* __restrict__ outputAll,      // same layout
    int chunkSize,
    int overlapSize,
    int samplerate,
    double factor,
    int chunkCount)                      // NEW: number of chunks in the contiguous buffer
{
    int bin   = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. chunkSize-1
    int chunk = blockIdx.y * blockDim.y + threadIdx.y;   // 0 .. chunkCount-1

    if (bin >= chunkSize || chunk >= chunkCount)
        return;

    int hopIn  = chunkSize - overlapSize;

    int idx     = chunk * chunkSize + bin;
    int prevIdx = (chunk > 0) ? (chunk - 1) * chunkSize + bin : idx;

    if (chunk == 0)
    {
        outputAll[idx] = inputAll[idx];
        return;
    }

    float2 cur  = inputAll[idx];
    float2 prev = inputAll[prevIdx];

    float phaseCur  = atan2f(cur.y,  cur.x);
    float phasePrev = atan2f(prev.y, prev.x);

    float mag = hypotf(cur.x, cur.y);

    float deltaPhase = phaseCur - phasePrev;

    float freqPerBin       = (float)samplerate / (float)chunkSize;
    float expectedPhaseAdv = 2.0f * M_PI * freqPerBin * bin * hopIn / (float)samplerate;

    float delta = deltaPhase - expectedPhaseAdv;
    delta = fmodf(delta + M_PI, 2.0f * M_PI) - M_PI; // Wrap to [-PI, PI]

    float phaseOut = phasePrev + expectedPhaseAdv + (float)((double)delta * factor);

    outputAll[idx].x = mag * cosf(phaseOut);
    outputAll[idx].y = mag * sinf(phaseOut);
}