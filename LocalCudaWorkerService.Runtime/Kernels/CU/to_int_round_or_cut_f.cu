
				extern "C" __global__ void to_int_round_or_cut_f(
										   const float* __restrict__ input,
										   int* __restrict__ output,
										   int cutoffMode,
										   int length)
				{
					int gid = blockIdx.x * blockDim.x + threadIdx.x;
					if (gid >= length) return;

					float v = input[gid];
					int result;
					if (cutoffMode != 0)
					{
						result = (int)(v);
					}
					else
					{
						result = (v >= 0.0f)
							? (int)floorf(v + 0.5f)
							: (int)ceilf(v - 0.5f);
					}
					output[gid] = result;
				}