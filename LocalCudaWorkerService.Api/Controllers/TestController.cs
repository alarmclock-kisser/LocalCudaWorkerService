using LocalCudaWorkerService.Runtime;
using LocalCudaWorkerService.Shared;
using LocalCudaWorkerService.Shared.Cuda;
using ManagedCuda.VectorTypes;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace LocalCudaWorkerService.Api.Controllers
{
	[ApiController]
	[Route("api/[controller]")]
	public class TestController : ControllerBase
	{
		private readonly CudaService cudaService;

		public TestController(CudaService cudaService)
		{
			this.cudaService = cudaService;
		}

		[HttpGet("test-cufft-forward-single")]
		[ProducesResponseType(typeof(float2[]), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<float2[]>> TestCufftForwardSingleAsync([FromQuery] int? fftSize = null, [FromQuery] string? deviceName = "NVIDIA")
		{
			fftSize ??= 1024;
			fftSize = Math.Clamp(fftSize.Value, 16, 65536);
			if (fftSize.Value != (int) Math.Pow(2, Math.Ceiling(Math.Log2(fftSize.Value))))
			{
				Console.WriteLine($"[Warning] Adjusting fftSize from {fftSize.Value} to next power of two.");
				fftSize = (int) Math.Pow(2, Math.Ceiling(Math.Log2(fftSize.Value)));
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(deviceName))
				{
					this.cudaService.Initialize(deviceName);
				}
				if (!this.cudaService.Initialized)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CUDA not initialized",
						Detail = "CUDA service is not initialized. Please initialize with a valid device name or index.",
						Status = 500
					});
				}
			}

			try
			{
				// Create float[] sin wave input data
				float[] inputData = new float[fftSize.Value];
				for (int i = 0; i < fftSize.Value; i++)
				{
					inputData[i] = (float) Math.Sin(2 * Math.PI * i / fftSize.Value);
				}

				var resultData = await this.cudaService.ExecuteFftAsyncSafe(inputData);
				if (resultData == null)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CuFFT forward test failed",
						Detail = "FFT execution returned null result.",
						Status = 500
					});
				}

				return this.Ok(resultData);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "CuFFT forward test failed",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpGet("test-cufft-inverse-single")]
		[ProducesResponseType(typeof(float[]), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<float[]>> TestCufftInverseSingleAsync([FromQuery] int? fftSize = null, [FromQuery] string? deviceName = "NVIDIA")
		{
			fftSize ??= 1024;
			fftSize = Math.Clamp(fftSize.Value, 16, 65536);
			fftSize = (int) Math.Pow(2, Math.Ceiling(Math.Log2(fftSize.Value)));

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(deviceName))
				{
					this.cudaService.Initialize(deviceName);
				}
				if (!this.cudaService.Initialized)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CUDA not initialized",
						Detail = "CUDA service is not initialized. Please initialize with a valid device name or index.",
						Status = 500
					});
				}
			}

			try
			{
				// Create float[] sin wave input data
				float[] inputData = new float[fftSize.Value];
				for (int i = 0; i < fftSize.Value; i++)
				{
					inputData[i] = (float) Math.Sin(2 * Math.PI * i / fftSize.Value);
				}

				var fftData = await this.cudaService.ExecuteFftAsyncSafe(inputData);
				if (fftData == null)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CuFFT inverse test failed",
						Detail = "FFT execution returned null result.",
						Status = 500
					});
				}

				var resultData = await this.cudaService.ExecuteIfftAsyncSafe(fftData);
				if (resultData == null)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CuFFT inverse test failed",
						Detail = "iFFT execution returned null result.",
						Status = 500
					});
				}

				return this.Ok(resultData);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "CuFFT inverse test failed",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpGet("test-execute-single-round")]
		[ProducesResponseType(typeof(KernelExecuteResult), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> TestExecuteSingleRoundAsync([FromQuery] string? deviceName = "NVIDIA", [FromQuery] int? inputSize = 1024)
		{
			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(deviceName))
				{
					this.cudaService.Initialize(deviceName);
				}
				if (!this.cudaService.Initialized)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "CUDA not initialized",
						Detail = "CUDA service is not initialized. Please initialize with a valid device name or index.",
						Status = 500
					});
				}
			}

			Random rand = new Random();
			int inputLength = inputSize.HasValue ? inputSize.Value : 1024;
			float[] inputData = new float[inputLength];
			for (int i = 0; i < inputData.Length; i++)
			{
				// Random float between -99.999 and 99.999
				inputData[i] = (float) (rand.NextDouble() * 199.998 - 99.999);
			}

			// Convert inputData to base64
			string inputBase64 = Convert.ToBase64String(inputData.SelectMany(d => BitConverter.GetBytes(d)).ToArray());

			// Simple CUDA Kernel code to round float to int array
			string kernelCode = @"
				extern ""C"" __global__ void to_int_round_or_cut_f(
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
				}";

			KernelExecuteRequest request = new KernelExecuteRequest
			{
				DeviceName = deviceName ?? "NVIDIA",
				KernelCode = kernelCode,
				KernelName = "to_int_round_or_cut_f",
				InputDataBase64 = inputBase64,
				InputDataType = "float",
				OutputDataType = "int",
				OutputDataLength = inputLength.ToString(),
				WorkDimension = 1,
				ArgumentNames = ["input", "output", "cutoffMode", "length"],
				ArgumentTypes = ["float*", "int*", "int", "int"],
				ArgumentValues = ["0", "0", "0", inputLength.ToString()]
			};

			Dictionary<string, string> arguments = [];
			for (int i = 0; i < request.ArgumentNames.Count(); i++)
			{
				arguments.Add(request.ArgumentNames.ElementAt(i), request.ArgumentValues.ElementAt(i));
			}

			{
				try
				{
					Stopwatch sw = Stopwatch.StartNew();

					var resultString = await this.cudaService.ExecuteGenericKernelSingleAsyncSafe(request.KernelCode, request.InputDataBase64, request.InputDataType,
						request.OutputDataLength, request.OutputDataType, arguments, request.WorkDimension, request.DeviceName);

					if (resultString == null)
					{
						return this.StatusCode(500, new ProblemDetails
						{
							Title = "Kernel execution test failed",
							Detail = "Kernel execution returned null result.",
							Status = 500
						});
					}
					else
					{
						KernelExecuteResult result = new()
						{
							KernelName = request.KernelName,
							OutputDataBase64 = resultString,
							OutputDataType = request.OutputDataType,
							OutputDataLength = request.OutputDataLength,
							OutputPointer = "0",
							Message = "Kernel executed successfully.",
							Success = true,
							ExecutionTimeMs = sw.ElapsedMilliseconds
						};

						sw.Stop();

						return this.Ok(result);

					}
				}
				catch (Exception ex)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Kernel execution test failed",
						Detail = ex.Message,
						Status = 500
					});
				}
			}
		}

		[HttpGet("test-execute-batch-round")]
		[ProducesResponseType(typeof(KernelExecuteResult), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> TestExecuteBatchRoundAsync([FromQuery] string? deviceName = "NVIDIA", [FromQuery] int? inputSize = 1024)
		{
			Random rand = new Random();
			int inputLength = inputSize.HasValue ? inputSize.Value : 1024;
			List<float[]> inputChunks = [];
			for (int chunk = 0; chunk < 4; chunk++)
			{
				float[] chunkData = new float[inputLength];
				for (int i = 0; i < chunkData.Length; i++)
				{
					// Random float between -99.999 and 99.999
					chunkData[i] = (float) (rand.NextDouble() * 199.998 - 99.999);
				}
				inputChunks.Add(chunkData);
			}

			// Convert inputData to base64
			string[] inputBase64 = inputChunks.Select(c => Convert.ToBase64String(c.SelectMany(d => BitConverter.GetBytes(d)).ToArray())).ToArray();

			// Simple CUDA Kernel code to round float to int array
			string kernelCode = @"
				extern ""C"" __global__ void to_int_round_or_cut_f(
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
				}";

			KernelExecuteRequest request = new KernelExecuteRequest
			{
				DeviceName = deviceName ?? "NVIDIA",
				KernelCode = kernelCode,
				KernelName = "to_int_round_or_cut_f",
				InputDataBase64 = null,
				InputDataBase64Chunks = inputBase64,
				InputDataType = "float",
				OutputDataType = "int",
				OutputDataLength = inputLength.ToString(),
				WorkDimension = 1,
				ArgumentNames = ["input", "output", "cutoffMode", "length"],
				ArgumentTypes = ["float*", "int*", "int", "int"],
				ArgumentValues = ["0", "0", "0", inputLength.ToString()]
			};

			Dictionary<string, string> arguments = [];
			for (int i = 0; i < request.ArgumentNames.Count(); i++)
			{
				arguments.Add(request.ArgumentNames.ElementAt(i), request.ArgumentValues.ElementAt(i));
			}

			{
				try
				{
					Stopwatch sw = Stopwatch.StartNew();

					var resultStrings = await this.cudaService.ExecuteGenericKernelBatchAsyncSafe(request.KernelCode, request.InputDataBase64Chunks.ToArray(), request.InputDataType,
						request.OutputDataLength, request.InputDataBase64Chunks.Count().ToString(), request.OutputDataType, arguments, request.WorkDimension, request.DeviceName);

					if (resultStrings == null)
					{
						return this.StatusCode(500, new ProblemDetails
						{
							Title = "Kernel execution test failed",
							Detail = "Kernel execution returned null result.",
							Status = 500
						});
					}
					else
					{
						KernelExecuteResult result = new()
						{
							KernelName = request.KernelName,
							OutputDataBase64 = null,
							OutputDataBase64Chunks = resultStrings,
							OutputDataType = request.OutputDataType,
							OutputDataLength = request.OutputDataLength,
							OutputPointer = "0",
							Message = "Kernel executed successfully.",
							Success = true,
							ExecutionTimeMs = sw.ElapsedMilliseconds
						};

						sw.Stop();

						return this.Ok(result);

					}
				}
				catch (Exception ex)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Kernel execution test failed",
						Detail = ex.Message,
						Status = 500
					});
				}
			}
		}



	}
}
