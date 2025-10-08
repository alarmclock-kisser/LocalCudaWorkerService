using LocalCudaWorkerService.Runtime;
using ManagedCuda.VectorTypes;
using Microsoft.AspNetCore.Mvc;

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

		[HttpGet("cufft-forward-single")]
		[ProducesResponseType(typeof(float2[]), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<float2[]>> TestCufftForwardSingleAsync([FromQuery] int? fftSize = null, [FromQuery] string? deviceName = "NVIDIA")
		{
			fftSize ??= 1024;
			fftSize = Math.Clamp(fftSize.Value, 16, 65536);
			if (fftSize.Value != (int)Math.Pow(2, Math.Ceiling(Math.Log2(fftSize.Value))))
			{
				Console.WriteLine($"[Warning] Adjusting fftSize from {fftSize.Value} to next power of two.");
				fftSize = (int)Math.Pow(2, Math.Ceiling(Math.Log2(fftSize.Value)));
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
					inputData[i] = (float)Math.Sin(2 * Math.PI * i / fftSize.Value);
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

		[HttpGet("cufft-inverse-single")]
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
	}
}
