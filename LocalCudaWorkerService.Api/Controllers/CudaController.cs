using LocalCudaWorkerService.Runtime;
using LocalCudaWorkerService.Shared.Cuda;
using Microsoft.AspNetCore.Mvc;

namespace LocalCudaWorkerService.Api.Controllers
{
	[ApiController]
	[Route("api/[controller]")]
	public class CudaController : ControllerBase
	{
		private readonly CudaService cudaService;

		public CudaController(CudaService cudaService)
		{
			this.cudaService = cudaService;
		}

		[HttpGet("status")]
		[ProducesResponseType(typeof(CudaStatusInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaStatusInfo>> GetCudaStatusAsync()
		{
			try
			{
				var status = await Task.Run(() => new CudaStatusInfo(this.cudaService));

				return this.Ok(status);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error retrieving CUDA status",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpGet("devices")]
		[ProducesResponseType(typeof(IEnumerable<string>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<IEnumerable<string>>> GetCudaDevicesAsync()
		{
			try
			{
				var devices = (await Task.Run(() => this.cudaService.Devices)).Values;

				return this.Ok(devices);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error retrieving CUDA devices",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("initialize-index")]
		[ProducesResponseType(typeof(CudaStatusInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaStatusInfo>> InitializeCudaByIndexAsync([FromQuery] int deviceIndex = 0)
		{
			if (deviceIndex < 0)
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid device index",
					Detail = "Device index must be a non-negative integer.",
					Status = 400
				});
			}
			try
			{
				if (this.cudaService.Index == deviceIndex && this.cudaService.Initialized)
				{
					var currentStatus = new CudaStatusInfo(this.cudaService);
					return this.Ok(currentStatus);
				}

				if (deviceIndex >= this.cudaService.Devices.Count)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Device index out of range",
						Detail = $"Device index {deviceIndex} is out of range. Available devices: 0 to {this.cudaService.Devices.Count - 1}.",
						Status = 404
					});
				}

				this.cudaService.Initialize(deviceIndex);
				var status = new CudaStatusInfo(this.cudaService);
				if (!status.Initialized)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Failed to initialize CUDA",
						Detail = $"Could not initialize CUDA on device index {deviceIndex}.",
						Status = 400
					});
				}

				return this.Ok(status);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error initializing CUDA",
					Detail = ex.Message,
					Status = 500
				});
			}
			finally
			{
				await Task.Yield();
			}
		}

		[HttpPost("initialize-name")]
		[ProducesResponseType(typeof(CudaStatusInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaStatusInfo>> InitializeCudaByNameAsync([FromQuery] string deviceName = "")
		{
			if (string.IsNullOrWhiteSpace(deviceName))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid device name",
					Detail = "Device name must be a non-empty string.",
					Status = 400
				});
			}
			try
			{
				if (this.cudaService.SelectedDevice.Equals(deviceName, StringComparison.OrdinalIgnoreCase) && this.cudaService.Initialized)
				{
					var currentStatus = new CudaStatusInfo(this.cudaService);
					return this.Ok(currentStatus);
				}

				if (!this.cudaService.Devices.Values.Contains(deviceName, StringComparer.OrdinalIgnoreCase))
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Device name not found",
						Detail = $"Device name '{deviceName}' not found. Available devices: {string.Join(", ", this.cudaService.Devices.Values)}.",
						Status = 404
					});
				}

				this.cudaService.Initialize(deviceName);

				var status = new CudaStatusInfo(this.cudaService);
				if (!status.Initialized)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Failed to initialize CUDA",
						Detail = $"Could not initialize CUDA on device '{deviceName}'.",
						Status = 400
					});
				}

				return this.Ok(status);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error initializing CUDA",
					Detail = ex.Message,
					Status = 500
				});
			}
			finally
			{
				await Task.Yield();
			}
		}

		[HttpDelete("dispose")]
		[ProducesResponseType(typeof(CudaStatusInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaStatusInfo>> DisposeCudaAsync()
		{
			try
			{
				await Task.Run(() => this.cudaService.Dispose());

				var status = new CudaStatusInfo(this.cudaService);
				if (status.Initialized)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Failed to dispose CUDA",
						Detail = "CUDA service is still initialized after dispose call.",
						Status = 400
					});
				}

				return this.Ok(status);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error disposing CUDA",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("compile-raw")]
		[ProducesResponseType(typeof(CudaKernelInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaKernelInfo>> CompileCudaKernelAsync([FromBody] string code, [FromQuery] string? cudaDeviceName = null)
		{
			if (string.IsNullOrWhiteSpace(code))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid CUDA source",
					Detail = "CUDA source code must be a non-empty string.",
					Status = 400
				});
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(cudaDeviceName))
				{
					this.cudaService.Initialize(cudaDeviceName);
				}
			}
			if (!this.cudaService.Initialized || this.cudaService.Compiler == null)
			{
				return this.StatusCode(500, new ProblemDetails { Title = "OpenCL not initialized", Status = 500 });
			}

			try
			{
				CudaKernelInfo info = new();

				string? compileResult = await Task.Run(() => this.cudaService.Compiler.CompileString(code));
				if (string.IsNullOrEmpty(compileResult))
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Compilation failed",
						Detail = "CUDA kernel compilation returned no result.",
						Status = 400
					});
				}

				if (compileResult.Contains(' '))
				{
					info.CompilationLog = compileResult;
					info.SuccessfullyCompiled = false;
				}
				else
				{
					info = new(this.cudaService.Compiler, compileResult)
					{
						SuccessfullyCompiled = true,
						CompilationLog = ""
					};
				}

				return this.Ok(info);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error compiling CUDA kernel",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("compile-file")]
		[Consumes("multipart/form-data")]
		[ProducesResponseType(typeof(CudaKernelInfo), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CudaKernelInfo>> CompileFileAsync(IFormFile file, [FromQuery] string? cudaDeviceName = "NVIDIA")
		{
			if (file == null || file.Length == 0)
			{
				return this.BadRequest(new ProblemDetails { Title = "Invalid request", Detail = "No file uploaded or file empty.", Status = 400 });
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(cudaDeviceName))
				{
					this.cudaService.Initialize(cudaDeviceName);
				}
			}
			if (!this.cudaService.Initialized || this.cudaService.Compiler == null)
			{
				return this.StatusCode(500, new ProblemDetails { Title = "OpenCL not initialized", Status = 500 });
			}

			string code;
			using (var reader = new StreamReader(file.OpenReadStream()))
			{
				code = await reader.ReadToEndAsync();
			}
			if (string.IsNullOrWhiteSpace(code))
			{
				return this.BadRequest(new ProblemDetails { Title = "Invalid request", Detail = "Uploaded file is empty.", Status = 400 });
			}

			string? compileResult = this.cudaService.Compiler.CompileString(code);
			if (string.IsNullOrEmpty(compileResult))
			{
				return this.BadRequest(new ProblemDetails { Title = "Compilation failed", Detail = "Kernel compilation returned no result.", Status = 400 });
			}

			CudaKernelInfo info = new();

			if (compileResult.Contains(' '))
			{
				info.CompilationLog = compileResult;
				info.SuccessfullyCompiled = false;
			}
			else
			{
				info = new(this.cudaService.Compiler, compileResult)
				{
					SuccessfullyCompiled = true,
					CompilationLog = ""
				};
			}

			return this.Ok(compileResult);
		}


	}
}
