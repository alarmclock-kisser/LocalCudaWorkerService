using LocalCudaWorkerService.Runtime;
using LocalCudaWorkerService.Shared;
using LocalCudaWorkerService.Shared.Cuda;
using ManagedCuda.VectorTypes;
using Microsoft.AspNetCore.Mvc;
using System.Data.SqlTypes;
using System.Diagnostics;

namespace LocalCudaWorkerService.Api.Controllers
{
	[ApiController]
	[Route("api/[controller]")]
	public class CudaController : ControllerBase
	{
		private readonly ApiConfiguration apiConfig;
		private readonly CudaService cudaService;

		public CudaController(ApiConfiguration apiConfiguration, CudaService cudaService)
		{

			this.apiConfig = apiConfiguration;
			this.cudaService = cudaService;
		}

		[HttpGet("config")]
		[ProducesResponseType(typeof(ApiConfiguration), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public ActionResult<ApiConfiguration> GetApiConfiguration()
		{
			try
			{
				return this.Ok(this.apiConfig);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error retrieving API configuration",
					Detail = ex.Message,
					Status = 500
				});
			}
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

			if (!this.cudaService.Initialized && !string.IsNullOrWhiteSpace(cudaDeviceName))
			{
				this.cudaService.Initialize(cudaDeviceName);
			}
			if (!this.cudaService.Initialized || this.cudaService.Compiler == null)
			{
				return this.StatusCode(500, new ProblemDetails { Title = "CUDA nicht initialisiert", Status = 500 });
			}

			try
			{
				var compileResult = await this.cudaService.CompileStringAsyncSafe(code);
				if (string.IsNullOrEmpty(compileResult))
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Compilation failed",
						Detail = "CUDA kernel compilation returned no result.",
						Status = 400
					});
				}

				CudaKernelInfo info;
				if (compileResult.Contains(' '))
				{
					info = new CudaKernelInfo(this.cudaService.Compiler, "")
					{
						SuccessfullyCompiled = false,
						CompilationLog = compileResult
					};
				}
				else
				{
					info = new CudaKernelInfo(this.cudaService.Compiler, compileResult)
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

			if (!this.cudaService.Initialized && !string.IsNullOrWhiteSpace(cudaDeviceName))
			{
				this.cudaService.Initialize(cudaDeviceName);
			}
			if (!this.cudaService.Initialized || this.cudaService.Compiler == null)
			{
				return this.StatusCode(500, new ProblemDetails { Title = "CUDA nicht initialisiert", Status = 500 });
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

			var compileResult = await this.cudaService.CompileStringAsyncSafe(code);
			if (string.IsNullOrEmpty(compileResult))
			{
				return this.BadRequest(new ProblemDetails { Title = "Compilation failed", Detail = "Kernel compilation returned no result.", Status = 400 });
			}

			CudaKernelInfo info;
			if (compileResult.Contains(' '))
			{
				info = new CudaKernelInfo(this.cudaService.Compiler, "")
				{
					SuccessfullyCompiled = false,
					CompilationLog = compileResult
				};
			}
			else
			{
				info = new CudaKernelInfo(this.cudaService.Compiler, compileResult)
				{
					SuccessfullyCompiled = true,
					CompilationLog = ""
				};
			}

			return this.Ok(info);
		}

		[HttpPost("request-cufft")]
		[ProducesResponseType(typeof(CuFftResult), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<CuFftResult>> ProcessCufftRequestAsync([FromBody] CuFftRequest request)
		{
			if (request == null || request.Size <= 0 || request.Batches <= 0 || request.DataChunks == null || !request.DataChunks.Any())
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request",
					Detail = "Request must include valid Size, Batches, and non-empty DataChunks.",
					Status = 400
				});
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(request.DeviceName))
				{
					this.cudaService.Initialize(request.DeviceName);
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
				CuFftResult? result = null;
				if (request.Inverse.HasValue == true && request.Inverse.Value == true)
				{
					var ifftResult = await this.cudaService.ExecuteIfftBulkAsyncSafe(request.DataChunks.Cast<float2[]>());
					if (ifftResult != null)
					{
						result = new()
						{
							DataChunks = ifftResult.Cast<object[]>(),
							DataForm = "c"
						};
					}
				}
				else if (request.Inverse.HasValue == true && request.Inverse.Value == false)
				{
					var fftResult = await this.cudaService.ExecuteFftBulkAsyncSafe(request.DataChunks.Cast<float[]>());
					if (fftResult != null)
					{
						result = new()
						{
							DataChunks = fftResult.Cast<object[]>(),
							DataForm = "c"
						};
					}
				}

				if (result == null || !result.DataChunks.Any())
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Processing failed",
						Detail = "CUFFT processing returned no or empty result.",
						Status = 500
					});
				}

				return this.Ok(result);
			}
			catch (ArgumentException argEx)
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request data",
					Detail = argEx.Message,
					Status = 400
				});
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error processing CUFFT request",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("request-generic-execution-single")]
		[ProducesResponseType(typeof(KernelExecuteResult), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> ProcessGenericKernelExecutionSingleAsync([FromBody] KernelExecuteRequest request)
		{
			if (request == null || string.IsNullOrWhiteSpace(request.KernelCode))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request",
					Detail = "Request must include valid Kernel soucre code",
					Status = 400
				});
			}
			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(request.DeviceName))
				{
					this.cudaService.Initialize(request.DeviceName);
				}
				if (request.DeviceIndex.HasValue)
				{
					if (request.DeviceIndex.Value > this.cudaService.Devices.Count)
					{
						return this.NotFound(new ProblemDetails
						{
							Title = "Device index out of range",
							Detail = $"Device index {request.DeviceIndex.Value} is out of range. Available devices: 0 to {this.cudaService.Devices.Count - 1}.",
							Status = 404
						});
					}
					this.cudaService.Initialize(request.DeviceIndex.Value);
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
				Dictionary<string, string> arguments = [];
				for (int i = 0; i < request.ArgumentNames.Count(); i++)
				{
					string argName = request.ArgumentNames.ElementAt(i);
					string argValue = request.ArgumentValues.ElementAtOrDefault(i) ?? "";
					if (!string.IsNullOrWhiteSpace(argName))
					{
						arguments[argName] = argValue;
					}
				}

				Stopwatch sw = Stopwatch.StartNew();

				var execResult = await this.cudaService.ExecuteGenericKernelSingleAsyncSafe(request.KernelCode, request.InputDataBase64, request.InputDataType, request.OutputDataLength, request.OutputDataType, arguments, request.WorkDimension, request.DeviceName);
				if (string.IsNullOrWhiteSpace(execResult))
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Execution failed",
						Detail = "Kernel execution returned no result.",
						Status = 500
					});
				}

				KernelExecuteResult result = new()
				{
					KernelName = request.KernelName ?? "UnnamedKernel",
					Success = true,
					Message = "Execution successful",
					OutputDataBase64 = execResult,
					OutputDataType = request.OutputDataType,
					OutputDataLength = request.OutputDataLength,
					ExecutionTimeMs = sw.Elapsed.TotalMilliseconds
				};

				sw.Stop();
				return this.Ok(result);
			}
			catch (ArgumentException argEx)
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request data",
					Detail = argEx.Message,
					Status = 400
				});
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error executing kernel",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("request-generic-execution-single-base64")]
		[ProducesResponseType(typeof(string), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> ProcessGenericKernelExecutionSingleBase64Async([FromBody] string? base64Data, [FromQuery] string? kernelCode = null, [FromQuery] string? inputDataType = null, [FromQuery] string outputDataLength = "0", [FromQuery] string outputDataType = "object", [FromQuery] IEnumerable<string>? argNames = null, [FromQuery] IEnumerable<string>? argValues = null, [FromQuery] string? deviceName = null, [FromQuery] int? deviceIndex = null, [FromQuery] int workDimension = 1)
		{
			if (string.IsNullOrWhiteSpace(kernelCode))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request",
					Detail = "Request must include valid Kernel source code",
					Status = 400
				});
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(deviceName))
				{
					this.cudaService.Initialize(deviceName);
				}
				else if (deviceIndex.HasValue)
				{
					if (deviceIndex.Value > this.cudaService.Devices.Count)
					{
						return this.NotFound(new ProblemDetails
						{
							Title = "Device index out of range",
							Detail = $"Device index {deviceIndex.Value} is out of range. Available devices: 0 to {this.cudaService.Devices.Count - 1}.",
							Status = 404
						});
					}

					this.cudaService.Initialize(deviceIndex.Value);
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

			var request = new KernelExecuteRequest
			{
				KernelCode = kernelCode,
				InputDataBase64 = base64Data,
				InputDataType = inputDataType,
				OutputDataLength = outputDataLength,
				OutputDataType = outputDataType,
				ArgumentNames = argNames ?? [],
				ArgumentValues = argValues ?? [],
				DeviceName = deviceName,
				DeviceIndex = deviceIndex,
				WorkDimension = workDimension
			};

			return await this.ProcessGenericKernelExecutionSingleAsync(request);
		}

		[HttpPost("request-generic-execution-batch")]
		[ProducesResponseType(typeof(KernelExecuteResult), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> ProcessGenericKernelExecutionBatchAsync([FromBody] KernelExecuteRequest request)
		{
			if (request == null || string.IsNullOrWhiteSpace(request.KernelCode))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request",
					Detail = "Request must include valid Kernel source code",
					Status = 400
				});
			}
			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(request.DeviceName))
				{
					this.cudaService.Initialize(request.DeviceName);
				}
				if (request.DeviceIndex.HasValue)
				{
					if (request.DeviceIndex.Value > this.cudaService.Devices.Count)
					{
						return this.NotFound(new ProblemDetails
						{
							Title = "Device index out of range",
							Detail = $"Device index {request.DeviceIndex.Value} is out of range. Available devices: 0 to {this.cudaService.Devices.Count - 1}.",
							Status = 404
						});
					}
					this.cudaService.Initialize(request.DeviceIndex.Value);
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
				Dictionary<string, string> arguments = [];
				for (int i = 0; i < request.ArgumentNames.Count(); i++)
				{
					string argName = request.ArgumentNames.ElementAt(i);
					string argValue = request.ArgumentValues.ElementAtOrDefault(i) ?? "";
					if (!string.IsNullOrWhiteSpace(argName))
					{
						arguments[argName] = argValue;
					}
				}

				Stopwatch sw = Stopwatch.StartNew();

				var execResult = await this.cudaService.ExecuteGenericKernelBatchAsyncSafe(request.KernelCode, request.InputDataBase64Chunks?.ToArray(), request.InputDataType, request.OutputDataLength, request.InputDataStride.ToString(), request.OutputDataType, arguments, request.WorkDimension, request.DeviceName);
				if (execResult?.LongLength <= 0)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Title = "Execution failed",
						Detail = "Kernel execution returned no or empty result.",
						Status = 500
					});
				}

				KernelExecuteResult result = new()
				{
					KernelName = request.KernelName ?? "UnnamedKernel",
					Success = true,
					Message = "Execution successful",
					OutputDataBase64 = null,
					OutputDataBase64Chunks = execResult,
					OutputDataType = request.OutputDataType,
					OutputDataLength = request.OutputDataLength,
					ExecutionTimeMs = sw.Elapsed.TotalMilliseconds
				};

				sw.Stop();
				return this.Ok(result);
			}
			catch (ArgumentException argEx)
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request data",
					Detail = argEx.Message,
					Status = 400
				});
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error executing kernel",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("request-generic-execution-batch-base64")]
		[ProducesResponseType(typeof(IEnumerable<string>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 404)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<KernelExecuteResult>> ProcessGenericKernelExecutionBatchBase64Async([FromBody] IEnumerable<string>? base64DataChunks, [FromQuery] string? kernelCode = null, [FromQuery] string? inputDataType = null, [FromQuery] string outputDataLength = "0", [FromQuery] string outputDataType = "object", [FromQuery] IEnumerable<string>? argNames = null, [FromQuery] IEnumerable<string>? argValues = null, [FromQuery] string? deviceName = null, [FromQuery] int? deviceIndex = null, [FromQuery] int workDimension = 1, [FromQuery] int inputDataStride = 1)
		{
			if (string.IsNullOrWhiteSpace(kernelCode))
			{
				return this.BadRequest(new ProblemDetails
				{
					Title = "Invalid request",
					Detail = "Request must include valid Kernel source code",
					Status = 400
				});
			}

			if (!this.cudaService.Initialized)
			{
				if (!string.IsNullOrWhiteSpace(deviceName))
				{
					this.cudaService.Initialize(deviceName);
				}
				else if (deviceIndex.HasValue)
				{
					if (deviceIndex.Value > this.cudaService.Devices.Count)
					{
						return this.NotFound(new ProblemDetails
						{
							Title = "Device index out of range",
							Detail = $"Device index {deviceIndex.Value} is out of range. Available devices: 0 to {this.cudaService.Devices.Count - 1}.",
							Status = 404
						});
					}
					this.cudaService.Initialize(deviceIndex.Value);
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

			var request = new KernelExecuteRequest
			{
				KernelCode = kernelCode,
				InputDataBase64Chunks = base64DataChunks?.ToArray(),
				InputDataType = inputDataType,
				OutputDataLength = outputDataLength,
				OutputDataType = outputDataType,
				ArgumentNames = argNames ?? [],
				ArgumentValues = argValues ?? [],
				DeviceName = deviceName,
				DeviceIndex = deviceIndex,
				WorkDimension = workDimension
			};

			return await this.ProcessGenericKernelExecutionBatchAsync(request);
		}

		[HttpPost("serialize-as-objects")]
		[ProducesResponseType(typeof(IEnumerable<object>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<IEnumerable<object>>> SerializeAsObjectsAsync([FromBody] string base64Data, [FromQuery] string typeName)
		{
			try
			{
				object[] result = await CudaService.ConvertStringToTypeAsync(base64Data, typeName);
				if (result == null || result.Length == 0)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Serialization failed",
						Detail = "No data could be serialized from the provided input.",
						Status = 400
					});
				}

				return this.Ok(result);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error serializing data",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("serialize-as-objects-enumerable")]
		[ProducesResponseType(typeof(IEnumerable<object[]>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<IEnumerable<object[]>>> SerializeAsObjectsEnumerableAsync([FromBody] IEnumerable<string> base64DataChunks, [FromQuery] string typeName)
		{
			try
			{
				if (base64DataChunks == null || !base64DataChunks.Any())
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Invalid request",
						Detail = "No data chunks provided for serialization.",
						Status = 400
					});
				}

				List<object[]> results = [];

				foreach (var chunk in base64DataChunks)
				{
					object[] chunkResult = await CudaService.ConvertStringToTypeAsync(chunk, typeName);
					if (chunkResult != null && chunkResult.Length > 0)
					{
						results.Add(chunkResult);
					}
				}
				if (results.Count == 0)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Serialization failed",
						Detail = "No data could be serialized from the provided input chunks.",
						Status = 400
					});
				}

				return this.Ok(results);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error serializing data",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("serialize-as-base64")]
		[ProducesResponseType(typeof(string), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<string>> SerializeAsBase64Async([FromBody] IEnumerable<object> data, [FromQuery] string? typeName = null)
		{
			try
			{
				if (data == null || !data.Any())
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Invalid request",
						Detail = "No data provided for serialization.",
						Status = 400
					});
				}

				string? result = await CudaService.ConvertTypeToStringAsync(data.ToArray(), typeName);
				if (string.IsNullOrWhiteSpace(result))
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Serialization failed",
						Detail = "No data could be serialized from the provided input.",
						Status = 400
					});
				}

				return this.Ok(result);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error serializing data",
					Detail = ex.Message,
					Status = 500
				});
			}
		}

		[HttpPost("serialize-as-base64-enumerable")]
		[ProducesResponseType(typeof(IEnumerable<string>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 400)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<IEnumerable<string>>> SerializeAsBase64EnumerableAsync([FromBody] IEnumerable<IEnumerable<object>> dataChunks, [FromQuery] string? typeName = null)
		{
			try
			{
				if (dataChunks == null || !dataChunks.Any())
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Invalid request",
						Detail = "No data chunks provided for serialization.",
						Status = 400
					});
				}

				List<string> results = [];
				foreach (var chunk in dataChunks)
				{
					string? chunkResult = await CudaService.ConvertTypeToStringAsync(chunk.ToArray(), typeName);
					if (!string.IsNullOrWhiteSpace(chunkResult))
					{
						results.Add(chunkResult);
					}
				}
				if (results.Count == 0)
				{
					return this.BadRequest(new ProblemDetails
					{
						Title = "Serialization failed",
						Detail = "No data could be serialized from the provided input chunks.",
						Status = 400
					});
				}

				return this.Ok(results);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Title = "Error serializing data",
					Detail = ex.Message,
					Status = 500
				});
			}
		}


	}
}
