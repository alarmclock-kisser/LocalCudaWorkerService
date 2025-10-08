using LocalCudaWorkerService.Shared;
using LocalCudaWorkerService.Shared.Cuda;
using Microsoft.AspNetCore.Components.Forms;
using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Client
{
	public class LocalApiClient
	{
		private readonly InternalClient internalClient;
		private readonly HttpClient httpClient;
		private readonly string baseUrl;

		private ApiConfiguration? cachedConfig = null;

		public string BaseUrl => this.baseUrl;

		public LocalApiClient(string baseUrl)
		{
			this.baseUrl = baseUrl.TrimEnd('/');
			this.httpClient = new HttpClient()
			{
				Timeout = TimeSpan.FromSeconds(30),
				BaseAddress = new Uri(this.baseUrl)
			};

			this.internalClient = new InternalClient(this.baseUrl, this.httpClient);
		}




		// HttpClient Methods
		public async Task<bool?> IsRegistered()
		{
			var config = this.cachedConfig;
			if (config == null)
			{
				config = await this.GetApiConfigAsync();
			}
			if (!string.IsNullOrWhiteSpace(config.ErrorMessage))
			{
				throw new Exception(config.ErrorMessage);
			}

			return (await this.GetRawBoolAsync(config.ExternalServerAddress, "api/externalcuda/is-worker-registered"));
		}

		public async Task<bool?> IsOnlineInitialized()
		{
			var config = this.cachedConfig;
			if (config == null)
			{
				config = await this.GetApiConfigAsync();
			}
			if (!string.IsNullOrWhiteSpace(config.ErrorMessage))
			{
				throw new Exception(config.ErrorMessage);
			}

			return (await this.GetRawBoolAsync(config.ExternalServerAddress, "api/externalcuda/is-worker-initialized"));
		}


		// HttpClient Helper Methods (raw endpoints)
		private async Task<string> GetRawAsync(string? endpoint, string contoller)
		{
			if (string.IsNullOrEmpty(endpoint))
			{
				var apiCfg = await this.GetApiConfigAsync();
				if (!string.IsNullOrWhiteSpace(apiCfg.ErrorMessage))
				{
					return $"Error: {apiCfg.ErrorMessage}";
				}

				endpoint = this.baseUrl;
			}

			if (!endpoint.EndsWith('/'))
			{
				endpoint += '/';
			}
			endpoint += contoller.TrimStart('/');

			var response = await this.httpClient.GetAsync(endpoint);
			response.EnsureSuccessStatusCode();
			return await response.Content.ReadAsStringAsync();
		}

		private async Task<bool?> GetRawBoolAsync(string? endpoint, string contoller)
		{
			var responseString = await this.GetRawAsync(endpoint, contoller);
			if (responseString.StartsWith("Error:"))
			{
				throw new Exception(responseString);
			}
			if (bool.TryParse(responseString, out var result))
			{
				return result;
			}
			else
			{
				return null;
			}
		}


		// Config & Connection
		public async Task<ApiConfiguration> GetApiConfigAsync()
		{
			try
			{
				this.cachedConfig = await this.internalClient.GetConfigAsync();
				return this.cachedConfig;
			}
			catch (Exception ex)
			{
				return new ApiConfiguration
				{
					 ErrorMessage = ex.Message
				};
			}
		}

		public async Task<string> ConnectToServerAsync(string? overwriteServerUrl = null)
		{
			try
			{
				return await this.internalClient.ConnectToServerAsync(overwriteServerUrl);
			}
			catch (Exception ex)
			{
				return $"Error: {ex.Message}" + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty);
			}
		}

		public async Task<bool> UnregisterFromServerAsync()
		{
			try
			{
				await this.DisposeCudaAsync();
				return await this.internalClient.UnregisterAsync();
			}
			catch (Exception ex)
			{
				Console.WriteLine("Client: Defaulted to true UnregisterFromServerAsync(), Exception: ");
				Console.WriteLine(ex);
				return true;
			}
		}

		public async Task<IEnumerable<string>> GetApiLogAsync(int? maxEntries = null)
		{
			try
			{
				return await this.internalClient.ApiLogAsync(maxEntries);
			}
			catch (Exception ex)
			{
				return ["Error: " + ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)];
			}
		}


		// Cuda + initialization
		public async Task<CudaStatusInfo> GetCudaStatusAsync()
		{
			try
			{
				return await this.internalClient.StatusAsync();
			}
			catch (Exception ex)
			{
				return new CudaStatusInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<IEnumerable<string>> GetCudaDeviceNamesAsync()
		{
			try
			{
				return await this.internalClient.DevicesAsync();
			}
			catch (Exception ex)
			{
				return ["Error: " + ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)];
			}
		}

		public async Task<CudaStatusInfo> InitializeCudaByIndexAsync(int deviceIndex)
		{
			try
			{
				return await this.internalClient.InitializeIndexAsync(deviceIndex);
			}
			catch (Exception ex)
			{
				return new CudaStatusInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<CudaStatusInfo> InitializeCudaByNameAsync(string deviceName)
		{
			try
			{
				return await this.internalClient.InitializeNameAsync(deviceName);
			}
			catch (Exception ex)
			{
				return new CudaStatusInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<CudaStatusInfo> DisposeCudaAsync()
		{
			try
			{
				return await this.internalClient.DisposeAsync();
			}
			catch (Exception ex)
			{
				return new CudaStatusInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}


		// Kernel compilation
		public async Task<CudaKernelInfo> CompileStringAsync(string kernelCode, string? deviceName = null)
		{
			try
			{
				return await this.internalClient.CompileRawAsync(deviceName, kernelCode);
			}
			catch (Exception ex)
			{
				return new CudaKernelInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<CudaKernelInfo> CompileFileAsync(FileParameter fileParameter, string? deviceName = null)
		{
			try
			{
				return await this.internalClient.CompileFileAsync(deviceName, fileParameter);
			}
			catch (Exception ex)
			{
				return new CudaKernelInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<CudaKernelInfo> CompileFileAsync(IBrowserFile browserFile)
		{
			try
			{
				using var stream = browserFile.OpenReadStream(browserFile.Size);
				using var ms = new MemoryStream();
				await stream.CopyToAsync(ms);
				var fileParameter = new FileParameter(stream, browserFile.Name, browserFile.ContentType);
				return await this.internalClient.CompileFileAsync(null, fileParameter);
			}
			catch (Exception ex)
			{
				return new CudaKernelInfo
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}


		// Kernel execution (DTO)
		public async Task<CuFftResult> ExecuteCuFftAsync(CuFftRequest request)
		{
			try
			{
				return await this.internalClient.RequestCufftAsync(request);
			}
			catch (Exception ex)
			{
				return new CuFftResult
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<KernelExecuteResult> ExecuteGenericSingleKernelAsync(KernelExecuteRequest request)
		{
			try
			{
				return await this.internalClient.RequestGenericExecutionSingleAsync(request);
			}
			catch (Exception ex)
			{
				return new KernelExecuteResult
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}

		public async Task<KernelExecuteResult> ExecuteGenericBatchKernelAsync(KernelExecuteRequest request)
		{
			try
			{
				return await this.internalClient.RequestGenericExecutionBatchAsync(request);
			}
			catch (Exception ex)
			{
				return new KernelExecuteResult
				{
					ErrorMessage = ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)
				};
			}
		}


		// Kernel execution (raw/base64)
		// -----


		// Data serialization
		public async Task<object[]> SerializeAsObjectArrayAsync(string base64Data, string dataType)
		{
			try
			{
				return (await this.internalClient.SerializeAsObjectsAsync(dataType, base64Data)).ToArray();
			}
			catch (Exception ex)
			{
				return [ $"Error: {ex.Message}" + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty) ];
			}
		}

		public async Task<object[]> SerializeAsObjectArrayArrayAsync(IEnumerable<string> base64Chunks, string dataType)
		{
			try
			{
				return (await this.internalClient.SerializeAsObjectsEnumerableAsync(dataType, base64Chunks)).ToArray();
			}
			catch (Exception ex)
			{
				return [$"Error: {ex.Message}" + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)];
			}
		}

		public async Task<string> SerializeAsBase64Async(object[] data, string dataType = "")
		{
			try
			{
				return (await this.internalClient.SerializeAsBase64Async(dataType, data));
			}
			catch (Exception ex)
			{
				return $"Error: {ex.Message}" + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty);
			}
		}

		public async Task<IEnumerable<string>> SerializeAsBase64ArrayAsync(IEnumerable<object[]> dataChunks, string dataType = "")
		{
			try
			{
				return await this.internalClient.SerializeAsBase64EnumerableAsync(dataType, dataChunks);
			}
			catch (Exception ex)
			{
				return ["Error: " + ex.Message + (!string.IsNullOrWhiteSpace(ex.InnerException?.Message) ? $" ({ex.InnerException?.Message})" : string.Empty)];
			}
		}

	}
}
