using LocalCudaWorkerService.Runtime;
using LocalCudaWorkerService.Shared;
using Microsoft.AspNetCore.Mvc;
using System.Net; // hinzugefügt

namespace LocalCudaWorkerService.Api.Controllers
{
	[ApiController]
	[Route("api/[controller]")]
	public class WorkerController : ControllerBase
	{
		private readonly ApiConfiguration apiConfig;

		public WorkerController(ApiConfiguration apiConfig)
		{
			this.apiConfig = apiConfig;
		}

		[HttpGet("info", Name = "GetInfo")]
		[HttpGet("config", Name = "GetConfig")]
		[ProducesResponseType(typeof(ApiConfiguration), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public ActionResult<ApiConfiguration> GetConfig()
		{
			try
			{
				return this.Ok(this.apiConfig);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "Internal Server Error",
					Detail = ex.Message
				});
			}
		}

		[HttpGet("connect-to-server")]
		[ProducesResponseType(typeof(string), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<string>> ConnectToServerAsync([FromQuery] string? overwriteServerUrl = null)
		{
			overwriteServerUrl ??= this.apiConfig.ExternalServerAddress;

			if (string.IsNullOrWhiteSpace(overwriteServerUrl))
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "External Server URL missing",
					Detail = "No external server URL provided."
				});
			}

			try
			{
				using var httpClient = new HttpClient();

				// 1) Connectivity-Check (toleriert 404 am Root)
				HttpResponseMessage? response = null;
				try
				{
					response = await httpClient.GetAsync(overwriteServerUrl);
					if (!response.IsSuccessStatusCode && response.StatusCode != HttpStatusCode.NotFound)
					{
						return this.StatusCode((int)response.StatusCode, new ProblemDetails
						{
							Status = (int)response.StatusCode,
							Title = "Connectivity check failed",
							Detail = $"GET {overwriteServerUrl} returned {(int)response.StatusCode} {response.StatusCode}."
						});
					}
				}
				catch (HttpRequestException hre)
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Status = 500,
						Title = "Connectivity error",
						Detail = hre.Message
					});
				}

				// Bei Erfolg oder toleriertem 404: Status aktualisieren
				this.apiConfig.ExternalServerAddress = overwriteServerUrl;
				this.apiConfig.SuccessfullyConnectedToExternalServer = true;

				var content = response != null
					? await SafeReadAsync(response)
					: string.Empty;

				// 2) Registrierung – beide Varianten testen (mit und ohne /api)
				var baseUrlRaw = (this.apiConfig.ExternalServerAddress ?? "").Trim().TrimEnd('/');

				// Basis ohne doppelte Segmente normalisieren
				string[] stripSuffixes =
				{
					"/api/ExternalCuda",
					"/ExternalCuda",
					"/api"
				};
				foreach (var suf in stripSuffixes)
				{
					if (baseUrlRaw.EndsWith(suf, StringComparison.OrdinalIgnoreCase))
					{
						baseUrlRaw = baseUrlRaw[..^suf.Length].TrimEnd('/');
						break;
					}
				}

				var workerAddress = (this.apiConfig.LocalServerUrl ?? "").TrimEnd('/');
				if (string.IsNullOrWhiteSpace(workerAddress))
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Status = 500,
						Title = "Local worker address missing",
						Detail = "apiConfig.LocalServerUrl is empty."
					});
				}

				var candidates = new[]
				{
					$"{baseUrlRaw}/api/api/ExternalCuda/register",
					$"{baseUrlRaw}/api/ExternalCuda/register",          // normal (Controller mit api/)
					$"{baseUrlRaw}/ExternalCuda/register"       // doppelt (aktueller Zustand)
				};

				var attempted = new List<string>();
				HttpResponseMessage? registerResponse = null;
				string? usedUrl = null;

				// Insecure Handler (nur vorübergehend!)
				var insecureHandler = new HttpClientHandler
				{
					ServerCertificateCustomValidationCallback = (_, _, _, _) => true
				};
				using var insecureClient = new HttpClient(insecureHandler);

				foreach (var url in candidates.Distinct())
				{
					try
					{
						var jsonBody = new StringContent($"\"{workerAddress}\"", System.Text.Encoding.UTF8, "application/json");
						var resp = await insecureClient.PostAsync(url, jsonBody);

						if ((int)resp.StatusCode == 400 || resp.StatusCode == HttpStatusCode.UnsupportedMediaType)
						{
							// Plain Text Fallback
							var plainBody = new StringContent(workerAddress, System.Text.Encoding.UTF8, "text/plain");
							resp = await insecureClient.PostAsync(url, plainBody);
						}

						attempted.Add($"{url} -> {(int)resp.StatusCode}");

						if (resp.StatusCode != HttpStatusCode.OK)
						{
							continue; // nächste Variante probieren
						}

						// Route existiert (kein 404) -> Ergebnis verwenden
						registerResponse = resp;
						usedUrl = url;
						break;
					}
					catch (Exception ex)
					{
						attempted.Add($"{url} -> EX: {ex.GetType().Name}");
					}
				}

				this.apiConfig.AdditionalProperties["RegisterAttempts"] = string.Join(" | ", attempted);

				if (registerResponse == null)
				{
					this.apiConfig.RegisteredAtExternalServer = false;
					return this.StatusCode(404, new ProblemDetails
					{
						Status = 404,
						Title = "Registration Failed",
						Detail = $"All variants 404. Tried: {string.Join(", ", candidates)}"
					});
				}

				this.apiConfig.AdditionalProperties["LastRegisterUrl"] = usedUrl ?? "";
				this.apiConfig.AdditionalProperties["LastRegisterStatus"] = ((int)registerResponse.StatusCode).ToString();

				if (registerResponse.IsSuccessStatusCode)
				{
					this.apiConfig.RegisteredAtExternalServer = true;

					// Body versuchen als bool zu interpretieren (text/plain oder json-string)
					bool successValue = true;
					try
					{
						var body = await registerResponse.Content.ReadAsStringAsync();
						if (!string.IsNullOrWhiteSpace(body))
						{
							var trimmed = body.Trim().Trim('"');
							if (bool.TryParse(trimmed, out var parsed))
							{
								successValue = parsed;
							}
							this.apiConfig.AdditionalProperties["LastRegisterResponseBody"] = trimmed;
						}
					}
					catch { /* Ignorieren, default true */ }

					return this.Ok(successValue);
				}
				else
				{
					this.apiConfig.RegisteredAtExternalServer = false;
					var errorContent = await SafeReadAsync(registerResponse);
					return this.StatusCode((int) registerResponse.StatusCode, new ProblemDetails
					{
						Status = (int) registerResponse.StatusCode,
						Title = "Registration Failed",
						Detail = $"POST {usedUrl} -> {(int) registerResponse.StatusCode} {registerResponse.StatusCode}. Response: {errorContent} | Attempts: {string.Join(" | ", attempted)}"
					});
				}
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "Internal Server Error",
					Detail = ex.Message
				});
			}

			static async Task<string> SafeReadAsync(HttpResponseMessage msg)
			{
				try
				{
					return (await msg.Content.ReadAsStringAsync()).Trim();
				}
				catch
				{
					return string.Empty;
				}
			}
		}

		[HttpDelete("unregister")]
		[ProducesResponseType(typeof(bool), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public async Task<ActionResult<bool>> UnregisterAsync()
		{
			if (string.IsNullOrWhiteSpace(this.apiConfig.ExternalServerAddress))
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "External Server URL missing",
					Detail = "No external server URL provided."
				});
			}

			if (!this.apiConfig.RegisteredAtExternalServer)
			{
				return this.StatusCode(400, new ProblemDetails
				{
					Status = 400,
					Title = "Not registered",
					Detail = "The worker is not registered at the external server."
				});
			}

			try
			{
				using var httpClient = new HttpClient();
				var baseUrlRaw = (this.apiConfig.ExternalServerAddress ?? "").Trim().TrimEnd('/');

				// Basis ohne doppelte Segmente normalisieren (wie bei Register)
				string[] stripSuffixes =
				{
					"/api/ExternalCuda",
					"/ExternalCuda",
					"/api"
				};
				foreach (var suf in stripSuffixes)
				{
					if (baseUrlRaw.EndsWith(suf, StringComparison.OrdinalIgnoreCase))
					{
						baseUrlRaw = baseUrlRaw[..^suf.Length].TrimEnd('/');
						break;
					}
				}

				var workerAddress = (this.apiConfig.LocalServerUrl ?? "").TrimEnd('/');
				if (string.IsNullOrWhiteSpace(workerAddress))
				{
					return this.StatusCode(500, new ProblemDetails
					{
						Status = 500,
						Title = "Local worker address missing",
						Detail = "apiConfig.LocalServerUrl is empty."
					});
				}

				// Try variants (prefer single /api first)
				var candidates = new[]
				{
					$"{baseUrlRaw}/api/ExternalCuda/unregister",
					$"{baseUrlRaw}/ExternalCuda/unregister",
					$"{baseUrlRaw}/api/api/ExternalCuda/unregister"
				}.Distinct();

				var attempted = new List<string>();
				HttpResponseMessage? successResponse = null;
				string? usedUrl = null;

				// Allow DELETE with body by creating HttpRequestMessage
				foreach (var url in candidates)
				{
					try
					{
						// JSON body first
						var jsonBody = new StringContent($"\"{workerAddress}\"", System.Text.Encoding.UTF8, "application/json");
						using var req = new HttpRequestMessage(HttpMethod.Delete, url) { Content = jsonBody };
						var resp = await httpClient.SendAsync(req);

						// If server rejects JSON with 400/UnsupportedMediaType, try text/plain
						if ((int) resp.StatusCode == 400 || resp.StatusCode == System.Net.HttpStatusCode.UnsupportedMediaType)
						{
							var plainBody = new StringContent(workerAddress, System.Text.Encoding.UTF8, "text/plain");
							using var req2 = new HttpRequestMessage(HttpMethod.Delete, url) { Content = plainBody };
							resp = await httpClient.SendAsync(req2);
						}

						attempted.Add($"{url} -> {(int) resp.StatusCode}");

						if (resp.IsSuccessStatusCode)
						{
							successResponse = resp;
							usedUrl = url;
							break;
						}
					}
					catch (Exception ex)
					{
						attempted.Add($"{url} -> EX: {ex.GetType().Name}");
					}
				}

				this.apiConfig.AdditionalProperties["UnregisterAttempts"] = string.Join(" | ", attempted);

				if (successResponse == null)
				{
					this.apiConfig.RegisteredAtExternalServer = true; // keep unchanged, unregister failed
					return this.StatusCode(404, new ProblemDetails
					{
						Status = 404,
						Title = "Unregistration Failed",
						Detail = $"All variants failed. Tried: {string.Join(", ", candidates)} | Attempts: {string.Join(" | ", attempted)}"
					});
				}

				// Erfolg
				this.apiConfig.RegisteredAtExternalServer = false;
				this.apiConfig.AdditionalProperties["LastUnregisterUrl"] = usedUrl ?? "";
				return this.Ok(true);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "Internal Server Error",
					Detail = ex.Message
				});
			}
		}

		[HttpGet("api-log")]
		[ProducesResponseType(typeof(IEnumerable<string>), 200)]
		[ProducesResponseType(typeof(ProblemDetails), 500)]
		public ActionResult<IEnumerable<string>> GetApiLog([FromQuery] int? maxEntries = null)
		{
			try
			{
				var log = CudaService.LogEntries.ToList();

				if (maxEntries.HasValue && maxEntries.Value > 0)
				{
					log = log.TakeLast(maxEntries.Value).ToList();
				}

				return this.Ok(log);
			}
			catch (Exception ex)
			{
				return this.StatusCode(500, new ProblemDetails
				{
					Status = 500,
					Title = "Internal Server Error",
					Detail = ex.Message
				});
			}
		}
	}
}
