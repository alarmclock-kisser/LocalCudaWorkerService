using LocalCudaWorkerService.Client;
using LocalCudaWorkerService.Shared;
using Microsoft.JSInterop;
using Radzen;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.WebApp.Pages
{
    public class StatusViewModel
    {
        private readonly LocalApiClient Api;
        private readonly AppConfiguration Config;
        private readonly NotificationService Notifications;
        private readonly IJSRuntime JS;
        private readonly DialogService DialogService;

        public StatusViewModel(LocalApiClient api, AppConfiguration config, NotificationService notifications, IJSRuntime js, DialogService dialogService)
        {
            this.Api = api ?? throw new ArgumentNullException(nameof(api));
            this.Config = config ?? throw new ArgumentNullException(nameof(config));
            this.Notifications = notifications ?? throw new ArgumentNullException(nameof(notifications));
            this.JS = js ?? throw new ArgumentNullException(nameof(js));
            this.DialogService = dialogService ?? throw new ArgumentNullException(nameof(dialogService));

            this.ErrorMessages = new List<string>();
            this.LogMessages = new List<string>();
        }

        public bool IsLoading { get; set; } = false;

        // Indicates that the local API is reachable (config retrieval succeeded)
        public bool IsOnline { get; set; } = false;

        // Indicates whether the local API has successfully connected to the external server
        public bool IsConnectedToExternalServer { get; set; } = false;

        public AppConfiguration AppConfig => this.Config;

        public string StatusText { get; set; } = "Unknown";
        public bool IsRegistered { get; set; } = false;
        public long LatencyMs { get; set; } = 0;
        public string ButtonText { get; set; } = "Register";
        // ButtonColor: CUDA-green only when registered AND connected to external server; otherwise gray
        public string ButtonColor => (this.IsRegistered && this.IsConnectedToExternalServer) ? "#76B900" : "#DDDDDD";
        public string LastChecked { get; set; } = "-";

        // This field can be used as overwriteServerUrl in RegisterAsync
        public string ServerUrlText { get; set; } = string.Empty;

        public List<string> ErrorMessages { get; set; }
        public List<string> LogMessages { get; set; }

        public DateTime LastCheckedUtc { get; set; } = DateTime.MinValue;

        public bool CanRegister => !this.IsLoading && !this.IsRegistered && (this.IsOnline || !string.IsNullOrWhiteSpace(this.ServerUrlText));

        public async Task InitializeAsync()
        {
			// Prefer external server address from API config (the worker knows it).
			try
            {
                var cfg = await this.Api.GetApiConfigAsync();
                if (cfg != null && string.IsNullOrWhiteSpace(cfg.ErrorMessage))
                {
                    if (!string.IsNullOrWhiteSpace(cfg.ExternalServerAddress))
                    {
                        this.ServerUrlText = cfg.ExternalServerAddress.Trim().TrimEnd('/');
                      }
                    else
                    {
                        // fallback to app configuration LocalApiUrl (not ideal but better than empty)
                        var cfgUrl = this.Config.LocalApiUrl?.Trim() ?? string.Empty;
                        if (!string.IsNullOrWhiteSpace(cfgUrl))
                        {
                            if (Uri.TryCreate(cfgUrl, UriKind.Absolute, out var parsed))
                            {
                                if (parsed.IsDefaultPort && this.Config.LocalApiPort > 0)
                                {
                                    var ub = new UriBuilder(parsed) { Port = this.Config.LocalApiPort };
                                    this.ServerUrlText = ub.Uri.ToString().TrimEnd('/');
                                }
                                else
                                {
                                    this.ServerUrlText = parsed.ToString().TrimEnd('/');
                                }
                            }
                            else
                            {
                                var trimmed = cfgUrl.TrimEnd('/');
                                if (this.Config.LocalApiPort > 0 && !trimmed.Contains(":"))
                                {
                                    this.ServerUrlText = trimmed + ":" + this.Config.LocalApiPort;
                                }
                                else
                                {
                                    this.ServerUrlText = trimmed;
                                }
                            }
                        }
                        else
                        {
                            this.ServerUrlText = string.Empty;
                        }
                    }

					Console.WriteLine("Using server URL from API config: " + this.ServerUrlText);
				}
                else
                {
                    // if config failed, still attempt to seed from AppConfiguration
                    var cfgUrl = this.Config.LocalApiUrl?.Trim() ?? string.Empty;
                    if (!string.IsNullOrWhiteSpace(cfgUrl))
                    {
                        this.ServerUrlText = cfgUrl;
                    }
                }
            }
            catch
            {
                this.ServerUrlText = string.Empty;
            }

            await this.RefreshStatusAsync();
        }

        public async Task RefreshStatusAsync()
        {
            this.IsLoading = true;
            try
            {
                var start = DateTime.UtcNow;

                var config = await this.Api.GetApiConfigAsync();

                // If config retrieved, local API is reachable
                this.IsOnline = config != null && string.IsNullOrWhiteSpace(config.ErrorMessage);

                // Update registration/connectivity flags from API config if available
                if (config != null && string.IsNullOrWhiteSpace(config.ErrorMessage))
                {
                    this.IsRegistered = config.RegisteredAtExternalServer;
                    this.IsConnectedToExternalServer = config.SuccessfullyConnectedToExternalServer;
                }
                else
                {
                    this.IsRegistered = false;
                    this.IsConnectedToExternalServer = false;
                }

                // Try to get logs, but tolerate failures
                try
                {
                    var logs = await this.Api.GetApiLogAsync(this.AppConfig.MaxLogEntries);
                    this.LogMessages = logs?.ToList() ?? new List<string>();
                }
                catch
                {
                    // ignore log retrieval errors
                }

                var end = DateTime.UtcNow;
                this.LatencyMs = (long)(end - start).TotalMilliseconds;
                this.LastCheckedUtc = end;
                this.LastChecked = end.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss");

                // Update status text and button text
                if (this.IsRegistered)
                {
                    if (this.IsConnectedToExternalServer)
                    {
                        this.StatusText = "Registered and Online";
                        this.ButtonText = "Un-Register";
                    }
                    else
                    {
                        this.StatusText = "Registered but External Offline";
                        this.ButtonText = "Re-Register";
                    }
                }
                else
                {
                    this.StatusText = "Not Registered";
                    this.ButtonText = "Register";
                }
            }
            catch (Exception ex)
            {
                this.StatusText = $"Error: {ex.Message}";
                this.IsRegistered = false;
                this.IsConnectedToExternalServer = false;
                this.ButtonText = "Retry";
                this.IsOnline = false;
                this.Notifications.Notify(new NotificationMessage
                {
                    Severity = NotificationSeverity.Error,
                    Summary = "Error",
                    Detail = ex.Message,
                    Duration = 8000
                });
            }
            finally
            {
				Console.WriteLine("Refreshed! IsRegistered: " + this.IsRegistered + ", IsConnectedToExternalServer: " + this.IsConnectedToExternalServer + ", IsOnline: " + this.IsOnline);

				this.IsLoading = false;
            }
        }

        public async Task<bool> RegisterAsync(string? overwriteServerUrl = null)
        {
            var result = await this.Api.ConnectToServerAsync(overwriteServerUrl ?? this.ServerUrlText);

            if (result != null && result.StartsWith("Error:", StringComparison.OrdinalIgnoreCase))
            {
                this.ErrorMessages.Add(result);
                this.StatusText = result;
                this.Notifications.Notify(new NotificationMessage
                {
                    Severity = NotificationSeverity.Error,
                    Summary = "Registration failed",
                    Detail = result,
                    Duration = 8000
                });

				Console.WriteLine("Registration failed: " + result);
				await this.RefreshStatusAsync();
                return false;
            }

            // Erfolg
            this.Notifications.Notify(new NotificationMessage
            {
                Severity = NotificationSeverity.Success,
                Summary = "Registered",
                Detail = result ?? "Registered",
                Duration = 4000
            });

            // Kleines Delay, damit Backend Zeit hat, Status zu persistieren (reduziert Race)
            await Task.Delay(300);
            Console.WriteLine("Registered successfully: " + result);
			await this.RefreshStatusAsync();
            return true;
        }

        public async Task<bool> UnregisterAsync()
        {
            try
            {
                var ok = await this.Api.UnregisterFromServerAsync();
                if (ok)
                {
                    this.Notifications.Notify(new NotificationMessage
                    {
                        Severity = NotificationSeverity.Success,
                        Summary = "Unregistered",
                        Detail = "Worker successfully unregistered from external server.",
                        Duration = 4000
                    });

                    // Aktualisiere Status aus API (autoritative Quelle)
                    await Task.Delay(300);

					Console.WriteLine("Unregistered successfully.");
					await this.RefreshStatusAsync();
                    return true;
                }
                else
                {
                    this.Notifications.Notify(new NotificationMessage
                    {
                        Severity = NotificationSeverity.Error,
                        Summary = "Unregister failed",
                        Detail = "Unregister endpoint returned an error or was unreachable.",
                        Duration = 8000
                    });

                    Console.WriteLine("Unregister failed.");
					await this.RefreshStatusAsync();
                    return false;
                }
            }
            catch (Exception ex)
            {
                this.Notifications.Notify(new NotificationMessage
                {
                    Severity = NotificationSeverity.Error,
                    Summary = "Unregister error",
                    Detail = ex.Message,
                    Duration = 8000
                });

                Console.WriteLine("Unregister exception: " + ex.Message);
				await this.RefreshStatusAsync();
                return false;
            }
        }
    }
}
