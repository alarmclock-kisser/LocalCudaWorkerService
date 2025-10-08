using LocalCudaWorkerService.Runtime;
using LocalCudaWorkerService.Shared;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.HttpOverrides;
using Microsoft.OpenApi.Models;
using System.Net;
using System.Net.Sockets;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;

namespace LocalCudaWorkerService.Api
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // CUDA Konfiguration
            var externalUrl = builder.Configuration.GetValue<string>("ExternalServerAddress", "https://localhost:32141/");
			var cudaDeviceId = builder.Configuration.GetValue<int>("DefaultDeviceIndex", -1);
            var cudaDeviceName = builder.Configuration.GetValue<string>("DefaultDeviceName", "");
            var maxUploadSizeMb = builder.Configuration.GetValue<long>("MaxUploadSizeMb", 128) * 1024 * 1024;

            // Port aus LaunchSettings / Env / Config ermitteln
            int httpsPort = ResolveHttpsPort(builder.Configuration, defaultPort: 32141, out var httpsUrlRaw);

            // Optional: Logging des ermittelten Ports
            builder.Logging.AddConsole().AddDebug();
            builder.Logging.SetMinimumLevel(LogLevel.Debug);
            var loggerFactory = LoggerFactory.Create(logging =>
            {
                logging.AddConsole();
                logging.AddDebug();
                logging.SetMinimumLevel(LogLevel.Debug);
            });
            var startupLogger = loggerFactory.CreateLogger("Startup");
            startupLogger.LogInformation("Resolved HTTPS Port: {Port} (from: {Source})", httpsPort, httpsUrlRaw ?? "(fallback)");

            // Öffentliche (globale) IP oder Fallback lokale IP ermitteln
            var serverIp = await ResolveServerIpAsync(builder.Configuration, startupLogger);

            // CUDA Service
            var cudaService = new CudaService(cudaDeviceId, cudaDeviceName ?? "");
            builder.Services.AddSingleton(cudaService);

            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen(o =>
            {
                o.SwaggerDoc("v1", new OpenApiInfo
                {
                    Version = "v1",
                    Title = "LocalCudaWorkerService API",
                    Description = "ASP.NET REST-API using CUDA as an online worker for executing kernels via HTTP.",
                    TermsOfService = new Uri("https://example.com/terms"),
                    Contact = new OpenApiContact { Name = "github: alarmclock-kisser", Email = "marcel.king91299@gmail.com" }
                });
                o.OperationFilter<PlainTextRequestBodyFilter>();
            });

            builder.WebHost.ConfigureKestrel((context, options) =>
            {
                options.Limits.MaxRequestBodySize = maxUploadSizeMb;

                // Immer extern lauschen (0.0.0.0), unabhängig von ASPNETCORE_URLS (localhost Eintrag)
                options.ListenAnyIP(httpsPort, lo =>
                {
                    lo.UseHttps();
                });

                options.Configure(context.Configuration.GetSection("Kestrel"));
            });

            builder.Services.Configure<IISServerOptions>(o => o.MaxRequestBodySize = maxUploadSizeMb);
            builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = maxUploadSizeMb);

            builder.Services.AddControllers().AddJsonOptions(o =>
            {
                o.JsonSerializerOptions.IncludeFields = true;
            });

            // ApiConfig singleton
            var config = new ApiConfiguration()
            {
                ApplicationName = "LocalCudaWorkerService API",
                LocalServerIp = serverIp,
                LocalServerUrl = (httpsPort > 0) ? $"https://{serverIp}:{httpsPort}/" : "",
                LocalServerPort = httpsPort,
                UseHttps = true,
                ExternalServerAddress = externalUrl,
				MaxUploadSizeMb = (int)(maxUploadSizeMb / (1024 * 1024)),
                DefaultDeviceIndex = cudaDeviceId,
                DefaultDeviceName = cudaDeviceName,
                AdditionalProperties = []
            };
            builder.Services.AddSingleton(config);

            var app = builder.Build();

            app.Lifetime.ApplicationStarted.Register(() =>
            {
                try
                {
                    var server = app.Services.GetRequiredService<IServer>();
                    var feat = server.Features.Get<IServerAddressesFeature>();
                    if (feat != null)
                    {
                        startupLogger.LogInformation("Listening on: {Addresses}", string.Join(", ", feat.Addresses));
                    }
                }
                catch (Exception ex)
                {
                    startupLogger.LogWarning(ex, "Failed to enumerate server addresses.");
                }
            });

            app.UseForwardedHeaders(new ForwardedHeadersOptions
            {
                ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
            });

            app.UseHttpsRedirection();

            app.UseSwagger(c => c.RouteTemplate = "swagger/{documentName}/swagger.json");
            app.UseSwaggerUI(c =>
            {
                c.SwaggerEndpoint("./v1/swagger.json", "LocalCudaWorkerService API v1");
                c.RoutePrefix = "swagger";
                c.DisplayRequestDuration();
            });

            app.MapGet("/swagger/v1/swagger.json", ctx =>
            {
                ctx.Response.Headers.Location = "/api/swagger/v1/swagger.json";
                ctx.Response.StatusCode = StatusCodes.Status302Found;
                return Task.CompletedTask;
            }).ExcludeFromDescription();

            if (app.Environment.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.MapControllers();

			// Try get well formed Url from externalUrl
            if (Uri.TryCreate(externalUrl, UriKind.Absolute, out var extUri))
            {
                app.Lifetime.ApplicationStarted.Register(() =>
                {
                    startupLogger.LogInformation("Configured External Server Address: {Url}", extUri.ToString());
                });
            }
            else
            {
                startupLogger.LogWarning("The configured ExternalServerAddress is not a valid URL: {Url}", externalUrl);
			}

			app.UseStaticFiles();

			app.Run();

            // --- Lokale Hilfsfunktionen ---
            static int ResolveHttpsPort(IConfiguration config, int defaultPort, out string? rawSourceUrl)
            {
                rawSourceUrl = null;

                string? urls =
                    Environment.GetEnvironmentVariable("ASPNETCORE_URLS")
                    ?? Environment.GetEnvironmentVariable("DOTNET_URLS")
                    ?? config["ASPNETCORE_URLS"]
                    ?? config["urls"]
                    ?? config["applicationUrl"]
                    ?? config["ApplicationUrl"];

                if (!string.IsNullOrWhiteSpace(urls))
                {
                    rawSourceUrl = urls;
                    foreach (var part in urls.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    {
                        if (part.StartsWith("https://", StringComparison.OrdinalIgnoreCase)
                            && Uri.TryCreate(part, UriKind.Absolute, out var httpsUri))
                        {
                            if (httpsUri.Port > 0)
                                return httpsUri.Port;
                        }
                    }
                }

                int configured = config.GetValue<int?>("Service:HttpsPort") ?? 0;
                if (configured > 0) return configured;

                return defaultPort;
            }

            static string GetLocalIpAddress()
            {
                string localIP = string.Empty;
                var host = Dns.GetHostEntry(Dns.GetHostName());
                foreach (var ip in host.AddressList)
                {
                    if (ip.AddressFamily == AddressFamily.InterNetwork)
                    {
                        localIP = ip.ToString();
                        break;
                    }
                }
                return localIP;
            }

            static async Task<string> ResolveServerIpAsync(IConfiguration config, ILogger logger, CancellationToken ct = default)
            {
                // 1) Explizite Vorgabe (z.B. via Env oder appsettings: "PublicServerIp")
                var explicitIp = config.GetValue<string>("PublicServerIp");
                if (!string.IsNullOrWhiteSpace(explicitIp))
                {
                    logger.LogInformation("Using configured public IP: {Ip}", explicitIp);
                    return explicitIp.Trim();
                }

                // 2) Öffentliche IP via externe Dienste versuchen
                string[] endpoints =
                {
                    "https://api.ipify.org",
                    "https://checkip.amazonaws.com",
                    "https://ifconfig.me/ip"
                };

                using var http = new HttpClient() { Timeout = TimeSpan.FromSeconds(10) };
                foreach (var url in endpoints)
                {
                    try
                    {
                        var txt = (await http.GetStringAsync(url, ct)).Trim();
                        if (IPAddress.TryParse(txt, out var ip) && ip.AddressFamily == AddressFamily.InterNetwork)
                        {
                            logger.LogInformation("Detected public IP via {Endpoint}: {Ip}", url, txt);
                            return txt;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogDebug(ex, "Public IP fetch failed at {Url}", url);
                    }
                }

                // 3) Fallback lokale IP
                var local = GetLocalIpAddress();
                logger.LogWarning("Could not determine public IP. Falling back to local IP {Ip}", local);
                return local;
            }
        }
    }
}
